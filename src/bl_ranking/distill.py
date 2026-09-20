"""Distil the hosted TabPFN payout model into a local surrogate.

Hosted latency is flat in batch size (4.83 s for 11 rows, 4.72 s for 3,000),
so a synchronous hosted call cannot reach a page-load budget -- but offline
labelling is nearly free. The real run labelled 25,000 rows in 9 calls and
31.7 s; the resulting CatBoost regressor answers in ~2 ms.

The teacher set is cross-joined (user x every brand) because that is the shape
serving scores; the vendored payout context holds only rows that produced a
payout, so a surrogate trained on it would be blind where serving mostly asks.

The report leads with agreement on the brand *order*, not payout MAE: the
funnel consumes an order, and mostly its top.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("bl_ranking.distill")

# A call costs the same at 250 rows as at 3,000 (4.34 s vs 4.72 s), so batch
# big: round trips are the only cost, and fewer stay inside the rate limit.
TEACHER_BATCH_ROWS = 3000


@dataclass
class DistillationReport:
    n_users: int = 0
    n_rows: int = 0
    n_teacher_calls: int = 0
    teacher_seconds: float = 0.0
    surrogate_fit_seconds: float = 0.0
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_users": self.n_users,
            "n_rows": self.n_rows,
            "n_teacher_calls": self.n_teacher_calls,
            "teacher_seconds": round(self.teacher_seconds, 2),
            "surrogate_fit_seconds": round(self.surrogate_fit_seconds, 2),
            **self.metrics,
        }


def build_serving_grid(
    feature_rows: pd.DataFrame,
    brands: list[str],
    columns: list[str],
    n_users: int,
    seed: int = 42,
) -> pd.DataFrame:
    """Users x every brand — the exact shape a /predict call produces."""
    base = feature_rows.drop(columns=["client_name"], errors="ignore").drop_duplicates()
    if len(base) > n_users:
        base = base.sample(n=n_users, random_state=seed)
    base = base.reset_index(drop=True)
    base["_user_id"] = np.arange(len(base))

    brand_frame = pd.DataFrame({"client_name": [b for b in brands if b != "other"]})
    grid = base.merge(brand_frame, how="cross")
    # Keep _user_id alongside the model columns so rankings can be grouped.
    return grid[["_user_id", *columns]]


def label_with_teacher(
    grid: pd.DataFrame,
    columns: list[str],
    teacher: Any,
    batch_rows: int = TEACHER_BATCH_ROWS,
) -> tuple[np.ndarray, int, float]:
    """Run the hosted model over the grid in as few calls as possible."""
    preds: list[np.ndarray] = []
    calls = 0
    started = time.monotonic()
    for start in range(0, len(grid), batch_rows):
        chunk = grid.iloc[start : start + batch_rows][columns]
        t0 = time.monotonic()
        preds.append(np.asarray(teacher.predict(chunk), dtype=float))
        calls += 1
        logger.info(
            "teacher_batch_scored",
            extra={"rows": len(chunk), "seconds": round(time.monotonic() - t0, 2), "call": calls},
        )
    return np.concatenate(preds), calls, time.monotonic() - started


def train_surrogate(X: pd.DataFrame, y: np.ndarray, columns: list[str], seed: int = 42):
    """CatBoost regressor over the same feature frame the teacher saw."""
    from catboost import CatBoostRegressor

    frame = X[columns]
    cat_features = frame.select_dtypes(include=["object", "category"]).columns.tolist()
    model = CatBoostRegressor(
        random_seed=seed,
        depth=8,
        n_estimators=1200,
        learning_rate=0.06,
        loss_function="RMSE",
        task_type="CPU",
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(frame, y, cat_features=cat_features)
    return model


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr

    corr, _ = spearmanr(a, b)
    return float(corr)


def _kendall_tau(order_a: list[str], order_b: list[str]) -> float:
    """Rank correlation between two orderings of the same brands."""
    from scipy.stats import kendalltau

    pos_b = {b: i for i, b in enumerate(order_b)}
    seq = [pos_b[b] for b in order_a if b in pos_b]
    if len(seq) < 2:
        return float("nan")
    corr, _ = kendalltau(range(len(seq)), seq)
    return float(corr)


def evaluate_ranking_agreement(
    grid: pd.DataFrame,
    columns: list[str],
    teacher_payout: np.ndarray,
    surrogate_payout: np.ndarray,
    cb_model: Any,
) -> dict[str, Any]:
    """Compare the two brand orderings the funnel would render.

    Position is ``P(lead) x payout``, so the classifier has to be in the loop;
    raw payout alone would flatter or penalise depending on its confidence.
    """
    frame = grid[columns]
    p_lead = cb_model.predict_proba(frame)[:, 1]

    work = grid[["_user_id", "client_name"]].copy()
    work["exp_teacher"] = teacher_payout * p_lead
    work["exp_surrogate"] = surrogate_payout * p_lead

    top1_hits = 0
    top3_overlap = 0.0
    taus: list[float] = []
    top1_value_gap: list[float] = []
    gap_agreed: list[float] = []
    gap_disagreed: list[float] = []
    users = 0

    for _, group in work.groupby("_user_id", sort=False):
        ordered = group.sort_values("exp_teacher", ascending=False)
        teacher_order = ordered["client_name"].tolist()
        surrogate_order = group.sort_values("exp_surrogate", ascending=False)["client_name"].tolist()
        if not teacher_order:
            continue
        users += 1

        # How decided was the teacher's own top choice? A disagreement on a
        # near-tie costs almost nothing; one on a clear winner is a real miss.
        # Reporting these separately is what turns "97.9% agreement" from a
        # number into a claim about impact.
        values = ordered["exp_teacher"].to_numpy()
        gap = float((values[0] - values[1]) / values[0]) if len(values) > 1 and values[0] > 0 else float("nan")

        agreed = teacher_order[0] == surrogate_order[0]
        if agreed:
            top1_hits += 1
        if not np.isnan(gap):
            (gap_agreed if agreed else gap_disagreed).append(gap)
        top3_overlap += len(set(teacher_order[:3]) & set(surrogate_order[:3])) / min(3, len(teacher_order))
        taus.append(_kendall_tau(teacher_order, surrogate_order))

        # Revenue view: the teacher's opinion of the brand the surrogate
        # promoted, against the teacher's own best.
        by_brand = group.set_index("client_name")["exp_teacher"]
        best = float(by_brand.max())
        chosen = float(by_brand.get(surrogate_order[0], np.nan))
        if best > 0 and not np.isnan(chosen):
            top1_value_gap.append((best - chosen) / best)

    finite_taus = [t for t in taus if not np.isnan(t)]
    return {
        # Median relative gap between the teacher's #1 and #2, split by whether
        # the surrogate agreed. A much smaller gap on disagreements means the
        # misses land where the choice barely mattered.
        "rank_top1_gap_when_agreed": round(float(np.median(gap_agreed)), 4) if gap_agreed else None,
        "rank_top1_gap_when_disagreed": round(float(np.median(gap_disagreed)), 4) if gap_disagreed else None,
        "rank_top1_agreement": round(top1_hits / users, 4) if users else None,
        "rank_top3_overlap": round(top3_overlap / users, 4) if users else None,
        "rank_kendall_tau_mean": round(float(np.mean(finite_taus)), 4) if finite_taus else None,
        "rank_top1_expected_payout_loss_pct": round(100 * float(np.mean(top1_value_gap)), 3)
        if top1_value_gap
        else None,
        "rank_users_evaluated": users,
    }


def distil_payout_model(
    feature_rows: pd.DataFrame,
    brands: list[str],
    columns: list[str],
    teacher: Any,
    cb_model: Any,
    n_users: int = 2500,
    holdout_frac: float = 0.25,
    seed: int = 42,
) -> tuple[Any, DistillationReport]:
    """Build the teacher set, fit the surrogate, and measure the gap."""
    report = DistillationReport()

    grid = build_serving_grid(feature_rows, brands, columns, n_users=n_users, seed=seed)
    report.n_users = int(grid["_user_id"].nunique())
    report.n_rows = len(grid)
    logger.info("distillation_grid_built", extra={"users": report.n_users, "rows": report.n_rows})

    y_teacher, calls, seconds = label_with_teacher(grid, columns, teacher)
    report.n_teacher_calls = calls
    report.teacher_seconds = seconds

    # Split by user, never by row: sharing a user across the split leaks
    # their payout level and overstates fidelity on new traffic.
    rng = np.random.default_rng(seed)
    user_ids = grid["_user_id"].unique()
    holdout_users = set(rng.choice(user_ids, size=max(1, int(len(user_ids) * holdout_frac)), replace=False).tolist())
    is_holdout = grid["_user_id"].isin(holdout_users).to_numpy()

    t0 = time.monotonic()
    model = train_surrogate(grid[~is_holdout], y_teacher[~is_holdout], columns, seed=seed)
    report.surrogate_fit_seconds = time.monotonic() - t0

    eval_grid = grid[is_holdout].reset_index(drop=True)
    eval_teacher = y_teacher[is_holdout]
    eval_surrogate = np.asarray(model.predict(eval_grid[columns]), dtype=float)

    abs_err = np.abs(eval_surrogate - eval_teacher)
    denom = np.where(np.abs(eval_teacher) < 1e-9, np.nan, np.abs(eval_teacher))
    report.metrics = {
        "teacher": "tabpfn_hosted",
        "holdout_users": len(holdout_users),
        "holdout_rows": int(is_holdout.sum()),
        "payout_mae_vs_teacher": round(float(abs_err.mean()), 4),
        "payout_mape_vs_teacher": round(float(np.nanmean(abs_err / denom)), 4),
        "payout_spearman_vs_teacher": round(_spearman(eval_surrogate, eval_teacher), 4),
        **evaluate_ranking_agreement(eval_grid, columns, eval_teacher, eval_surrogate, cb_model),
    }
    logger.info("distillation_complete", extra=report.metrics)
    return model, report


def save_surrogate(model: Any, report: DistillationReport, columns: list[str], out_dir: Path) -> tuple[Path, Path]:
    from bl_ranking.payout_transport import SURROGATE_FILENAME, SURROGATE_META_FILENAME

    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / SURROGATE_FILENAME
    meta_path = out_dir / SURROGATE_META_FILENAME
    model.save_model(str(model_path), format="cbm")
    meta_path.write_text(json.dumps({"columns": columns, **report.to_dict()}, indent=2))
    return model_path, meta_path
