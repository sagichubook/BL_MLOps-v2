"""Evaluate the brand order, not just the two legs separately.

The vendored ``train_test`` reports classifier precision/recall and payout
MAE. Neither answers the question the business pays for: when we put a brand
first, was it the right brand?

The data records which brand bought, not the candidate set or the order shown,
so a true NDCG is not recoverable. What is: score every brand for each held-out
session as serving would, and ask where the brand that really bought landed.
That gives recall@k and MRR against realised sales -- a lower bound, since a
brand we rank first might also have bought had it been shown, which is why
every key is prefixed ``rankproxy_``.
"""
from __future__ import annotations

import logging
import math
from typing import Any

import pandas as pd

from bl_ranking.distill import TEACHER_BATCH_ROWS, label_with_teacher

logger = logging.getLogger("bl_ranking.ranking_eval")


def wald_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% interval on a proportion.

    Every recall here is a proportion over a finite number of sales, and the
    headline comparison is against a random-ranker baseline that may sit
    inside the interval. Shipping the point estimate alone invites reading
    "0.089 vs 0.100" as a real gap when one week cannot establish one.
    """
    if n <= 0:
        return (float("nan"), float("nan"))
    p = successes / n
    half = z * math.sqrt(max(p * (1 - p), 0.0) / n)
    return (max(0.0, p - half), min(1.0, p + half))


def _session_key(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Identify a session from its features.

    The vendored preprocessing drops ``session_id`` before returning train and
    test frames, so sessions have to be recovered. Every feature except
    ``client_name`` is a property of the session, not of the (session, brand)
    pair, so the tuple of those features is a faithful session identity.
    """
    feature_cols = [c for c in columns if c != "client_name"]
    return frame[feature_cols].astype(str).agg("\x1f".join, axis=1)


def evaluate_ranking(
    x_test: pd.DataFrame,
    y_test: pd.DataFrame,
    brands: list[str],
    columns: list[str],
    cb_model: Any,
    payout_predictor: Any,
    max_sessions: int = 4000,
    seed: int = 42,
) -> dict[str, Any]:
    """Rank every brand for each held-out session; score against real sales.

    Returns a flat dict of MLflow-loggable metrics. All keys are prefixed
    ``rankproxy_`` so nobody mistakes them for a true relevance-judged NDCG.
    """
    brands = [b for b in brands if b != "other"]
    if not brands:
        return {}

    test = x_test.drop(columns=["split_day"], errors="ignore").copy()
    labels = y_test.drop(columns=["split_day"], errors="ignore").copy()
    test = test.reset_index(drop=True)
    labels = labels.reset_index(drop=True)
    test["_session"] = _session_key(test, columns)

    # The ground truth we have: which brand bought, and for how much.
    sold_mask = labels["sold_to_client"].to_numpy() == 1
    truth = pd.DataFrame(
        {
            "_session": test["_session"],
            "client_name": test["client_name"],
            "payout": labels["payout"].to_numpy(),
            "sold": sold_mask,
        }
    )
    truth = truth[truth["sold"]]
    if truth.empty:
        logger.warning("ranking_eval_no_positive_sessions")
        return {}

    # Evaluate on sessions with a known sale; cap for hosted-call budget.
    eval_sessions = truth["_session"].drop_duplicates()
    if len(eval_sessions) > max_sessions:
        eval_sessions = eval_sessions.sample(n=max_sessions, random_state=seed)
    eval_sessions_set = set(eval_sessions)

    session_features = (
        test[test["_session"].isin(eval_sessions_set)]
        .drop_duplicates(subset=["_session"])
        .drop(columns=["client_name"])
        .reset_index(drop=True)
    )
    grid = session_features.merge(pd.DataFrame({"client_name": brands}), how="cross")
    logger.info(
        "ranking_eval_grid", extra={"sessions": len(session_features), "brands": len(brands), "rows": len(grid)}
    )

    payout = label_with_teacher(grid, columns, payout_predictor, batch_rows=TEACHER_BATCH_ROWS)[0]
    p_lead = cb_model.predict_proba(grid[columns])[:, 1]

    scored = grid[["_session", "client_name"]].copy()
    scored["expected_payout"] = payout * p_lead
    scored["rank"] = scored.groupby("_session")["expected_payout"].rank(ascending=False, method="first")

    # Join our rank onto each realised sale.
    merged = truth[truth["_session"].isin(eval_sessions_set)].merge(
        scored, on=["_session", "client_name"], how="left"
    )
    merged = merged.dropna(subset=["rank"])
    if merged.empty:
        logger.warning("ranking_eval_no_joinable_sales")
        return {}

    ranks = merged["rank"].to_numpy()
    realised = merged["payout"].to_numpy()

    # Payout captured if only the top brand were shown: sum of realised
    # payout on sales we put first, over total realised payout.
    captured_at_1 = float(realised[ranks == 1].sum())
    captured_at_3 = float(realised[ranks <= 3].sum())
    total_realised = float(realised.sum())

    n = len(ranks)
    hits_1, hits_3 = int((ranks == 1).sum()), int((ranks <= 3).sum())
    ci1_low, ci1_high = wald_ci(hits_1, n)
    ci3_low, ci3_high = wald_ci(hits_3, n)

    metrics = {
        "rankproxy_recall_at_1": round(float((ranks == 1).mean()), 4),
        "rankproxy_recall_at_1_ci_low": round(ci1_low, 4),
        "rankproxy_recall_at_1_ci_high": round(ci1_high, 4),
        "rankproxy_recall_at_3": round(float((ranks <= 3).mean()), 4),
        "rankproxy_recall_at_3_ci_low": round(ci3_low, 4),
        "rankproxy_recall_at_3_ci_high": round(ci3_high, 4),
        "rankproxy_mrr": round(float((1.0 / ranks).mean()), 4),
        "rankproxy_mean_rank_of_buyer": round(float(ranks.mean()), 3),
        "rankproxy_payout_captured_at_1_pct": round(100 * captured_at_1 / total_realised, 2)
        if total_realised
        else None,
        "rankproxy_payout_captured_at_3_pct": round(100 * captured_at_3 / total_realised, 2)
        if total_realised
        else None,
        "rankproxy_sessions_evaluated": int(merged["_session"].nunique()),
        "rankproxy_sales_evaluated": n,
        # Raw counts, so folds pool with integer arithmetic rather than by
        # averaging proportions over unequal weeks.
        "rankproxy_hits_at_1": hits_1,
        "rankproxy_hits_at_3": hits_3,
        "rankproxy_n_brands_ranked": len(brands),
        # How many brands bought anything at all in the held-out week. Without
        # it, a recall@3 of 1.0 reads as a triumph or a bug rather than the
        # structural fact it usually is: few brands buy, so the interesting
        # question is the order within them, not whether they make the top k.
        "rankproxy_distinct_buying_brands": int(merged["client_name"].nunique()),
        # Random-ranker baselines. Every recall@k above is unreadable without
        # the k/n it has to beat.
        "rankproxy_random_baseline_recall_at_1": round(1.0 / len(brands), 4),
        "rankproxy_random_baseline_recall_at_3": round(min(3.0 / len(brands), 1.0), 4),
        # 1 when the random baseline falls outside recall@1's interval, i.e.
        # the week had enough sales to say anything at all about position one.
        "rankproxy_recall_at_1_beats_chance": int(not (ci1_low <= 1.0 / len(brands) <= ci1_high)),
    }
    logger.info("ranking_eval_complete", extra=metrics)
    return metrics
