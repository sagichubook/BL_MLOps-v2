"""MLflow-tracked wrapper around the vendored ``BLPayoutModelsFit``.

``fit_()`` runs unchanged -- the parity test proves it by running the class
directly and through this wrapper on identical data. Around it this adds what
a weekly job needs and a research script does not: an input quality gate,
drift notes, params/metrics/the researcher log as artifacts, the
full-population fallback table, a cache-mode TabPFN fit, the distilled
surrogate and its fidelity report, and registration with champion/previous.

``train_test`` additionally runs a ranking evaluation (see ``ranking_eval``):
classifier F1 and payout MAE can both look healthy while the brand order --
the product -- is wrong.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import catboost
import joblib
import mlflow
import pandas as pd
import sklearn
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from bl_ranking.config import NO_CHAMPION_YET_ERROR_CODES, get_settings
from bl_ranking.data_quality import check_training_input, compare_to_previous
from bl_ranking.logging_utils import JsonFormatter, configure_logging
from bl_ranking.name_index import INDEX_FILENAME
from bl_ranking.original.bl_models_train import BLPayoutModelsFit
from bl_ranking.payout_transport import HistoricalBrandPayout, TabPFNCachedRegressor

logger = logging.getLogger("bl_ranking.train_pipeline")


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------


def _git_sha(repo_dir: Path) -> str:
    """Commit SHA for run provenance.

    GIT_SHA first: .git is not in the Docker build context, so inside an image
    neither `git` nor the repo-reading fallback below can work, and every
    containerised run would log "unknown" — worthless exactly where
    reproducibility matters. The Dockerfile bakes it in as a build arg.
    """
    baked = os.environ.get("GIT_SHA", "").strip()
    if baked and baked != "unknown":
        return baked
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        pass
    return "unknown"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------
# Logging plumbing around the vendored logger
# --------------------------------------------------------------------------


def _find_log_file() -> str | None:
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.FileHandler):
            return handler.baseFilename
    return None


def _restore_stdout_json_logging() -> None:
    # setup_bl_logger() (vendored) clears root handlers and attaches only its
    # own FileHandler. Re-attach stdout without touching that file handler.
    root = logging.getLogger()
    if not any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in root.handlers
    ):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        root.addHandler(handler)


_METRIC_PATTERNS = {
    "payout_mape_overall": re.compile(r"Overall.*?Payout mean_absolute_percentage_error:([\d.]+)", re.S),
    "payout_mae_overall": re.compile(r"Payout mean_absolute_error:([\d.]+)"),
    "classifier_accuracy_all_days": re.compile(r"for all test.*?accuracy\s+([\d.]+)", re.S),
    "classifier_f1_positive_all_days": re.compile(r"for all test.*?\n\s*1\s+[\d.]+\s+[\d.]+\s+([\d.]+)", re.S),
}


def _parse_metrics_from_log(text: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name, pattern in _METRIC_PATTERNS.items():
        match = pattern.search(text)
        if match:
            try:
                metrics[name] = float(match.group(1))
            except ValueError:
                continue
    return metrics


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------


def _register_and_alias(cb_model_path: Path, settings) -> str:
    from catboost import CatBoostClassifier

    cb = CatBoostClassifier(allow_writing_files=False)
    cb.load_model(str(cb_model_path), format="cbm")
    model_info = mlflow.catboost.log_model(
        cb, name="model", registered_model_name=settings.mlflow_registry_model_name_classifier
    )
    version = str(model_info.registered_model_version)

    client = MlflowClient(registry_uri=settings.mlflow_registry_uri or None)
    name = settings.mlflow_registry_model_name_classifier
    try:
        current_champion = client.get_model_version_by_alias(name, "champion")
        if str(current_champion.version) != version:
            client.set_registered_model_alias(name, "previous", current_champion.version)
    except MlflowException as exc:
        if exc.error_code not in NO_CHAMPION_YET_ERROR_CODES:
            # Masking this would skip 'previous' and break rollback silently.
            raise
        logger.info("no existing 'champion' alias for %s — first registration", name)
    client.set_registered_model_alias(name, "champion", version)
    return version


def _previous_production_metrics(settings) -> dict:
    """Data-quality metrics from the last successful production run."""
    try:
        client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
        experiment = client.get_experiment_by_name(settings.mlflow_experiment_name)
        if experiment is None:
            return {}
        runs = client.search_runs(
            [experiment.experiment_id],
            filter_string="params.mode = 'production' and attributes.status = 'FINISHED'",
            order_by=["attributes.start_time DESC"],
            max_results=1,
        )
        return dict(runs[0].data.metrics) if runs else {}
    except Exception:
        logger.info("could not read previous production run metrics (first run?)", exc_info=False)
        return {}


# --------------------------------------------------------------------------
# Production extras
# --------------------------------------------------------------------------


def _vendored_training_frames(model: BLPayoutModelsFit):
    """Recover the training frames via the vendored preprocessing.

    ``fit_()`` builds these internally and returns nothing. A second pass
    costs ~15 s against a ~2 min CatBoost fit and keeps fit_() untouched.
    """
    x_train, y_train, _, _ = model.bl_preprocessing()
    x_train_payout, y_train_payout = model.prepare_for_cont_payout_prediction_train(x_train, y_train)
    return x_train, y_train, x_train_payout, y_train_payout


def _build_surrogate(
    settings,
    output_dir: Path,
    x_train: pd.DataFrame,
    context: dict,
    brands: list[str],
    cb_model_path: Path,
    n_users: int,
) -> dict:
    """Distil the hosted teacher into a local model and report the gap."""
    from catboost import CatBoostClassifier

    from bl_ranking.distill import distil_payout_model, save_surrogate

    teacher = TabPFNCachedRegressor(settings.tabpfn_token, context=context)
    teacher.fit(context["x"], context["y"])

    cb = CatBoostClassifier(allow_writing_files=False, thread_count=1)
    cb.load_model(str(cb_model_path), format="cbm")

    surrogate, report = distil_payout_model(
        feature_rows=x_train,
        brands=brands,
        columns=list(context["columns"]),
        teacher=teacher,
        cb_model=cb,
        n_users=n_users,
    )
    model_path, meta_path = save_surrogate(surrogate, report, list(context["columns"]), output_dir)
    mlflow.log_artifact(str(model_path))
    mlflow.log_artifact(str(meta_path))
    return report.to_dict()


def _ranking_evaluation(model: BLPayoutModelsFit, settings) -> dict:
    """Score one held-out week's brand order.

    Rebuilds both models via the vendored methods rather than reaching inside
    ``fit_()``; costs a second CatBoost fit, keeps fit_() untouched.
    """
    from bl_ranking.ranking_eval import evaluate_ranking

    x_train, y_train, x_test, y_test = model.bl_preprocessing()
    if x_test.empty:
        return {}
    cb = model.catbosot_model_sold_to_client(x_train, y_train)
    x_train_payout, y_train_payout = model.prepare_for_cont_payout_prediction_train(x_train, y_train)
    teacher, context = model.tabpfn_regression_payout(x_train_payout, y_train_payout)
    brands = sorted(x_train["client_name"].dropna().unique().tolist())
    return evaluate_ranking(
        x_test=x_test,
        y_test=y_test,
        brands=brands,
        columns=list(context["columns"]),
        cb_model=cb,
        payout_predictor=teacher,
    )


def _truncated_copy(input_path: str, input_file: str, weeks_back: int, dest_dir: Path) -> tuple[str, str]:
    """A copy of the input ending ``weeks_back`` weeks earlier.

    Walk-forward has to move the *end* of the data, because the vendored
    ``split_by_time()`` always holds out the last 7 days relative to whatever
    it is given. Slicing the input is what lets the folds run without touching
    that function — which the parity test requires stay untouched.
    """
    frame = pd.read_csv(input_path + input_file, low_memory=False)
    session_dt = pd.to_datetime(frame["session_dt"], errors="coerce")
    cutoff = session_dt.max() - pd.Timedelta(days=7 * weeks_back)
    frame = frame[session_dt <= cutoff]
    dest_dir.mkdir(parents=True, exist_ok=True)
    name = f"fold_{weeks_back}_{input_file}"
    frame.to_csv(dest_dir / name, index=False)
    logger.info("fold_input_written", extra={"weeks_back": weeks_back, "rows": len(frame), "cutoff": str(cutoff)})
    return str(dest_dir) + "/", name


def _pool_folds(per_fold: list[dict], n_brands: int) -> dict:
    """Pool recall over folds with integer arithmetic, and re-derive the CI.

    A single week does not carry enough sales to separate recall@1 from the
    random baseline; pooling is the point of running folds at all, so the
    pooled interval is the number to read.
    """
    from bl_ranking.ranking_eval import wald_ci

    sales = sum(int(m.get("rankproxy_sales_evaluated", 0)) for m in per_fold)
    if not sales:
        return {}
    hits_1 = sum(int(m.get("rankproxy_hits_at_1", 0)) for m in per_fold)
    hits_3 = sum(int(m.get("rankproxy_hits_at_3", 0)) for m in per_fold)
    lo1, hi1 = wald_ci(hits_1, sales)
    lo3, hi3 = wald_ci(hits_3, sales)
    baseline = 1.0 / n_brands
    return {
        "rankproxy_pooled_folds": len(per_fold),
        "rankproxy_pooled_sales": sales,
        "rankproxy_pooled_recall_at_1": round(hits_1 / sales, 4),
        "rankproxy_pooled_recall_at_1_ci_low": round(lo1, 4),
        "rankproxy_pooled_recall_at_1_ci_high": round(hi1, 4),
        "rankproxy_pooled_recall_at_3": round(hits_3 / sales, 4),
        "rankproxy_pooled_recall_at_3_ci_low": round(lo3, 4),
        "rankproxy_pooled_recall_at_3_ci_high": round(hi3, 4),
        "rankproxy_pooled_recall_at_1_beats_chance": int(not (lo1 <= baseline <= hi1)),
    }


def _ranking_evaluation_folds(
    input_path: str, input_file: str, output_predictors_path: str, settings, folds: int
) -> dict:
    """Walk-forward: repeat the evaluation over successive weekly cutoffs."""
    import shutil
    import tempfile

    per_fold: list[dict] = []
    brand_counts: list[int] = []
    tmp = Path(tempfile.mkdtemp(prefix="bl_folds_"))
    try:
        for k in range(folds):
            if k == 0:
                fold_path, fold_file = input_path, input_file
            else:
                fold_path, fold_file = _truncated_copy(input_path, input_file, k, tmp)
            model = BLPayoutModelsFit(fold_path, fold_file, output_predictors_path, train_test=True)
            _restore_stdout_json_logging()
            metrics = _ranking_evaluation(model, settings)
            if not metrics:
                continue
            if metrics.get("rankproxy_n_brands_ranked"):
                brand_counts.append(int(metrics["rankproxy_n_brands_ranked"]))
            per_fold.append(metrics)
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"fold{k}_{key}", value)
            logger.info("fold_complete", extra={"fold": k, **metrics})
            if k > 0:
                (tmp / fold_file).unlink(missing_ok=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    if not per_fold:
        return {}
    # Smallest brand count across folds: the largest random baseline, so
    # "beats chance" stays the conservative reading.
    pooled = _pool_folds(per_fold, min(brand_counts) if brand_counts else 1)
    return {**per_fold[0], **pooled}


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def run_training(
    mode: str,
    input_path: str,
    input_file: str,
    output_predictors_path: str,
    build_surrogate: bool | None = None,
    rank_eval: bool = False,
    rank_eval_folds: int = 1,
    surrogate_users: int = 10_000,
    skip_quality_gate: bool = False,
) -> str | None:
    """mode: 'train_test' or 'production'. Returns the registered model
    version string in production mode, else None."""
    configure_logging()
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    if settings.mlflow_registry_uri:
        mlflow.set_registry_uri(settings.mlflow_registry_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    train_test = mode == "train_test"
    if settings.tabpfn_token:
        os.environ["TABPFN_TOKEN"] = settings.tabpfn_token
    if build_surrogate is None:
        build_surrogate = bool(settings.tabpfn_token) and not train_test

    os.makedirs(output_predictors_path, exist_ok=True)
    csv_path = input_path + input_file

    # --- gate the input before spending two minutes on a CatBoost fit -----
    quality = check_training_input(csv_path) if not skip_quality_gate else None
    if quality is not None and not quality.ok:
        raise ValueError("training input failed quality checks: " + "; ".join(quality.errors))

    try:
        import tabpfn_client

        tabpfn_client_version = getattr(tabpfn_client, "__version__", "unknown")
    except Exception:
        tabpfn_client_version = "unknown"

    from bl_ranking.name_index import active_index

    index = active_index()
    run_name = f"{mode}-{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    registered_version: str | None = None

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(
            {
                "mode": mode,
                "data_file": input_file,
                "n_rows_raw": (quality.metrics.get("dq_n_rows") if quality else None),
                "data_date_span_days": (quality.metrics.get("dq_span_days") if quality else None),
                "catboost_depth": 8,
                "catboost_n_estimators": 800,
                "catboost_eval_metric": "F1",
                "tabpfn_context_size": 1000,
                "surrogate_enabled": bool(build_surrogate),
                "git_sha": _git_sha(Path(__file__).resolve().parents[2]),
                "python_version": platform.python_version(),
                "pandas_version": pd.__version__,
                "catboost_version": catboost.__version__,
                "scikit_learn_version": sklearn.__version__,
                "tabpfn_client_version": tabpfn_client_version,
                "mlflow_version": mlflow.__version__,
                "names_dataset_version": index.source_version if index else "real-library",
                "name_index_active": index is not None,
            }
        )
        if quality is not None:
            mlflow.log_metrics({k: v for k, v in quality.metrics.items() if isinstance(v, (int, float))})
            if quality.warnings:
                mlflow.set_tag("data_quality_warnings", "; ".join(quality.warnings)[:4500])

            drift = compare_to_previous(quality.metrics, _previous_production_metrics(settings))
            mlflow.log_metric("dq_drift_flags", len(drift))
            if drift:
                mlflow.set_tag("data_drift_notes", "; ".join(drift)[:4500])

        # --- the vendored run, untouched ---------------------------------
        t0 = time.monotonic()
        model = BLPayoutModelsFit(input_path, input_file, output_predictors_path, train_test=train_test)
        _restore_stdout_json_logging()
        log_file_path = _find_log_file()
        model.fit_()  # vendored, unchanged
        mlflow.log_metric("runtime_seconds", time.monotonic() - t0)

        if log_file_path and Path(log_file_path).exists():
            mlflow.log_artifact(log_file_path, artifact_path="researcher_log")
            metrics = _parse_metrics_from_log(Path(log_file_path).read_text())
            if metrics:
                mlflow.log_metrics(metrics)
            else:
                logger.info("no known metric patterns matched in researcher log; raw log is still archived")

        mlflow.set_tag("deviation_from_original", "none in fit_() — see docs/part2.md for serving-side deviations")

        # --- train_test: evaluate the ranking, not just the two legs -----
        if train_test:
            if rank_eval and settings.tabpfn_token:
                try:
                    ranking_metrics = (
                        _ranking_evaluation_folds(
                            input_path, input_file, output_predictors_path, settings, rank_eval_folds
                        )
                        if rank_eval_folds > 1
                        else _ranking_evaluation(model, settings)
                    )
                    if ranking_metrics:
                        mlflow.log_metrics({k: v for k, v in ranking_metrics.items() if isinstance(v, (int, float))})
                        logger.info("ranking evaluation logged", extra=ranking_metrics)
                except Exception:
                    logger.exception("ranking evaluation failed (non-fatal); per-leg metrics still logged")
            mlflow.set_tag("artifacts_written", "false")
            logger.info("train_test run complete — no artifacts registered (matches the original script's contract)")
            return None

        # --- production: artifacts, surrogate, registration ---------------
        artifact_files = {
            "catboost_model": Path(output_predictors_path) / "CB_bl_lead.cbm",
            "tabpfn_context": Path(output_predictors_path) / "payout_tfm_context.joblib",
            "all_clients": Path(output_predictors_path) / "all_clients.csv",
        }
        hashes = {}
        for name, path in artifact_files.items():
            mlflow.log_artifact(str(path))
            hashes[f"sha256_{name}"] = _sha256(path)
        mlflow.log_params(hashes)

        # Ships with the model so a rollback restores feature semantics too.
        index_src = settings.name_index_path_obj
        if not index_src.exists():
            index_src = Path(__file__).resolve().parents[2] / settings.name_index_path
        if index_src.exists():
            mlflow.log_artifact(str(index_src))
            mlflow.log_param("sha256_name_index", _sha256(index_src)[:16])
        else:
            logger.warning("name index not found at %s; not logged with the run", settings.name_index_path)

        context = joblib.load(artifact_files["tabpfn_context"])
        all_clients = pd.read_csv(artifact_files["all_clients"])
        brands = [b for b in all_clients["client_name"].dropna().astype(str).tolist() if b != "other"]

        # Cache-mode fit, so serving never fits per request.
        cached_path = Path(output_predictors_path) / "payout_tfm_model.json"
        if settings.tabpfn_token:
            cached = TabPFNCachedRegressor(settings.tabpfn_token, context=context)
            cached.fit(context["x"], context["y"])
            cached.save(cached_path)
            mlflow.log_artifact(str(cached_path))
            mlflow.set_tag("tabpfn_model_id", cached.model_id or "unknown")
        else:
            logger.warning(
                "no TABPFN_TOKEN set — skipping the cache-mode TabPFN fit; "
                "serving will fit once from the raw context at process startup instead"
            )

        # Fallback table over the FULL payout population, not the context.
        x_train, _, x_train_payout, y_train_payout = _vendored_training_frames(model)
        fallback = HistoricalBrandPayout.compute(x_train_payout, y_train_payout)
        fallback_path = Path(output_predictors_path) / "brand_historical_payout.json"
        fallback.save(fallback_path)
        mlflow.log_artifact(str(fallback_path))
        coverage = fallback.coverage_of(brands)
        mlflow.log_metrics(
            {
                "fallback_brand_coverage_pct": coverage["coverage_pct"],
                "fallback_brands_covered": coverage["n_covered"],
                "fallback_rows": int(len(x_train_payout)),
            }
        )
        if coverage["uncovered_brands"]:
            mlflow.set_tag("fallback_uncovered_brands", ", ".join(coverage["uncovered_brands"])[:4500])

        # Distilled local surrogate + the fidelity report that justifies it.
        if build_surrogate and settings.tabpfn_token:
            try:
                report = _build_surrogate(
                    settings,
                    Path(output_predictors_path),
                    x_train,
                    context,
                    brands,
                    artifact_files["catboost_model"],
                    n_users=surrogate_users,
                )
                mlflow.log_metrics({k: v for k, v in report.items() if isinstance(v, (int, float))})
                logger.info("surrogate distilled", extra=report)
            except Exception:
                logger.exception("surrogate distillation failed (non-fatal); hosted transport is unaffected")
                mlflow.set_tag("surrogate_build", "failed")

        registered_version = _register_and_alias(artifact_files["catboost_model"], settings)
        version_file = Path(output_predictors_path) / "MODEL_VERSION.json"
        version_file.write_text(
            json.dumps(
                {
                    "model_version": registered_version,
                    "run_id": run.info.run_id,
                    "registered_at": datetime.now(UTC).isoformat(),
                    "name_index": INDEX_FILENAME,
                }
            )
        )
        mlflow.log_artifact(str(version_file))
        mlflow.set_tag("model_version", registered_version)
        logger.info("production run complete — registered model version %s", registered_version)

    return registered_version


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the BL brand-ranking models with MLflow tracking.")
    parser.add_argument("--mode", choices=["train_test", "production"], required=True)
    parser.add_argument("--input-path", default=None)
    parser.add_argument("--input-file", default=None)
    parser.add_argument("--output-path", default=None)
    parser.add_argument(
        "--no-surrogate", action="store_true", help="skip distillation (production mode only)"
    )
    parser.add_argument(
        "--surrogate-users",
        type=int,
        default=10_000,
        help="distinct users in the distillation grid. Measured against 2,500: top-1 agreement "
        "0.9792 -> 0.9832 and expected-payout loss 0.052%% -> 0.033%%, for ~14 min more on a "
        "weekly job. The CatBoost fit grows superlinearly, so this is near the practical knee.",
    )
    parser.add_argument(
        "--rank-eval",
        action="store_true",
        help="train_test only: also evaluate the brand ordering on the held-out week "
        "(costs a second CatBoost fit and ~2 min of hosted calls)",
    )
    parser.add_argument(
        "--rank-eval-folds",
        type=int,
        default=1,
        help="walk forward over N successive weekly cutoffs and pool the result. One week "
        "rarely has enough sales to separate recall@1 from the random baseline; ~3 folds does. "
        "Costs roughly 4 min per fold.",
    )
    parser.add_argument("--skip-quality-gate", action="store_true", help="bypass input validation (not for production)")
    args = parser.parse_args()

    settings = get_settings()
    run_training(
        mode=args.mode,
        input_path=args.input_path or settings.data_path,
        input_file=args.input_file or settings.data_file,
        output_predictors_path=args.output_path or settings.artifacts_path,
        build_surrogate=False if args.no_surrogate else None,
        rank_eval=args.rank_eval,
        rank_eval_folds=args.rank_eval_folds,
        surrogate_users=args.surrogate_users,
        skip_quality_gate=args.skip_quality_gate,
    )


if __name__ == "__main__":
    main()
