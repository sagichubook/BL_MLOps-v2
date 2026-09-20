"""FastAPI serving app.

Artifacts load once, in the lifespan hook. A startup canary then runs one real
prediction before the process reports ready: ``load_model()`` makes no network
call, so a process holding an evicted hosted fit looks healthy until the first
user arrives.
"""
from __future__ import annotations

import asyncio
import json
import logging
import secrets
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from starlette.middleware.base import BaseHTTPMiddleware

from bl_ranking.config import NO_CHAMPION_YET_ERROR_CODES, get_settings
from bl_ranking.logging_utils import configure_logging
from bl_ranking.payout_transport import (
    SURROGATE_FILENAME,
    SURROGATE_META_FILENAME,
    HistoricalBrandPayout,
    PayoutResultCache,
    ResilientPayoutRegressor,
    StubPayoutRegressor,
    SurrogatePayoutRegressor,
    build_live_predictor,
)
from bl_ranking.predict_service import ServingArtifacts, rank_brands
from bl_ranking.schemas import HealthResponse, RankBrandsResponse, ReadyResponse, UserSessionRequest

configure_logging()
logger = logging.getLogger("bl_ranking.api")

_state: dict = {"artifacts": None, "startup_error": None, "canary_ok": None}


class _Counter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._values: dict[str, int] = {}

    def increment(self, key: str = "total", _reason: str | None = None) -> None:
        with self._lock:
            self._values[key] = self._values.get(key, 0) + 1

    def as_dict(self) -> dict[str, int]:
        with self._lock:
            return dict(self._values)

    @property
    def count(self) -> int:
        return sum(self.as_dict().values())


_fallback_counter = _Counter()
_request_counter = _Counter()
_latency_samples: list[float] = []
_latency_lock = threading.Lock()


def _record_latency(ms: float) -> None:
    # Bounded reservoir, not a histogram library: enough for /metrics locally
    # without a dependency. Real deployments compute these in Prometheus.
    with _latency_lock:
        _latency_samples.append(ms)
        if len(_latency_samples) > 2000:
            del _latency_samples[: len(_latency_samples) - 2000]


def _latency_percentiles() -> dict[str, float]:
    with _latency_lock:
        sample = sorted(_latency_samples)
    if not sample:
        return {}
    def pct(p: float) -> float:
        return round(sample[min(int(len(sample) * p), len(sample) - 1)], 2)
    return {"p50_ms": pct(0.50), "p95_ms": pct(0.95), "p99_ms": pct(0.99), "n": len(sample)}


# Local work only: the payout transport bounds in-flight hosted calls itself.
_EXECUTOR = ThreadPoolExecutor(max_workers=8, thread_name_prefix="predict")



# Gated in middleware, not as a route dependency: FastAPI validates the body
# before a dependency can reject the caller, so an unauthenticated request
# would otherwise reach Pydantic — burning CPU and enumerating the schema
# through 422 field names. Probes stay open.
_AUTHED_PATHS = frozenset({"/predict"})


class ApiKeyMiddleware(BaseHTTPMiddleware):
    """Constant-time X-API-Key check. Unset means open, which is only correct
    behind a gateway that terminates auth; /ready reports which is in force."""

    def __init__(self, app, api_key: str | None) -> None:
        super().__init__(app)
        self._api_key = api_key

    async def dispatch(self, request: Request, call_next):
        if self._api_key and request.url.path in _AUTHED_PATHS:
            presented = request.headers.get("x-api-key", "")
            if not secrets.compare_digest(presented, self._api_key):
                _request_counter.increment("401")
                return JSONResponse(status_code=401, content={"detail": "missing or invalid X-API-Key"})
        return await call_next(request)


_ARTIFACT_FILES = (
    "CB_bl_lead.cbm",
    "payout_tfm_context.joblib",
    "all_clients.csv",
    "payout_tfm_model.json",
    "brand_historical_payout.json",
    "MODEL_VERSION.json",
    SURROGATE_FILENAME,
    SURROGATE_META_FILENAME,
)
_ESSENTIAL_ARTIFACTS = {"CB_bl_lead.cbm", "payout_tfm_context.joblib", "all_clients.csv"}


def resolve_artifacts_dir(settings) -> tuple:
    """Resolve which artifacts to serve.

    'registry' (default) resolves the 'champion' alias and downloads that
    run's files, which is what makes rollback a one-command operation.
    'local_dir' reads artifacts_path directly, for local dev.
    """
    artifacts_root = settings.artifacts_path_obj
    if settings.serving_artifact_source == "local_dir":
        return artifacts_root, None, None

    import mlflow
    from mlflow import MlflowClient
    from mlflow.exceptions import MlflowException

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    if settings.mlflow_registry_uri:
        mlflow.set_registry_uri(settings.mlflow_registry_uri)
    client = MlflowClient(
        tracking_uri=settings.mlflow_tracking_uri, registry_uri=settings.mlflow_registry_uri or None
    )
    name = settings.mlflow_registry_model_name_classifier
    try:
        mv = client.get_model_version_by_alias(name, "champion")
    except MlflowException as exc:
        if exc.error_code not in NO_CHAMPION_YET_ERROR_CODES:
            # Anything other than "no champion yet" (e.g. an unreachable
            # tracking server) is a real operational problem, not a benign
            # first-run state — don't mask it as one.
            raise
        logger.warning(
            "no MLflow 'champion' alias for registered model '%s' — "
            "falling back to artifacts_path; run training in production mode first",
            name,
        )
        return artifacts_root, None, None

    dest = artifacts_root / "resolved" / str(mv.version)
    dest.mkdir(parents=True, exist_ok=True)
    for fname in _ARTIFACT_FILES:
        if (dest / fname).exists():
            continue
        try:
            mlflow.artifacts.download_artifacts(run_id=mv.run_id, artifact_path=fname, dst_path=str(dest))
        except Exception:
            if fname in _ESSENTIAL_ARTIFACTS:
                logger.exception("required artifact %s failed to download from run %s", fname, mv.run_id)
                raise
            logger.info("optional artifact %s not present on run %s", fname, mv.run_id)
    return dest, str(mv.version), mv.run_id


def load_artifacts() -> ServingArtifacts:
    settings = get_settings()
    artifacts_dir, registry_version, run_id = resolve_artifacts_dir(settings)
    if run_id:
        logger.info("serving artifacts resolved from MLflow run %s (registry version %s)", run_id, registry_version)
    predictors_path = str(artifacts_dir) + ("/" if not str(artifacts_dir).endswith("/") else "")

    import catboost
    import joblib
    import pandas as pd
    from catboost import CatBoostClassifier

    # thread_count=1: predict_proba otherwise uses every core per call and
    # oversubscribes under concurrency, which the thread pool already supplies.
    cb_model = CatBoostClassifier(allow_writing_files=False, thread_count=1)
    cb_model.load_model(str(artifacts_dir / "CB_bl_lead.cbm"), format="cbm")

    all_clients = pd.read_csv(artifacts_dir / "all_clients.csv")

    context_path = artifacts_dir / "payout_tfm_context.joblib"
    context = joblib.load(context_path)
    tfm_columns = list(context["columns"])

    fallback_path = artifacts_dir / "brand_historical_payout.json"
    historical = (
        HistoricalBrandPayout.load(fallback_path)
        if fallback_path.exists()
        else HistoricalBrandPayout.compute(context["x"], context["y"].to_frame(), source="context_only_degraded")
    )
    brands = [b for b in all_clients["client_name"].dropna().astype(str).tolist() if b != "other"]
    coverage = historical.coverage_of(brands)
    if coverage["coverage_pct"] < 100:
        logger.warning("fallback_table_incomplete", extra=coverage)

    model_version = registry_version or "unknown"
    version_file = artifacts_dir / "MODEL_VERSION.json"
    if version_file.exists():
        model_version = json.loads(version_file.read_text()).get("model_version", model_version)

    # --- the local surrogate, if this model version shipped one -----------
    surrogate = None
    surrogate_meta = None
    surrogate_path = artifacts_dir / SURROGATE_FILENAME
    if surrogate_path.exists():
        try:
            surrogate = SurrogatePayoutRegressor.load(
                surrogate_path, tfm_columns, artifacts_dir / SURROGATE_META_FILENAME
            )
            surrogate_meta = surrogate.meta
            logger.info("surrogate_loaded", extra={k: v for k, v in (surrogate_meta or {}).items() if not isinstance(v, list)})
        except Exception:
            logger.exception("failed to load payout surrogate; continuing without it")

    mode = settings.tabpfn_transport
    cache = PayoutResultCache(settings.payout_cache_size, settings.payout_cache_ttl_s)

    # Ladder, best first: the surrogate beats a per-brand mean, so it sits
    # above the historical table whatever the primary is.
    fallbacks: list[tuple[str, object]] = []
    if surrogate is not None and mode != "surrogate":
        fallbacks.append(("surrogate", surrogate))
    fallbacks.append(("historical", historical))

    if mode == "stub":
        primary, primary_name = StubPayoutRegressor(), "stub"
    elif mode == "surrogate":
        if surrogate is None:
            raise RuntimeError(
                "tabpfn_transport=surrogate but no payout_surrogate.cbm in the served artifacts; "
                "run production training with distillation enabled, or choose another transport"
            )
        primary, primary_name = surrogate, "surrogate"
    else:
        if not settings.tabpfn_token:
            raise RuntimeError(f"tabpfn_transport={mode} requires TABPFN_TOKEN")
        primary = build_live_predictor(
            settings.tabpfn_token,
            artifacts_dir / "payout_tfm_model.json",
            context_path,
            use_cache_mode=(mode != "live_no_cache"),
        )
        primary_name = mode

    payout_predictor = ResilientPayoutRegressor(
        primary=primary,
        fallbacks=fallbacks,
        timeout_s=settings.tabpfn_predict_timeout_s,
        model_version=model_version,
        max_concurrency=settings.tabpfn_max_concurrency,
        admission_wait_s=settings.tabpfn_admission_wait_s,
        cache=cache,
        on_event=_fallback_counter.increment,
        primary_name=primary_name,
    )

    return ServingArtifacts(
        cb_model=cb_model,
        payout_predictor=payout_predictor,
        tfm_columns=tfm_columns,
        predictors_path=predictors_path,
        all_clients=all_clients,
        catboost_version=catboost.__version__,
        tabpfn_transport_mode=mode,
        model_version=model_version,
        loaded_at=datetime.now(UTC),
        surrogate_meta=surrogate_meta,
    )


def _canary(artifacts: ServingArtifacts) -> bool:
    """One real prediction before readiness. Catches what a load-time check
    cannot -- a saved hosted fit whose server-side state is gone -- and warms
    that state so the first user does not pay the cold path."""
    from bl_ranking.predict_service import rank_brands as _rank

    probe = {
        "session_dt": "2026-01-06 19:24:22", "conversion_dt": None, "register_date": "2026-01-06 19:26:07",
        "campaign_id": 1, "page": "canary", "auto_city": "Fort Lauderdale", "auto_country": "United States",
        "auto_state": "Florida", "device_type": "mobile", "sub1": "0", "sub2": "0", "sub3": "0",
        "business_type": "C Corporation", "credit_score": "Very Poor - Under 550", "industry": "construction",
        "loan_amount": "$25,000 - $49,999", "loan_reason": "Equipment purchase",
        "monthly_revenue": "$20,000 - $49,999", "time_in_business": "2+ years",
        "fname": "Canary", "lname": "Probe", "cellphone": 5550000000,
    }
    started = time.monotonic()
    result = _rank(probe, artifacts, logging.getLogger("bl_ranking.canary"))
    elapsed_ms = (time.monotonic() - started) * 1000
    logger.info(
        "startup_canary",
        extra={
            "tier": result.payout_tier,
            "n_brands": len(result.rankings),
            "latency_ms": round(elapsed_ms, 1),
            "degraded": result.fallback_used,
        },
    )
    return not result.fallback_used


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Reset explicitly: a failed reload must not leave stale artifacts ready.
    _state.update({"artifacts": None, "startup_error": None, "canary_ok": None})
    settings = get_settings()
    try:
        loop = asyncio.get_event_loop()
        artifacts = await loop.run_in_executor(None, load_artifacts)
        _state["artifacts"] = artifacts
        logger.info(
            "artifacts_loaded",
            extra={"model_version": artifacts.model_version, "mode": artifacts.tabpfn_transport_mode},
        )
        if settings.startup_canary:
            try:
                _state["canary_ok"] = await loop.run_in_executor(None, _canary, artifacts)
            except Exception:
                logger.exception("startup_canary_failed")
                _state["canary_ok"] = False
    except Exception as exc:  # noqa: BLE001
        # /ready is unauthenticated: expose the exception type only, not
        # internal paths, credentials or SDK error text.
        _state["startup_error"] = type(exc).__name__
        logger.exception("artifact_load_failed")
    yield


app = FastAPI(title="BL Brand Ranking API", lifespan=lifespan)
# Unauthenticated probe endpoints stay open; /predict requires the key.
app.add_middleware(ApiKeyMiddleware, api_key=get_settings().api_key)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    # Liveness only; /ready says whether it can serve a prediction.
    return HealthResponse(status="ok")


@app.get("/ready", response_model=ReadyResponse)
async def ready() -> JSONResponse:
    artifacts: ServingArtifacts | None = _state["artifacts"]
    if artifacts is None:
        body = ReadyResponse(ready=False, detail=_state["startup_error"] or "artifacts not loaded yet")
        return JSONResponse(status_code=503, content=body.model_dump())

    settings = get_settings()
    stats = artifacts.payout_predictor.stats() if hasattr(artifacts.payout_predictor, "stats") else {}
    meta = artifacts.surrogate_meta or {}
    return JSONResponse(
        status_code=200,
        content=ReadyResponse(
            ready=True,
            model_version=artifacts.model_version,
            artifacts_loaded_at=artifacts.loaded_at.isoformat(),
            tabpfn_transport_mode=artifacts.tabpfn_transport_mode,
            fallback_count=_fallback_counter.count,
            canary_ok=_state.get("canary_ok"),
            auth_required=bool(settings.api_key),
            payout_stats=stats,
            surrogate_top1_agreement=meta.get("rank_top1_agreement"),
        ).model_dump(),
    )


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics() -> str:
    """Prometheus text exposition. Hand-rolled: one endpoint's counters do
    not justify a client library, and the names are scraper-stable.

    Counters are per worker process. With web_concurrency > 1 a scrape lands
    on whichever worker answers, so values look erratic unless the scraper
    targets each worker or the deployment uses prometheus_client's
    multiprocess mode.
    """
    artifacts: ServingArtifacts | None = _state["artifacts"]
    lines = [
        "# HELP bl_ranking_ready 1 when the service can serve predictions",
        "# TYPE bl_ranking_ready gauge",
        f"bl_ranking_ready {1 if artifacts is not None else 0}",
    ]
    for status, count in _request_counter.as_dict().items():
        lines.append(f'bl_ranking_requests_total{{status="{status}"}} {count}')
    for reason, count in _fallback_counter.as_dict().items():
        lines.append(f'bl_ranking_payout_fallback_total{{reason="{reason}"}} {count}')
    if artifacts is not None and hasattr(artifacts.payout_predictor, "stats"):
        stats = artifacts.payout_predictor.stats()
        for tier, count in stats.get("payout_calls", {}).items():
            lines.append(f'bl_ranking_payout_calls_total{{tier="{tier}"}} {count}')
        cache = stats.get("payout_cache") or {}
        for key in ("hits", "misses", "entries"):
            if key in cache:
                lines.append(f"bl_ranking_payout_cache_{key} {cache[key]}")
    for key, value in _latency_percentiles().items():
        lines.append(f"bl_ranking_latency_{key} {value}")
    return "\n".join(lines) + "\n"


@app.post("/predict", response_model=RankBrandsResponse)
async def predict(payload: UserSessionRequest, request: Request) -> RankBrandsResponse:
    settings = get_settings()
    artifacts: ServingArtifacts | None = _state["artifacts"]
    if artifacts is None:
        _request_counter.increment("503")
        raise HTTPException(status_code=503, detail="model artifacts not loaded")

    request_id = request.headers.get("x-request-id", "-")
    request_logger = logging.getLogger("bl_ranking.request")
    start = time.monotonic()
    try:
        result = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(
                _EXECUTOR, rank_brands, payload.to_predictor_dict(), artifacts, request_logger
            ),
            timeout=settings.request_timeout_s,
        )
    except TimeoutError as exc:
        _request_counter.increment("504")
        raise HTTPException(status_code=504, detail="prediction exceeded request budget") from exc
    except ValueError as exc:
        _request_counter.increment("422")
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    elapsed_ms = (time.monotonic() - start) * 1000
    _record_latency(elapsed_ms)
    _request_counter.increment("200")
    logger.info(
        "predict_served",
        extra={
            "request_id": request_id,
            "model_version": result.model_version,
            "payout_tier": result.payout_tier,
            "fallback_used": result.fallback_used,
            "n_brands_ranked": len(result.rankings),
            "latency_ms": round(elapsed_ms, 2),
        },
    )
    return RankBrandsResponse(
        rankings=result.rankings,
        model_version=result.model_version,
        fallback_used=result.fallback_used,
        payout_tier=result.payout_tier,
        latency_ms=round(elapsed_ms, 2),
    )
