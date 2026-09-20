"""Runtime configuration, sourced entirely from the environment (with an
optional local .env for development). No hard-coded paths or secrets.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- data ---
    data_path: str = Field(default="data/raw/", description="Directory containing the input CSV")
    data_file: str = Field(default="bl_full_data.csv")

    # --- artifacts ---
    artifacts_path: str = Field(default="artifacts/", description="Where training writes model artifacts")
    name_index_path: str = Field(
        default="assets/name_gender_index.json.gz",
        description="Compact first-name gender index that replaces the 1.9 GB names_dataset at runtime. "
        "Built by scripts/build_name_index.py; tracked in git because it is a small, deterministic "
        "distillation of a pinned dependency.",
    )

    # --- MLflow ---
    # MLflow's plain filesystem store ("file:./mlruns") is in maintenance
    # mode as of mlflow 3.x and never supported the Model Registry (which
    # registration and rollback depend on) — a database-backed store is
    # required. sqlite is the zero-infrastructure default for local dev;
    # docker compose runs a real `mlflow server` against it, and Databricks
    # sets MLFLOW_TRACKING_URI=databricks (see databricks/).
    mlflow_tracking_uri: str = Field(default="sqlite:///mlruns.db")
    mlflow_registry_uri: str | None = Field(
        default=None,
        description="Registry URI when it differs from the tracking URI. Set to 'databricks-uc' to register "
        "into Unity Catalog, in which case the model name must be catalog.schema.name.",
    )
    mlflow_experiment_name: str = Field(default="bl_brand_ranking")
    mlflow_registry_model_name_classifier: str = Field(default="bl_lead_classifier")
    # No separate registry entry for the payout model: the TabPFN cache
    # reference, the surrogate and the fallback table are logged as artifacts
    # of the same run as the classifier, so one alias move rolls back the
    # whole serving bundle atomically. See docs/part2.md.
    serving_artifact_source: str = Field(
        default="registry",
        description="'registry' resolves the 'champion' alias from the MLflow Model Registry at startup "
        "(the real rollback path — see scripts/rollback.py); 'local_dir' reads artifacts_path directly, "
        "for quick local dev without a reachable MLflow server.",
    )

    # --- TabPFN / payout transport ---
    tabpfn_token: str | None = Field(default=None, description="Read from TABPFN_TOKEN env var")
    tabpfn_transport: str = Field(
        default="surrogate",
        description="Payout leg. 'surrogate' = local distilled model (2 ms); 'live' = hosted TabPFN, "
        "exact original answers (~3 s); 'live_no_cache' = hosted without the cached fit, kept as the "
        "benchmark baseline; 'stub' = deterministic offline, for tests.",
    )
    tabpfn_predict_timeout_s: float = Field(
        default=6.0, description="Deadline for one hosted TabPFN call, measured from when it actually starts"
    )
    tabpfn_max_concurrency: int = Field(
        default=8,
        description="Concurrent in-flight hosted TabPFN calls per process. Beyond this, requests go straight "
        "to the fallback tier instead of queueing — bounded latency under overload rather than a queue that "
        "silently eats the whole request budget. See payout_transport.ResilientPayoutRegressor.",
    )
    tabpfn_admission_wait_s: float = Field(
        default=0.25, description="How long a request waits for a hosted-call slot before shedding to fallback"
    )
    payout_cache_size: int = Field(default=4096, description="Entries in the per-process payout result cache (0 disables)")
    payout_cache_ttl_s: float = Field(
        default=900.0,
        description="Payout cache entry lifetime. Bounded because the hosted fit is refreshed weekly and a "
        "stale payout silently mis-ranks brands; short enough to be safe, long enough to absorb bursts.",
    )
    startup_canary: bool = Field(
        default=True,
        description="Run one real payout prediction at startup. Validates the hosted fit still exists "
        "(a saved model_id can be evicted server-side), warms the server-side KV cache, and gates /ready "
        "on the payout leg actually working rather than merely loading.",
    )

    # --- serving ---
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    web_concurrency: int = Field(
        default=2,
        description="uvicorn worker PROCESSES. Measured: a bigger thread pool inside one process barely moves "
        "throughput (CPU/GIL-bound pandas preprocessing doesn't parallelise across threads); matching worker "
        "processes to CPU cores does. Set to the host's core count.",
    )
    request_timeout_s: float = Field(default=10.0, description="Overall /predict request budget")
    api_key: str | None = Field(
        default=None,
        description="If set, /predict requires this key in the X-API-Key header. Unset leaves the endpoint "
        "open, which is only appropriate behind a gateway that terminates auth itself.",
    )

    @property
    def artifacts_path_obj(self) -> Path:
        return Path(self.artifacts_path)

    @property
    def name_index_path_obj(self) -> Path:
        return Path(self.name_index_path)


# MLflow raises RESOURCE_DOES_NOT_EXIST when the registered model doesn't
# exist at all, and INVALID_PARAMETER_VALUE when it exists but has no
# "champion" alias yet -- the latter is what actually happens here, since
# log_model auto-creates the registered model just above this lookup.
NO_CHAMPION_YET_ERROR_CODES = {"RESOURCE_DOES_NOT_EXIST", "INVALID_PARAMETER_VALUE"}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached: constructing Settings re-reads .env from disk and re-validates,
    measured at ~1 ms, and /predict asks for it on every request. Config is
    read once at startup by design; tests that mutate the environment call
    ``get_settings.cache_clear()`` (see tests/conftest.py)."""
    return Settings()
