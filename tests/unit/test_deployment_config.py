"""Deployment wiring whose failure modes produce no error message: an
artifact root containers write past each other on, an env var nothing reads,
a missing COPY costing 1.9 GB per worker, a health check that lies."""
from __future__ import annotations

import re

import pytest
import yaml

from bl_ranking import REPO_ROOT

COMPOSE = yaml.safe_load((REPO_ROOT / "docker-compose.yml").read_text())


DOCKERFILE = (REPO_ROOT / "Dockerfile").read_text()
# Instructions only. Checking the raw text would match the word "curl" in a
# comment explaining why curl is not used.
DOCKERFILE_INSTRUCTIONS = "\n".join(
    line for line in DOCKERFILE.splitlines() if line.strip() and not line.lstrip().startswith("#")
)


# --------------------------------------------------------------------------
# MLflow artifact routing
# --------------------------------------------------------------------------


def _mlflow_script() -> str:
    command = COMPOSE["services"]["mlflow"]["command"]
    return command[-1] if isinstance(command, list) else command


def test_mlflow_command_is_a_single_shell_line():
    """A folded YAML block keeps *more-indented* continuation lines literal, so
    flags written under the command become separate shell commands. That
    happened: the server came up with no artifact root, no --serve-artifacts
    and bound to 127.0.0.1, and every substring assertion below still passed
    because the text was present — just inert."""
    script = _mlflow_script()
    assert "\n" not in script.strip(), (
        "the mlflow command spans multiple lines; each flag after the first becomes "
        "its own shell command and is silently ignored"
    )


def test_mlflow_artifact_root_uses_the_proxy_scheme():
    """A plain path makes clients bypass the server's HTTP API, so `train`
    writes to its own disk and `serve` finds nothing. Fatal across containers,
    invisible on one filesystem."""
    # Split on the server invocation so these are checked as *arguments to it*,
    # not merely as text somewhere in the script.
    script = _mlflow_script()
    assert "mlflow server" in script
    args = script.split("mlflow server", 1)[1]
    assert "--default-artifact-root mlflow-artifacts:/" in args
    assert "--artifacts-destination" in args
    assert "--serve-artifacts" in args


def test_mlflow_binds_where_other_containers_can_reach_it():
    """The default bind is 127.0.0.1, which is unreachable from the train and
    serve containers — and the failure looks like a slow health check."""
    args = _mlflow_script().split("mlflow server", 1)[1]
    assert "--host 0.0.0.0" in args
    assert "--port 5000" in args


def test_train_and_serve_point_at_the_mlflow_service():
    for service in ("train", "serve", "scheduler"):
        assert COMPOSE["services"][service]["environment"]["MLFLOW_TRACKING_URI"] == "http://mlflow:5000"


# --------------------------------------------------------------------------
# Load generator
# --------------------------------------------------------------------------


def test_loadtest_sets_the_env_var_locust_actually_reads():
    """Locust reads --host from LOCUST_HOST; TARGET_HOST is read by nothing."""
    env = COMPOSE["services"]["loadtest"]["environment"]
    assert "LOCUST_HOST" in env
    assert "TARGET_HOST" not in env


# --------------------------------------------------------------------------
# The name index must reach the images
# --------------------------------------------------------------------------


def test_image_ships_the_name_index_and_requires_it():
    """Without assets/ the vendored import falls back to the real library
    (~1.9 GB per process); BL_REQUIRE_NAME_INDEX makes that fail loudly
    instead of silently expensive."""
    assert "COPY assets/" in DOCKERFILE
    assert "BL_REQUIRE_NAME_INDEX=1" in DOCKERFILE


def test_image_pins_utc():
    """Every schedule and timestamp feature assumes it."""
    assert "TZ=UTC" in DOCKERFILE


def test_train_and_serve_are_stages_of_one_base():
    """Separate images would let the training and serving runtimes drift —
    the train/serve skew test_train_serve_feature_parity exists to catch. One
    base stage makes them identical by construction. (Layer storage is only
    deduplicated under BuildKit; see the Dockerfile.)"""
    assert "FROM python:3.14-slim AS base" in DOCKERFILE
    assert "FROM base AS serve" in DOCKERFILE
    assert "FROM base AS train" in DOCKERFILE


def test_image_needs_no_apt():
    """Every pinned dependency ships a cp314 wheel, so an apt layer bought
    only size and a dependency on a Debian mirror being reachable — which it
    is not from every build network."""
    assert "apt-get" not in DOCKERFILE_INSTRUCTIONS


def test_git_sha_is_baked_in():
    """.git is not in the build context, so without this every containerised
    run logs git_sha='unknown' and the provenance param is worthless."""
    assert "ARG GIT_SHA" in DOCKERFILE and "ENV GIT_SHA=${GIT_SHA}" in DOCKERFILE


# --------------------------------------------------------------------------
# Health checks
# --------------------------------------------------------------------------


def test_serve_healthcheck_gates_on_readiness_not_liveness():
    """/health only proves the process answers HTTP; /ready includes the
    canary, the only check that catches an evicted hosted fit."""
    test = COMPOSE["services"]["serve"]["healthcheck"]["test"]
    assert any("/ready" in part for part in test)
    assert not any("/health" in part for part in test)
    assert "/ready" in DOCKERFILE


def test_healthcheck_uses_the_interpreter_not_curl():
    """curl would mean an apt layer for one HTTP call; Python is already
    in the image."""
    assert "curl" not in DOCKERFILE_INSTRUCTIONS
    assert "curl" not in " ".join(COMPOSE["services"]["serve"]["healthcheck"]["test"])


# --------------------------------------------------------------------------
# Scheduler
# --------------------------------------------------------------------------


def test_scheduler_runs_unattended_and_restarts():
    scheduler = COMPOSE["services"]["scheduler"]
    assert scheduler["restart"] == "unless-stopped"
    assert "profiles" not in scheduler, "the weekly schedule must run by default, not behind a profile"


def test_scheduler_and_train_share_one_image():
    """Two images would let the scheduled run drift from the manual one."""
    assert COMPOSE["services"]["scheduler"]["build"] == COMPOSE["services"]["train"]["build"]


def test_builds_declare_host_network():
    """A rootless daemon's default bridge may have no DNS, so the pip layer —
    the only step needing a network — fails to resolve PyPI without this."""
    for service in ("train", "serve", "loadtest"):
        assert COMPOSE["services"][service]["build"].get("network") == "host"


# --------------------------------------------------------------------------
# Databricks bundle
# --------------------------------------------------------------------------


def test_databricks_bundle_is_valid_yaml_and_names_the_wheel_entry_point():
    job = yaml.safe_load((REPO_ROOT / "databricks" / "resources" / "bl_ranking_job.yml").read_text())
    task = job["resources"]["jobs"]["bl_ranking_weekly_train"]["tasks"][0]
    assert task["python_wheel_task"]["entry_point"] == "bl-ranking-train"
    assert task["python_wheel_task"]["package_name"] == "bl_ranking"


def test_databricks_entry_point_exists_in_pyproject():
    """Undeclared, the job fails at run time on a cluster a week later."""
    pyproject = (REPO_ROOT / "pyproject.toml").read_text()
    assert "bl-ranking-train = \"bl_ranking.train_pipeline:main\"" in pyproject


def test_databricks_job_is_single_node():
    """pandas/CatBoost on one machine; workers would cost money and do nothing."""
    job = yaml.safe_load((REPO_ROOT / "databricks" / "resources" / "bl_ranking_job.yml").read_text())
    cluster = job["resources"]["jobs"]["bl_ranking_weekly_train"]["job_clusters"][0]["new_cluster"]
    assert cluster["num_workers"] == 0
    assert cluster["custom_tags"]["ResourceClass"] == "SingleNode"


def test_databricks_job_alerts_on_failure():
    """A weekly job that stops looks healthy until serving goes stale."""
    job = yaml.safe_load((REPO_ROOT / "databricks" / "resources" / "bl_ranking_job.yml").read_text())
    spec = job["resources"]["jobs"]["bl_ranking_weekly_train"]
    assert spec["email_notifications"]["on_failure"]
    assert spec["max_concurrent_runs"] == 1


def test_databricks_serving_does_not_scale_to_zero():
    """A cold start of tens of seconds, on a path where the user waits."""
    serving = yaml.safe_load((REPO_ROOT / "databricks" / "resources" / "bl_ranking_serving.yml").read_text())
    entity = serving["resources"]["model_serving_endpoints"]["bl-ranking-predict"]["config"]["served_entities"][0]
    assert entity["scale_to_zero_enabled"] is False


def test_databricks_secret_is_referenced_not_inlined():
    text = (REPO_ROOT / "databricks" / "resources" / "bl_ranking_serving.yml").read_text()
    assert "{{secrets/" in text
    assert "TABPFN_TOKEN: eyJ" not in text and "TABPFN_TOKEN: sk-" not in text


def test_bundle_declares_the_wheel_artifact():
    bundle = yaml.safe_load((REPO_ROOT / "databricks" / "databricks.yml").read_text())
    assert bundle["artifacts"]["bl_ranking_wheel"]["type"] == "whl"
    assert bundle["targets"]["prod"]["mode"] == "production"


# --------------------------------------------------------------------------
# Credential hygiene
# --------------------------------------------------------------------------

# A secret-shaped value: an opaque run of token characters, long enough not to
# be a variable name, a path or an example. Code that merely *names*
# TABPFN_TOKEN is legitimate and everywhere, so matching the name alone would
# flag a dozen honest lines — and a scanner that cries wolf gets switched off.
_SECRET_KEY = re.compile(r"(TOKEN|SECRET|PASSWORD|API_KEY)\s*[:=]\s*[\"']?([A-Za-z0-9._\-+/]{24,})", re.I)
_CONFIGISH = (".env", ".example", ".yml", ".yaml", ".toml", ".sh", ".txt", ".cfg", ".ini")


def _secret_findings(text: str) -> list[str]:
    findings = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or "{{secrets/" in stripped or "${" in stripped:
            continue
        match = _SECRET_KEY.search(stripped)
        if match and "..." not in match.group(2):
            findings.append(stripped[:70])
    return findings


def test_the_secret_scanner_actually_detects_a_secret():
    """Guards the guard: a scanner that matches nothing passes vacuously."""
    assert _secret_findings("TABPFN_TOKEN=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9abcdef")
    assert _secret_findings('  api_key: "sk-0123456789abcdef0123456789abcdef"')
    # ...and stays quiet on the honest mentions that fill the codebase.
    assert not _secret_findings('raise RuntimeError(f"transport={mode} requires TABPFN_TOKEN")')
    assert not _secret_findings("TABPFN_TOKEN=")
    assert not _secret_findings("TABPFN_TOKEN=... pytest tests/parity -q")
    assert not _secret_findings('TABPFN_TOKEN: "{{secrets/bl_ranking/tabpfn_token}}"')


def test_env_file_is_ignored_and_untracked():
    """.env holds a real TABPFN_TOKEN during development. Gitignoring it is not
    enough on its own — this asserts git agrees, and that no past commit
    slipped it into the index."""
    pytest.importorskip("dulwich")
    from dulwich.ignore import IgnoreFilterManager
    from dulwich.repo import Repo

    repo = Repo(str(REPO_ROOT))
    assert IgnoreFilterManager.from_repo(repo).is_ignored(".env"), ".env must be gitignored"

    tracked = {path.decode() for path in repo.open_index()}
    assert ".env" not in tracked, ".env is tracked — rotate the token and remove it from the index"
    assert ".env.example" in tracked
    for line in (REPO_ROOT / ".env.example").read_text().splitlines():
        if line.startswith("TABPFN_TOKEN"):
            assert line.strip() == "TABPFN_TOKEN=", "the example must not ship a token value"


def test_no_secret_shaped_values_are_tracked():
    """A token pasted into a tracked config file is the failure this guards."""
    pytest.importorskip("dulwich")
    from dulwich.repo import Repo

    repo = Repo(str(REPO_ROOT))
    suspicious = []
    for path in sorted({p.decode() for p in repo.open_index()}):
        if path.endswith(_CONFIGISH):
            for finding in _secret_findings((REPO_ROOT / path).read_text(errors="ignore")):
                suspicious.append(f"{path}: {finding}")
    assert not suspicious, f"possible committed secrets: {suspicious}"
