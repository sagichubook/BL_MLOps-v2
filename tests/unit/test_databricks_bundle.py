"""Validate the bundle against Databricks' own API models.

`tests/unit/test_deployment_config.py` asserts the bundle says the right
*things* (UTC, single node, no scale-to-zero). This asserts Databricks would
*accept* it — a misspelled field is silently dropped at deploy time, so the
job runs with a default nobody chose and the mistake surfaces a week later on
a cluster.

The check is a round trip through the SDK's typed models: `from_dict` keeps
only fields the model declares, so any key that survives the YAML but vanishes
from `as_dict()` is a key Databricks does not know. Comparing key structure
rather than values keeps enum normalisation and default-filling out of it.

This still does not prove a live deploy works — see the `databricks-validate`
CI job, which runs the real `databricks bundle validate` when a workspace
secret is configured.
"""
from __future__ import annotations

import re

import pytest
import yaml
from databricks.sdk.service import jobs, serving

from bl_ranking import REPO_ROOT

RESOURCES = REPO_ROOT / "databricks" / "resources"

# Bundle substitution syntax (${var.x}, ${bundle.target}, ${workspace.host})
# is resolved by the CLI at deploy time and is not valid API input, so it is
# replaced before the specs reach the models.
#
# ${var.x} resolves to that variable's *declared default* rather than a dummy.
# That is not cosmetic: several fields are enums, and the SDK silently drops a
# value it cannot parse — substituting a generic placeholder would make a
# perfectly good `pause_status: PAUSED` look like an unknown field, and worse,
# would stop the test noticing if a declared default were genuinely invalid.
_SUBST = re.compile(r"\$\{([^}]+)\}")


def _declared_defaults() -> dict[str, str]:
    """Variable defaults from databricks.yml and the resource files."""
    defaults: dict[str, str] = {}
    for path in [REPO_ROOT / "databricks" / "databricks.yml", *sorted(RESOURCES.glob("*.yml"))]:
        doc = yaml.safe_load(path.read_text()) or {}
        for name, spec in (doc.get("variables") or {}).items():
            if isinstance(spec, dict) and "default" in spec:
                defaults[name] = str(spec["default"])
    return defaults


def _substitute(node, defaults: dict[str, str]):
    if isinstance(node, dict):
        return {k: _substitute(v, defaults) for k, v in node.items()}
    if isinstance(node, list):
        return [_substitute(v, defaults) for v in node]
    if isinstance(node, str):
        def resolve(match: re.Match) -> str:
            ref = match.group(1)
            if ref.startswith("var."):
                return defaults.get(ref[4:], "placeholder")
            return "placeholder"
        return _SUBST.sub(resolve, node)
    return node


def _dropped_keys(original, roundtripped, path="") -> list[str]:
    """Keys the model did not recognise, reported with their full path."""
    dropped = []
    if isinstance(original, dict):
        if not isinstance(roundtripped, dict):
            return dropped
        for key, value in original.items():
            here = f"{path}.{key}" if path else key
            if key not in roundtripped:
                dropped.append(here)
            else:
                dropped += _dropped_keys(value, roundtripped[key], here)
    elif isinstance(original, list) and isinstance(roundtripped, list):
        for i, value in enumerate(original):
            if i < len(roundtripped):
                dropped += _dropped_keys(value, roundtripped[i], f"{path}[{i}]")
    return dropped


def _load(name: str) -> dict:
    return _substitute(yaml.safe_load((RESOURCES / name).read_text()), _declared_defaults())


@pytest.fixture(scope="module")
def job_spec() -> dict:
    return _load("bl_ranking_job.yml")["resources"]["jobs"]["bl_ranking_weekly_train"]


@pytest.fixture(scope="module")
def serving_spec() -> dict:
    return _load("bl_ranking_serving.yml")["resources"]["model_serving_endpoints"]["bl-ranking-predict"]


def test_job_spec_is_accepted_by_the_jobs_model(job_spec):
    settings = jobs.JobSettings.from_dict(job_spec)
    dropped = _dropped_keys(job_spec, settings.as_dict())
    assert not dropped, f"Databricks does not define these job fields (typo or wrong nesting): {dropped}"


def test_serving_config_is_accepted_by_the_serving_model(serving_spec):
    config = serving.EndpointCoreConfigInput.from_dict(serving_spec["config"])
    dropped = _dropped_keys(serving_spec["config"], config.as_dict())
    assert not dropped, f"Databricks does not define these serving fields: {dropped}"


def test_a_typo_would_actually_be_caught(job_spec):
    """Guards the guard. If the round trip stopped detecting unknown keys this
    file would pass while proving nothing."""
    broken = {**job_spec, "timeout_secconds": 10800}
    dropped = _dropped_keys(broken, jobs.JobSettings.from_dict(broken).as_dict())
    assert "timeout_secconds" in dropped


def test_schedule_survives_the_round_trip(job_spec):
    """The quartz expression and timezone are the whole point of the job."""
    settings = jobs.JobSettings.from_dict(job_spec)
    assert settings.schedule.quartz_cron_expression == "0 0 5 ? * SUN"
    assert settings.schedule.timezone_id == "UTC"
    # A pause_status the SDK cannot parse is dropped rather than rejected, so
    # the declared default has to be a real PauseStatus or the job would
    # silently deploy with whatever the server defaults to.
    assert settings.schedule.pause_status in set(jobs.PauseStatus)


def test_wheel_task_parses_with_the_declared_entry_point(job_spec):
    task = jobs.JobSettings.from_dict(job_spec).tasks[0]
    assert task.python_wheel_task.package_name == "bl_ranking"
    assert task.python_wheel_task.entry_point == "bl-ranking-train"
    assert task.python_wheel_task.parameters[:2] == ["--mode", "production"]


def test_served_entity_parses_and_keeps_its_environment(serving_spec):
    entity = serving.EndpointCoreConfigInput.from_dict(serving_spec["config"]).served_entities[0]
    assert entity.scale_to_zero_enabled is False
    # The transport choice has to reach the endpoint, or Databricks serving
    # would silently fall back to the hosted path this system moved off.
    assert entity.environment_vars["TABPFN_TRANSPORT"] == "surrogate"


def test_registered_model_is_declared_in_unity_catalog():
    spec = _load("bl_ranking_serving.yml")["resources"]["registered_models"]["bl_lead_classifier"]
    assert {"name", "catalog_name", "schema_name"} <= set(spec)
