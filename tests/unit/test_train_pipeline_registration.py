"""Regression coverage for _register_and_alias's MLflow error-code handling
-- no network/TABPFN_TOKEN required (only mlflow + catboost, against a local
sqlite tracking store), so this runs in default CI unlike the token-gated
parity/smoke tests that originally caught this bug.

The bug: mlflow.catboost.log_model(..., registered_model_name=...) auto-
creates the registered model, so by the time _register_and_alias looks up
the "champion" alias on a *brand-new* model, MLflow raises MlflowException
with error_code="INVALID_PARAMETER_VALUE" (alias not set on an existing
model) -- not "RESOURCE_DOES_NOT_EXIST" (which only fires for a model that
doesn't exist at all, e.g. a typo'd name). A too-narrow except clause that
only recognized RESOURCE_DOES_NOT_EXIST treated every first-time
registration as a fatal error.
"""
from __future__ import annotations

import pytest
from catboost import CatBoostClassifier

from bl_ranking.config import Settings
from bl_ranking.train_pipeline import _register_and_alias


def _sqlite_registry_supported() -> bool:
    try:
        import mlflow
        mlflow.set_tracking_uri("sqlite:///:memory:")
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        client.get_registered_model("___probe___")
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _sqlite_registry_supported(),
    reason="mlflow-skinny does not support sqlite model registry",
)


def _tiny_cbm(path) -> None:
    x = [[0, 1], [1, 0], [0, 0], [1, 1]]
    y = [0, 1, 0, 1]
    model = CatBoostClassifier(iterations=3, depth=2, verbose=False, allow_writing_files=False)
    model.fit(x, y)
    model.save_model(str(path), format="cbm")


def test_register_and_alias_first_registration_does_not_raise(tmp_path, monkeypatch):
    import mlflow

    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.set_experiment("test_register_and_alias")

    settings = Settings(mlflow_registry_model_name_classifier="test_first_reg_model")
    cbm_path = tmp_path / "model.cbm"
    _tiny_cbm(cbm_path)

    with mlflow.start_run():
        version = _register_and_alias(cbm_path, settings)

    assert version == "1"

    from mlflow import MlflowClient

    client = MlflowClient()
    champion = client.get_model_version_by_alias("test_first_reg_model", "champion")
    assert str(champion.version) == "1"


def test_register_and_alias_second_registration_sets_previous(tmp_path, monkeypatch):
    import mlflow

    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.set_experiment("test_register_and_alias")

    settings = Settings(mlflow_registry_model_name_classifier="test_second_reg_model")
    cbm_path = tmp_path / "model.cbm"
    _tiny_cbm(cbm_path)

    with mlflow.start_run():
        v1 = _register_and_alias(cbm_path, settings)
    with mlflow.start_run():
        v2 = _register_and_alias(cbm_path, settings)

    assert v1 == "1"
    assert v2 == "2"

    from mlflow import MlflowClient

    client = MlflowClient()
    assert str(client.get_model_version_by_alias("test_second_reg_model", "champion").version) == "2"
    assert str(client.get_model_version_by_alias("test_second_reg_model", "previous").version) == "1"
