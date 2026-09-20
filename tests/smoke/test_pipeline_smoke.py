"""End-to-end smoke test: train (production mode) -> MLflow registration ->
serving artifact resolution via the registry's 'champion' alias -> a real
/predict call through the FastAPI app.

Training must use the real tabpfn-client API (fit_() calls it
unconditionally — see tests/parity/test_train_parity.py's docstring), so
this is gated on a real TABPFN_TOKEN. Serving is deliberately pinned to the
stub transport (TABPFN_TRANSPORT=stub) so the serving leg itself stays
network-free, fast, and deterministic — isolating "does the pipeline
plumbing work end to end" from "is the live TabPFN call itself healthy",
which the parity test and the load test already cover separately.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from bl_ranking.train_pipeline import run_training
from tests.conftest import requires_tabpfn_token


@requires_tabpfn_token
def test_train_register_serve_end_to_end(tmp_path, synthetic_bl_csv, example_user_data, monkeypatch):
    input_path, input_file = synthetic_bl_csv
    artifacts_path = str(tmp_path / "artifacts") + "/"
    registry_name = "smoke_test_bl_lead_classifier"

    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setenv("MLFLOW_REGISTRY_MODEL_NAME_CLASSIFIER", registry_name)

    version = run_training(
        mode="production",
        input_path=input_path,
        input_file=input_file,
        output_predictors_path=artifacts_path,
        build_surrogate=False,
        skip_quality_gate=True,
    )
    assert version is not None

    monkeypatch.setenv("ARTIFACTS_PATH", artifacts_path)
    monkeypatch.setenv("TABPFN_TRANSPORT", "stub")

    from bl_ranking import api

    with TestClient(api.app) as client:
        ready = client.get("/ready")
        assert ready.status_code == 200
        assert ready.json()["model_version"] == version
        # The startup canary must have produced a real prediction, not just
        # loaded files — the distinction an evicted hosted fit turns on.
        assert ready.json()["canary_ok"] is True

        resp = client.post(
            "/predict",
            json={**example_user_data, "session_dt": "2026-01-06T19:24:22", "register_date": "2026-01-06T19:26:07",
                  "conversion_dt": "2026-01-06T19:26:10"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["model_version"] == version
        assert isinstance(body["rankings"], dict)
        # The tier that answered must be reported, so a degraded ranking is
        # never indistinguishable from a healthy one at the call site.
        assert body["payout_tier"] == "stub"
        assert body["fallback_used"] is False
