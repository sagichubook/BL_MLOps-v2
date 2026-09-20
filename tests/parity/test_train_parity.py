"""Parity test: train_pipeline.py calls the vendored BLPayoutModelsFit with
the exact same constructor arguments and an unmodified fit_() — this test
proves that empirically rather than just by code inspection, by running the
original class directly and via the wrapper on the *same* synthetic data
and comparing the deterministic parts exactly.

Requires a real TABPFN_TOKEN: fit_() calls the hosted tabpfn-client API
unconditionally (both modes), so this cannot run against a stub without
modifying vendored code. Skipped automatically when no token is set —
see tests/conftest.py.
"""
from __future__ import annotations

import json
import os

import joblib
import pandas as pd
from catboost import CatBoostClassifier

from bl_ranking.original.bl_models_train import BLPayoutModelsFit
from bl_ranking.train_pipeline import run_training
from tests.conftest import requires_tabpfn_token


@requires_tabpfn_token
def test_production_mode_wrapper_matches_direct_original_call(tmp_path, synthetic_bl_csv, monkeypatch):
    input_path, input_file = synthetic_bl_csv

    direct_out = str(tmp_path / "direct") + "/"
    wrapped_out = str(tmp_path / "wrapped") + "/"

    # (a) run the vendored class directly, completely unwrapped
    direct_model = BLPayoutModelsFit(input_path, input_file, direct_out, train_test=False)
    direct_model.fit_()

    # (b) run the exact same class through train_pipeline.run_training
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    # skip_quality_gate: the synthetic fixture is deliberately tiny, and the
    # production gate (rightly) refuses a 300-row export. build_surrogate is
    # off because this test is about the vendored fit_() being untouched, not
    # about distillation — which has its own coverage.
    run_training(
        mode="production",
        input_path=input_path,
        input_file=input_file,
        output_predictors_path=wrapped_out,
        build_surrogate=False,
        skip_quality_gate=True,
    )

    # --- exact-equality comparisons of the deterministic parts ---
    direct_clients = pd.read_csv(direct_out + "all_clients.csv")
    wrapped_clients = pd.read_csv(wrapped_out + "all_clients.csv")
    pd.testing.assert_frame_equal(
        direct_clients.sort_values("client_name").reset_index(drop=True),
        wrapped_clients.sort_values("client_name").reset_index(drop=True),
    )

    direct_context = joblib.load(direct_out + "payout_tfm_context.joblib")
    wrapped_context = joblib.load(wrapped_out + "payout_tfm_context.joblib")
    assert direct_context["columns"] == wrapped_context["columns"]
    pd.testing.assert_frame_equal(
        direct_context["x"].reset_index(drop=True), wrapped_context["x"].reset_index(drop=True)
    )

    direct_cb = CatBoostClassifier(allow_writing_files=False)
    direct_cb.load_model(direct_out + "CB_bl_lead.cbm", format="cbm")
    wrapped_cb = CatBoostClassifier(allow_writing_files=False)
    wrapped_cb.load_model(wrapped_out + "CB_bl_lead.cbm", format="cbm")

    fixed_frame = direct_context["x"][direct_context["columns"]].head(20)
    direct_proba = direct_cb.predict_proba(fixed_frame)
    wrapped_proba = wrapped_cb.predict_proba(fixed_frame)
    assert direct_proba.tolist() == wrapped_proba.tolist(), (
        "CatBoost predict_proba differs between the direct vendored call and the wrapped call "
        "on identical input data — the wrapper must not change training behavior."
    )

    # --- the wrapper's additive artifacts exist and are self-consistent ---
    assert (tmp_path / "wrapped" / "MODEL_VERSION.json").exists() or os.path.exists(wrapped_out + "MODEL_VERSION.json")
    version_info = json.loads(open(wrapped_out + "MODEL_VERSION.json").read())
    assert version_info["model_version"]
