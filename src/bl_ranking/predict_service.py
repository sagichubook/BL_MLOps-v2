"""Serving wrapper around the vendored ``BLPayoutModelsPredict``.

Five overrides; every other method -- feature engineering, ranking maths -- is
inherited unchanged.

1. ``import_preprocess`` reads a module-level ``user_data`` global instead of
   ``self.user_data`` (bl_exp_payout_predictor.py:58). It works only because
   ``__main__`` defines one; in a server it raises ``NameError`` or silently
   scores every request against whichever user last set the global.
2. Artifacts load once at startup: the original fits a ``TabPFNRegressor``
   inside ``load_models()``, which ``predict_()`` calls per request (13 s).
3. ``all_clients`` comes from that startup load, not a per-request CSV read.
4. No per-instance logger: the original's ``setup_bl_logger()`` clears the
   *root* handlers and opens a log file per instantiation. Warnings still
   reach structured logs via ``logging.captureWarnings(True)``.
5. ``ValueError`` rather than bare ``Exception`` for a missing
   ``register_date``, so api.py returns 422 there without swallowing real
   bugs as 500s.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from bl_ranking.original.bl_exp_payout_predictor import BLPayoutModelsPredict
from bl_ranking.payout_transport import PayoutPredictor, ResilientPayoutRegressor

logger = logging.getLogger("bl_ranking.predict_service")


@dataclass
class ServingArtifacts:
    """Loaded once at startup, reused by every request."""

    cb_model: Any
    payout_predictor: PayoutPredictor
    tfm_columns: list[str]
    predictors_path: str  # kept for parity with the original constructor signature; not re-read per request
    all_clients: pd.DataFrame  # loaded once at startup instead of re-reading all_clients.csv per request
    catboost_version: str
    tabpfn_transport_mode: str
    model_version: str
    loaded_at: datetime
    surrogate_meta: dict | None = None


class SafeBLPayoutModelsPredict(BLPayoutModelsPredict):
    def __init__(self, artifacts: ServingArtifacts, user_data: dict, request_logger: logging.Logger) -> None:
        # Skips the original __init__/setup_bl_logger (deviation 4), which
        # clears the root logger's handlers per instantiation.
        self.predictors_path = artifacts.predictors_path
        self.user_data = user_data
        self.user_data_file = user_data
        self.logger = request_logger
        self._artifacts = artifacts

    def import_preprocess(self) -> pd.DataFrame:
        # Verbatim copy of the original, with deviations 1, 3 and 5.
        bl_data = pd.DataFrame([self.user_data])
        all_clients = self._artifacts.all_clients
        needed_columns = [
            "session_dt", "conversion_dt", "register_date",
            "campaign_id", "page", "auto_city", "auto_country", "auto_state", "device_type", "sub1",
            "sub2", "sub3",
            "business_type", "credit_score", "industry", "loan_amount", "loan_reason", "monthly_revenue",
            "time_in_business", "fname", "lname", "cellphone",
        ]
        bl_data = bl_data[needed_columns]
        if bl_data["register_date"].isna().any():
            raise ValueError("user cannot be a lead - register_date is absent")

        bl_data = bl_data.merge(all_clients, how="cross")
        bl_data = bl_data[bl_data["client_name"] != "other"]
        self.logger.info("raw data rows: %s", bl_data.shape[0])
        bl_data = bl_data.rename(columns={"auto_city": "city", "auto_state": "state", "auto_country": "country"})
        self.logger.info("register date exists - can be leads: %s", bl_data.shape[0])
        bl_data[["country", "state", "city", "sub1", "sub2", "sub3"]] = (
            bl_data[["country", "state", "city", "sub1", "sub2", "sub3"]].fillna("Other")
        )
        bl_data["country_state"] = np.where(
            bl_data["country"] == "United States", bl_data["state"], bl_data["country"]
        )
        bl_data["session_dt"] = pd.to_datetime(bl_data["session_dt"], errors="coerce")
        bl_data["register_date"] = pd.to_datetime(bl_data["register_date"], errors="coerce")
        bl_data["sub1"] = bl_data["sub1"].astype(str)
        bl_data["sub2"] = bl_data["sub2"].astype(str)
        bl_data["sub3"] = bl_data["sub3"].astype(str)
        bl_data["cellphone_prefix"] = bl_data["cellphone"].astype(int).astype(str).str[:3].astype(str)
        return bl_data

    def load_models(self):
        return self._artifacts.payout_predictor, self._artifacts.tfm_columns, self._artifacts.cb_model


@dataclass
class RankingResponse:
    rankings: dict[str, dict[str, float]]
    model_version: str
    fallback_used: bool
    latency_ms: float
    payout_tier: str = "unknown"  # which rung answered


def rank_brands(user_data: dict, artifacts: ServingArtifacts, request_logger: logging.Logger | None = None) -> RankingResponse:
    request_logger = request_logger or logger
    start = datetime.now(UTC)

    if isinstance(artifacts.payout_predictor, ResilientPayoutRegressor):
        # Before predict_(), which can short-circuit without calling predict()
        # and would inherit the previous outcome on this thread.
        artifacts.payout_predictor.reset_call_state()

    predictor = SafeBLPayoutModelsPredict(artifacts, user_data, request_logger)
    rankings = predictor.predict_()  # inherited, unchanged: bl_preprocessing + prediction_expected_payout

    fallback_used = False
    payout_tier = artifacts.tabpfn_transport_mode
    if isinstance(artifacts.payout_predictor, ResilientPayoutRegressor):
        fallback_used = artifacts.payout_predictor.last_call_used_fallback
        outcome = artifacts.payout_predictor.last_outcome
        if outcome is not None:
            payout_tier = outcome.tier

    latency_ms = (datetime.now(UTC) - start).total_seconds() * 1000
    if not isinstance(rankings, dict) or not rankings or "expected_payout" in rankings:
        # predict_() returns {"expected_payout": 0, "prob_lead": 0} when no
        # brand clears the 0.01 floor — an empty result, not an error.
        rankings = {}

    return RankingResponse(
        rankings=rankings,
        model_version=artifacts.model_version,
        fallback_used=fallback_used,
        latency_ms=latency_ms,
        payout_tier=payout_tier,
    )
