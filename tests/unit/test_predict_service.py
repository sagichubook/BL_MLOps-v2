"""Regression tests for the two documented deviations in predict_service.py.

The sharpest one: bl_exp_payout_predictor.py:58 originally reads a bare
module-level `user_data` global instead of `self.user_data`. In a
long-lived server, that either crashes or — worse — silently scores every
request against whichever user's data last touched the global. This test
scores two different users back-to-back in the *same* process and asserts
each gets its own answer, with no cross-contamination.
"""
from __future__ import annotations

import logging

import pandas as pd
import pytest
from catboost import CatBoostClassifier

from bl_ranking.payout_transport import StubPayoutRegressor
from bl_ranking.predict_service import SafeBLPayoutModelsPredict, ServingArtifacts, rank_brands

TFM_COLUMNS = [
    "campaign_id", "page", "city", "device_type", "sub1", "sub2", "sub3",
    "business_type", "industry", "loan_reason", "cellphone_prefix",
    "country_state", "credit_score_num", "loan_amount_num", "monthly_revenue_num",
    "time_in_business_num", "session_day", "session_day_of_week",
    "session_hour", "from_start_to_register", "ratio_loan_amount_revenue",
    "gender", "fname_len", "lname_len", "client_name",
]


def _tiny_catboost() -> CatBoostClassifier:
    # Minimal model just so predict_proba() runs; values don't need to be
    # meaningful for this test, only deterministic given the same input row.
    x = pd.DataFrame(
        {
            "campaign_id": [1, 2, 3, 4],
            "page": ["p1", "p2", "p1", "p2"],
            "city": ["a", "b", "a", "b"],
            "device_type": ["mobile", "desktop", "mobile", "desktop"],
            "sub1": ["1", "2", "1", "2"],
            "sub2": ["1", "2", "1", "2"],
            "sub3": ["1", "2", "1", "2"],
            "business_type": ["llc", "corp", "llc", "corp"],
            "industry": ["a", "b", "a", "b"],
            "loan_reason": ["a", "b", "a", "b"],
            "cellphone_prefix": ["786", "212", "786", "212"],
            "country_state": ["fl", "ny", "fl", "ny"],
            "credit_score_num": [501, 721, 501, 721],
            "loan_amount_num": [17500, 47500, 17500, 47500],
            "monthly_revenue_num": [5000, 35000, 5000, 35000],
            "time_in_business_num": [3, 36, 3, 36],
            "session_day": [1, 2, 1, 2],
            "session_day_of_week": ["Monday", "Tuesday", "Monday", "Tuesday"],
            "session_hour": [10, 11, 10, 11],
            "from_start_to_register": [30.0, 60.0, 30.0, 60.0],
            "ratio_loan_amount_revenue": [3.5, 1.4, 3.5, 1.4],
            "gender": ["male", "female", "male", "female"],
            "fname_len": [4, 5, 4, 5],
            "lname_len": [5, 6, 5, 6],
            "client_name": ["brandA", "brandB", "brandA", "brandB"],
        }
    )
    y = [0, 1, 1, 0]
    cat_features = x.select_dtypes(include=["object"]).columns.tolist()
    model = CatBoostClassifier(iterations=5, depth=2, verbose=False, allow_writing_files=False)
    model.fit(x, y, cat_features=cat_features)
    return model


@pytest.fixture
def artifacts(tmp_path) -> ServingArtifacts:
    predictors_path = str(tmp_path) + "/"
    all_clients = pd.DataFrame({"client_name": ["brandA", "brandB", "other"]})
    all_clients.to_csv(tmp_path / "all_clients.csv", index=False)
    return ServingArtifacts(
        cb_model=_tiny_catboost(),
        payout_predictor=StubPayoutRegressor(),
        tfm_columns=TFM_COLUMNS,
        predictors_path=predictors_path,
        all_clients=all_clients,
        catboost_version="test",
        tabpfn_transport_mode="stub",
        model_version="test-v1",
        loaded_at=__import__("datetime").datetime.now(),
    )


def _user(fname: str, credit_score: str) -> dict:
    return {
        "session_dt": "2026-01-06 19:24:22",
        "conversion_dt": None,
        "register_date": "2026-01-06 19:26:07",
        "campaign_id": 1,
        "page": "p1",
        "auto_city": "Fort Lauderdale",
        "auto_country": "United States",
        "auto_state": "Florida",
        "device_type": "mobile",
        "sub1": "1",
        "sub2": "1",
        "sub3": "1",
        "business_type": "C Corporation",
        "credit_score": credit_score,
        "industry": "construction",
        "loan_amount": "$25,000 - $49,999",
        "loan_reason": "Equipment purchase",
        "monthly_revenue": "$20,000 - $49,999",
        "time_in_business": "2+ years",
        "fname": fname,
        "lname": "Rodriguez",
        "cellphone": 7869914030,
    }


def test_two_users_scored_back_to_back_do_not_contaminate_each_other(artifacts):
    user_a = _user("Alice", "Excellent (720+)")
    user_b = _user("Bob", "Very Poor - Under 550")

    result_a1 = rank_brands(user_a, artifacts, logging.getLogger("test"))
    result_b = rank_brands(user_b, artifacts, logging.getLogger("test"))
    result_a2 = rank_brands(user_a, artifacts, logging.getLogger("test"))

    # Same user, same inputs, called before and after a different user in
    # the same process -> identical result. This is exactly what the
    # global-variable bug would break.
    assert result_a1.rankings == result_a2.rankings
    # Different users (different credit_score/fname -> different features)
    # get different rankings — this is what the global-variable bug breaks:
    # under the bug, result_b would just be a repeat of result_a1.
    assert result_a1.rankings != result_b.rankings


def test_import_preprocess_uses_self_user_data_not_a_module_global(artifacts):
    # If this were still reading the bare `user_data` global, this call
    # would raise NameError (no such global exists in this test process).
    service = SafeBLPayoutModelsPredict(artifacts, _user("Carol", "Fair (600-649)"), logging.getLogger("test"))
    df = service.import_preprocess()
    assert (df["fname"] == "Carol").all()


def test_missing_register_date_is_rejected(artifacts):
    bad_user = _user("Dan", "Fair (600-649)")
    bad_user["register_date"] = None
    service = SafeBLPayoutModelsPredict(artifacts, bad_user, logging.getLogger("test"))
    with pytest.raises(ValueError):
        service.import_preprocess()
