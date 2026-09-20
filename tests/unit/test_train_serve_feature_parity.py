"""Train/serve feature-engineering parity.

bl_models_train.py and bl_exp_payout_predictor.py are two independently
vendored implementations of the same feature-engineering steps (survey
imputation, numeric bucketing, time features, gender lookup, ...) — nothing
in the vendored code links them, so a change to one that isn't mirrored in
the other would silently skew serving predictions away from what the model
was trained on. This is the single highest correctness risk in the system
and, unlike everything else the vendored scripts do, it has no test.

This test locks in that, today, the two paths produce identical engineered
feature vectors for the same underlying (user, brand) row — run through each
*real* code path (the vendored training class directly, unmodified; and
``SafeBLPayoutModelsPredict``, the actual production serving wrapper) — so a
future edit to either that breaks that equivalence fails CI instead of
shipping a silent train/serve skew. No network/TabPFN call is involved;
``bl_preprocessing()`` on both classes runs entirely before either model
sees data.
"""
from __future__ import annotations

import datetime as dt
import logging

import pandas as pd

from bl_ranking.original.bl_models_train import BLPayoutModelsFit
from bl_ranking.predict_service import SafeBLPayoutModelsPredict, ServingArtifacts

CLIENT_NAME = "brandA"

# The exact request fields BLPayoutModelsPredict.import_preprocess() reads
# (bl_exp_payout_predictor.py:60-64) — used verbatim as the serving-side
# input and, with training-only fields added, as the one training CSV row.
_RAW_FIELDS = {
    "session_dt": "2026-01-06 19:24:22",
    "conversion_dt": "2026-01-06 19:26:10",
    "register_date": "2026-01-06 19:26:07",
    "campaign_id": 120227360861540306,
    "page": "top10us.com/app/business-loans-v2",
    "auto_city": "Fort Lauderdale",
    "auto_country": "United States",
    "auto_state": "Florida",
    "device_type": "mobile",
    "sub1": "1121993",
    "sub2": "01121993 Ad set",
    "sub3": "1513124082",
    "business_type": "C Corporation",
    "credit_score": "Very Poor - Under 550",
    "industry": "construction",
    "loan_amount": "$25,000 - $49,999",
    "loan_reason": "Equipment purchase",
    "monthly_revenue": "$20,000 - $49,999",
    "time_in_business": "2+ years",
    "fname": "Rigoberto",
    "lname": "Rodriguez",
    "cellphone": 7869914030,
}

# bl_models_train.py:236-241 (train_columns) and
# bl_exp_payout_predictor.py:207-212 (pred_columns) — the same 24 columns,
# same order, in both vendored scripts.
FEATURE_COLUMNS = [
    "campaign_id", "page", "city", "device_type", "sub1", "sub2", "sub3",
    "business_type", "industry", "loan_reason", "cellphone_prefix",
    "country_state", "credit_score_num", "loan_amount_num", "monthly_revenue_num",
    "time_in_business_num", "session_day", "session_day_of_week",
    "session_hour", "from_start_to_register", "ratio_loan_amount_revenue",
    "gender", "fname_len", "lname_len", "client_name",
]


def _train_row(tmp_path) -> pd.Series:
    """Runs one raw row through the real TRAINING preprocessing path
    (BLPayoutModelsFit.bl_preprocessing — vendored, unchanged)."""
    target = dict(_RAW_FIELDS, session_id="s-target", client_name=CLIENT_NAME,
                   payout=42.0, disposition="Lead", disposition_source=CLIENT_NAME)
    # A second, later row purely so split_by_time's "last 7 days" test window
    # lands on it instead of our target row — bl_preprocessing always time-
    # splits regardless of train_test, so a single-row CSV would otherwise
    # put the only row in the held-out test split, not x_train.
    filler = dict(target, session_id="s-filler", session_dt="2026-02-20 12:00:00",
                  conversion_dt="2026-02-20 12:02:00", register_date="2026-02-20 12:02:30")
    tmp_path.mkdir(parents=True, exist_ok=True)
    input_path = str(tmp_path) + "/"
    input_file = "train_rows.csv"
    pd.DataFrame([target, filler]).to_csv(input_path + input_file, index=False)

    output_path = str(tmp_path / "out") + "/"
    model = BLPayoutModelsFit(input_path, input_file, output_path, train_test=False)
    x_train, _y_train, _x_test, _y_test = model.bl_preprocessing()
    assert len(x_train) == 1, "fixture is stale: expected exactly the target row in x_train"
    return x_train.iloc[0]


def _serve_row(tmp_path) -> pd.Series:
    """Runs the same raw fields through the real SERVING preprocessing path
    (SafeBLPayoutModelsPredict — the actual production wrapper, not the bare
    vendored class)."""
    all_clients = pd.DataFrame({"client_name": [CLIENT_NAME]})
    artifacts = ServingArtifacts(
        cb_model=None,
        payout_predictor=None,
        tfm_columns=FEATURE_COLUMNS,
        predictors_path=str(tmp_path) + "/",
        all_clients=all_clients,
        catboost_version="test",
        tabpfn_transport_mode="stub",
        model_version="test",
        loaded_at=dt.datetime.now(),
    )
    predictor = SafeBLPayoutModelsPredict(artifacts, dict(_RAW_FIELDS), logging.getLogger("test.parity"))
    bl_data = predictor.bl_preprocessing()  # inherited from BLPayoutModelsPredict, unchanged
    assert len(bl_data) == 1
    return bl_data.iloc[0]


def test_training_and_serving_produce_identical_engineered_features(tmp_path):
    train_row = _train_row(tmp_path / "train")
    serve_row = _serve_row(tmp_path / "serve")
    mismatches = [
        f"{col!r}: train={train_row[col]!r} serve={serve_row[col]!r}"
        for col in FEATURE_COLUMNS
        if train_row[col] != serve_row[col]
    ]
    assert not mismatches, "train/serve feature skew:\n" + "\n".join(mismatches)
