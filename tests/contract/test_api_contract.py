from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest
from catboost import CatBoostClassifier
from fastapi import FastAPI
from fastapi.testclient import TestClient

from bl_ranking import api
from bl_ranking.payout_transport import HistoricalBrandPayout, ResilientPayoutRegressor, StubPayoutRegressor
from bl_ranking.predict_service import ServingArtifacts

TFM_COLUMNS = [
    "campaign_id", "page", "city", "device_type", "sub1", "sub2", "sub3",
    "business_type", "industry", "loan_reason", "cellphone_prefix",
    "country_state", "credit_score_num", "loan_amount_num", "monthly_revenue_num",
    "time_in_business_num", "session_day", "session_day_of_week",
    "session_hour", "from_start_to_register", "ratio_loan_amount_revenue",
    "gender", "fname_len", "lname_len", "client_name",
]


def _tiny_catboost() -> CatBoostClassifier:
    x = pd.DataFrame(
        {c: (["a", "b"] if c in {"page", "city", "device_type", "sub1", "sub2", "sub3", "business_type",
                                   "industry", "loan_reason", "cellphone_prefix", "country_state",
                                   "session_day_of_week", "gender", "client_name"} else [1, 2])
         for c in TFM_COLUMNS}
    )
    y = [0, 1]
    cat_features = x.select_dtypes(include=["object"]).columns.tolist()
    model = CatBoostClassifier(iterations=3, depth=2, verbose=False, allow_writing_files=False)
    model.fit(x, y, cat_features=cat_features)
    return model


@pytest.fixture
def fake_artifacts(tmp_path) -> ServingArtifacts:
    all_clients = pd.DataFrame({"client_name": ["brandA", "brandB", "other"]})
    all_clients.to_csv(tmp_path / "all_clients.csv", index=False)
    return ServingArtifacts(
        cb_model=_tiny_catboost(),
        payout_predictor=StubPayoutRegressor(),
        tfm_columns=TFM_COLUMNS,
        predictors_path=str(tmp_path) + "/",
        all_clients=all_clients,
        catboost_version="test",
        tabpfn_transport_mode="stub",
        model_version="test-v1",
        loaded_at=dt.datetime.now(),
    )


@pytest.fixture
def client(monkeypatch, fake_artifacts):
    monkeypatch.setattr(api, "load_artifacts", lambda: fake_artifacts)
    with TestClient(api.app) as c:
        yield c


class _AlwaysFailsPrimary:
    def predict(self, X):
        raise RuntimeError("simulated TabPFN outage")


@pytest.fixture
def fallback_artifacts(tmp_path) -> ServingArtifacts:
    # A real ResilientPayoutRegressor wrapping a primary that always fails,
    # wired through the same construction path load_artifacts() uses in
    # "live"/"live_no_cache" mode — the fake_artifacts fixture above always
    # uses a bare StubPayoutRegressor, which meant the fallback path (and
    # the fallback_used response field) had zero coverage through the real
    # HTTP endpoint. This fixture closes that gap.
    all_clients = pd.DataFrame({"client_name": ["brandA", "brandB", "other"]})
    all_clients.to_csv(tmp_path / "all_clients.csv", index=False)
    fallback = HistoricalBrandPayout(mean_payout_by_brand={"brandA": 11.0, "brandB": 22.0}, global_mean_payout=5.0)
    payout_predictor = ResilientPayoutRegressor(
        primary=_AlwaysFailsPrimary(),
        fallbacks=[("historical", fallback)],
        timeout_s=1.0,
        model_version="test-v1",
        primary_name="live",
    )
    return ServingArtifacts(
        cb_model=_tiny_catboost(),
        payout_predictor=payout_predictor,
        tfm_columns=TFM_COLUMNS,
        predictors_path=str(tmp_path) + "/",
        all_clients=all_clients,
        catboost_version="test",
        tabpfn_transport_mode="live",
        model_version="test-v1",
        loaded_at=dt.datetime.now(),
    )


@pytest.fixture
def fallback_client(monkeypatch, fallback_artifacts):
    monkeypatch.setattr(api, "load_artifacts", lambda: fallback_artifacts)
    with TestClient(api.app) as c:
        yield c


VALID_PAYLOAD = {
    "session_dt": "2026-01-06T19:24:22",
    "conversion_dt": "2026-01-06T19:26:10",
    "register_date": "2026-01-06T19:26:07",
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


def test_health_returns_200(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_ready_returns_200_when_artifacts_loaded(client):
    r = client.get("/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["ready"] is True
    assert body["model_version"] == "test-v1"


def test_predict_returns_ranked_brands_with_model_version(client):
    r = client.post("/predict", json=VALID_PAYLOAD)
    assert r.status_code == 200
    body = r.json()
    assert body["model_version"] == "test-v1"
    assert isinstance(body["fallback_used"], bool)
    assert isinstance(body["latency_ms"], float)
    # Not just "if present, well-formed" -- a regression that made rankings
    # degenerate to {} (e.g. the all_clients cross-join or the
    # expected_payout > 0.01 filter breaking) must fail this test, not pass
    # it vacuously.
    assert len(body["rankings"]) >= 1
    for brand, ranking in body["rankings"].items():
        assert brand in {"brandA", "brandB"}
        assert ranking["rank"] >= 1
        assert ranking["expected_payout"] > 0


def test_predict_reports_fallback_used_through_real_endpoint(fallback_client):
    r = fallback_client.post("/predict", json=VALID_PAYLOAD)
    assert r.status_code == 200
    body = r.json()
    assert body["fallback_used"] is True
    # The historical-mean fallback values from fallback_artifacts, not a
    # live prediction -- confirms the fallback path actually ran, not just
    # that the flag was set independently of the returned numbers.
    assert body["rankings"]["brandB"]["expected_payout"] > body["rankings"]["brandA"]["expected_payout"]




def test_predict_rejects_missing_required_field(client):
    payload = dict(VALID_PAYLOAD)
    del payload["register_date"]
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_predict_rejects_wrong_type(client):
    payload = dict(VALID_PAYLOAD)
    payload["campaign_id"] = "not-a-number"
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_predict_rejects_tz_aware_timestamp(client):
    payload = dict(VALID_PAYLOAD)
    payload["session_dt"] = "2026-01-06T19:24:22+05:00"
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_predict_rejects_overlong_string_field(client):
    payload = dict(VALID_PAYLOAD)
    payload["fname"] = "x" * 10_000
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_predict_rejects_oversized_body(client):
    payload = dict(VALID_PAYLOAD, page="x" * 200_000)
    r = client.post("/predict", json=payload)
    # Pydantic field max_length=256 catches oversized string values.
    assert r.status_code == 422


def test_ready_returns_503_before_artifacts_loaded(monkeypatch, fake_artifacts):
    def _boom():
        raise RuntimeError("simulated startup failure with a secret-looking detail: sk-abc123")

    monkeypatch.setattr(api, "load_artifacts", _boom)
    with TestClient(api.app) as c:
        r = c.get("/ready")
        assert r.status_code == 503
        body = r.json()
        assert body["ready"] is False
        # /ready is unauthenticated -- the raw exception message must never
        # reach the client, only a generic exception-type label.
        assert body["detail"] == "RuntimeError"
        assert "secret-looking" not in (body["detail"] or "")


def test_api_key_is_checked_before_the_body_is_parsed(monkeypatch, fake_artifacts):
    """Auth must gate before Pydantic. As a route dependency it does not: an
    unauthenticated caller sending a malformed body gets 422, which burns CPU
    and enumerates the schema through the error's field names."""
    from bl_ranking.config import Settings

    monkeypatch.setattr(api, "load_artifacts", lambda: fake_artifacts)
    monkeypatch.setattr(api, "get_settings", lambda: Settings(api_key="secret-key"))

    app = FastAPI(lifespan=api.lifespan)
    app.add_middleware(api.ApiKeyMiddleware, api_key="secret-key")
    app.include_router(api.app.router)

    with TestClient(app) as client:
        assert client.post("/predict", json={}).status_code == 401
        assert client.post("/predict", json=VALID_PAYLOAD).status_code == 401
        assert client.post("/predict", json={}, headers={"X-API-Key": "wrong"}).status_code == 401
        assert client.post("/predict", json=VALID_PAYLOAD, headers={"X-API-Key": "secret-key"}).status_code == 200
        assert client.get("/health").status_code == 200  # probes stay open
