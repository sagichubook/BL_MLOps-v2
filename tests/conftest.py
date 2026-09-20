"""Shared test fixtures. Synthetic data only — no real customer data is
ever committed or required to run the test suite."""
from __future__ import annotations

import os
import random
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

@pytest.fixture(autouse=True)
def _fresh_settings():
    """get_settings() is lru_cached for the hot path; tests monkeypatch the
    environment, so the cache must not leak between them."""
    from bl_ranking.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


CLIENT_NAMES = ["fundera / nerdwallet", "businessloans.com", "xlt", "lendingtree"]
CREDIT_SCORES = ["Very Poor - Under 550", "Fair (600-649)", "Good (650-719)", "Excellent (720+)"]
LOAN_AMOUNTS = ["$10,000 - $24,999", "$25,000 - $49,999", "$50,000 - $74,999", "$100,000 - $199,999"]
MONTHLY_REVENUE = ["$0 - $9,999", "$20,000 - $49,999", "$50,000 - $99,999", "$100,000 - $199,999"]
TIME_IN_BUSINESS = ["less than 6 months", "6-12 months", "1-2 years", "2+ years"]
BUSINESS_TYPES = ["C Corporation", "S Corporation", "LLC", "Sole Proprietorship"]
INDUSTRIES = ["construction", "retail_trade", "healthcare", "other"]
LOAN_REASONS = ["Equipment purchase", "Business expansion", "Working capital", "Debt refinance"]
PAGES = ["top10us.com/app/business-loans-v2", "top10us.com/app/business-loans-v6"]
DEVICE_TYPES = ["mobile", "desktop"]
FIRST_NAMES = ["John", "Maria", "David", "Sarah", "Michael", "Linda", "Robert", "Patricia"]
LAST_NAMES = ["Smith", "Garcia", "Johnson", "Martinez", "Brown", "Davis"]
CITIES = [("Fort Lauderdale", "Florida"), ("Austin", "Texas"), ("Columbus", "Ohio")]


def _make_synthetic_rows(n: int, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    base_dt = datetime(2026, 1, 1)
    for _i in range(n):
        session_dt = base_dt + timedelta(days=rng.randint(0, 59), hours=rng.randint(0, 23), minutes=rng.randint(0, 59))
        register_date = session_dt + timedelta(minutes=rng.randint(1, 5))
        city, state = rng.choice(CITIES)
        is_lead = rng.random() < 0.35
        client_name = rng.choice(CLIENT_NAMES) if is_lead else None
        row = {
            "session_id": str(uuid.uuid4()),
            "session_dt": session_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "conversion_dt": (session_dt + timedelta(minutes=2)).strftime("%Y-%m-%d %H:%M:%S") if is_lead else "",
            "register_date": register_date.strftime("%Y-%m-%d %H:%M:%S"),
            "campaign_id": 120000000000000000 + rng.randint(0, 999999),
            "page": rng.choice(PAGES),
            "auto_city": city,
            "auto_country": "United States",
            "auto_state": state,
            "device_type": rng.choice(DEVICE_TYPES),
            "sub1": rng.randint(1000000, 9999999),
            "sub2": f"{rng.randint(1000000, 9999999)} Ad set",
            "sub3": rng.randint(1000000000, 9999999999),
            "business_type": rng.choice(BUSINESS_TYPES),
            "credit_score": rng.choice(CREDIT_SCORES),
            "industry": rng.choice(INDUSTRIES),
            "loan_amount": rng.choice(LOAN_AMOUNTS),
            "loan_reason": rng.choice(LOAN_REASONS),
            "monthly_revenue": rng.choice(MONTHLY_REVENUE),
            "time_in_business": rng.choice(TIME_IN_BUSINESS),
            "fname": rng.choice(FIRST_NAMES),
            "lname": rng.choice(LAST_NAMES),
            "cellphone": rng.randint(2000000000, 9999999999),
            "client_name": client_name,
            "payout": round(rng.uniform(10, 150), 2) if is_lead else None,
            "disposition": "Lead" if is_lead else None,
            "disposition_source": client_name if is_lead else None,
        }
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_bl_csv(tmp_path: Path) -> tuple[str, str]:
    """Writes a small synthetic CSV with bl_full_data.csv's schema and
    returns (input_path, input_file) the way BLPayoutModelsFit expects."""
    df = _make_synthetic_rows(300)
    input_path = str(tmp_path) + "/"
    input_file = "synthetic_bl_data.csv"
    df.to_csv(Path(input_path) / input_file, index=False)
    return input_path, input_file


@pytest.fixture
def example_user_data() -> dict:
    """Matches the hard-coded example in the vendored predictor's __main__."""
    return {
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


def has_tabpfn_token() -> bool:
    return bool(os.environ.get("TABPFN_TOKEN"))


requires_tabpfn_token = pytest.mark.skipif(
    not has_tabpfn_token(),
    reason="requires a real TABPFN_TOKEN — the vendored fit_() calls the hosted tabpfn-client "
    "API directly and cannot be stubbed without modifying vendored code",
)
