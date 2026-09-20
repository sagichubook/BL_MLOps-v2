"""Load simulation against /predict. See scripts/run_loadtests.sh.

Run a single-user arm as well as a concurrent one: with a slow transport a
concurrent run's percentiles measure queueing as much as the service, and the
single-user arm is what answers "how long is the user waiting".

Set API_KEY to exercise the authenticated path.
"""
from __future__ import annotations

import os
import random

from locust import HttpUser, between, events, task

API_KEY = os.environ.get("API_KEY", "")

CREDIT_SCORES = ["Very Poor - Under 550", "Fair (600-649)", "Good (650-719)", "Excellent (720+)"]
LOAN_AMOUNTS = ["$10,000 - $24,999", "$25,000 - $49,999", "$50,000 - $74,999", "$100,000 - $199,999"]
MONTHLY_REVENUE = ["$0 - $9,999", "$20,000 - $49,999", "$50,000 - $99,999", "$100,000 - $199,999"]
FIRST_NAMES = ["Rigoberto", "Maria", "David", "Sarah", "Michael", "Linda"]


def _random_payload() -> dict:
    return {
        "session_dt": "2026-01-06T19:24:22",
        "conversion_dt": "2026-01-06T19:26:10",
        "register_date": "2026-01-06T19:26:07",
        "campaign_id": 120227360861540306,
        "page": "top10us.com/app/business-loans-v2",
        "auto_city": "Fort Lauderdale",
        "auto_country": "United States",
        "auto_state": "Florida",
        "device_type": random.choice(["mobile", "desktop"]),
        "sub1": "1121993",
        "sub2": "01121993 Ad set",
        "sub3": "1513124082",
        "business_type": "C Corporation",
        "credit_score": random.choice(CREDIT_SCORES),
        "industry": "construction",
        "loan_amount": random.choice(LOAN_AMOUNTS),
        "loan_reason": "Equipment purchase",
        "monthly_revenue": random.choice(MONTHLY_REVENUE),
        "time_in_business": "2+ years",
        "fname": random.choice(FIRST_NAMES),
        "lname": "Rodriguez",
        "cellphone": random.randint(2000000000, 9999999999),
    }


class RankingUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task
    def predict(self):
        headers = {"X-API-Key": API_KEY} if API_KEY else {}
        with self.client.post(
            "/predict", json=_random_payload(), headers=headers, catch_response=True
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"status {resp.status_code}: {resp.text[:200]}")
                return
            body = resp.json()
            # A 200 from a fallback tier is a degraded success: not a
            # failure, but a run must not report its latency as healthy.
            tier = body.get("payout_tier", "unknown")
            _TIERS[tier] = _TIERS.get(tier, 0) + 1
            if body.get("fallback_used"):
                resp.request_meta["context"] = {"payout_tier": tier}


_TIERS: dict[str, int] = {}


@events.quitting.add_listener
def _report_tiers(environment, **_kwargs):
    total = sum(_TIERS.values())
    if not total:
        return
    print("\npayout tier distribution across the run:")
    for tier, count in sorted(_TIERS.items(), key=lambda kv: -kv[1]):
        print(f"  {tier:<14} {count:>6}  ({100 * count / total:5.1f}%)")
    degraded = total - _TIERS.get("live", 0) - _TIERS.get("surrogate", 0) - _TIERS.get("stub", 0)
    if degraded:
        print(f"  -> {degraded} request(s) were served by a fallback tier, not the primary transport.")
