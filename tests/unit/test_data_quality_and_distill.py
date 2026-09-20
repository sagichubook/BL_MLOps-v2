"""Input gating, drift notes, and the distillation fidelity maths.

The quality gate is what stops a bad export becoming next week's champion,
and the ranking-agreement maths is what justifies serving a surrogate at all
— so both need to be right for the reasons stated, not just to run.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bl_ranking.data_quality import check_training_input, compare_to_previous
from bl_ranking.distill import _kendall_tau, _spearman, build_serving_grid, evaluate_ranking_agreement

# --------------------------------------------------------------------------
# Quality gate
# --------------------------------------------------------------------------


def _write_csv(tmp_path, n_rows=20_000, days=60, registered_share=0.5, drop_column=None):
    rng = np.random.default_rng(0)
    start = pd.Timestamp("2025-12-11")
    session_dt = start + pd.to_timedelta(rng.integers(0, days, n_rows), unit="D")
    registered = rng.random(n_rows) < registered_share
    frame = pd.DataFrame(
        {
            "session_id": np.arange(n_rows),
            "session_dt": session_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "register_date": np.where(registered, session_dt.strftime("%Y-%m-%d %H:%M:%S"), None),
            "client_name": rng.choice(["a", "b", "c"], n_rows),
            "payout": np.where(rng.random(n_rows) < 0.3, rng.uniform(1, 300, n_rows), np.nan),
            "disposition": "Lead",
            "disposition_source": "a",
            "credit_score": "Good (650-719)",
            "industry": "construction",
            "loan_amount": "$25,000 - $49,999",
            "monthly_revenue": "$20,000 - $49,999",
            "time_in_business": "2+ years",
            "business_type": "LLC",
        }
    )
    if drop_column:
        frame = frame.drop(columns=[drop_column])
    path = tmp_path / "input.csv"
    frame.to_csv(path, index=False)
    return str(path)


def test_healthy_input_passes_and_reports_metrics(tmp_path):
    report = check_training_input(_write_csv(tmp_path))
    assert report.ok
    assert report.metrics["dq_n_rows"] == 20_000
    assert report.metrics["dq_n_brands"] == 3
    assert 0.45 < report.metrics["dq_registered_share"] < 0.55


def test_truncated_export_is_refused(tmp_path):
    report = check_training_input(_write_csv(tmp_path, n_rows=500))
    assert not report.ok
    assert any("truncated" in e for e in report.errors)


def test_missing_required_column_is_refused_before_reading_the_file(tmp_path):
    report = check_training_input(_write_csv(tmp_path, drop_column="payout"))
    assert not report.ok
    assert "payout" in report.errors[0]


def test_too_short_a_window_is_refused(tmp_path):
    report = check_training_input(_write_csv(tmp_path, days=5))
    assert not report.ok
    assert any("covers" in e for e in report.errors)


def test_mostly_unregistered_input_is_refused(tmp_path):
    """The trainable population is the registered rows; if almost none are
    registered the run would silently train on a fraction of the data."""
    report = check_training_input(_write_csv(tmp_path, registered_share=0.02))
    assert not report.ok
    assert any("register_date" in e for e in report.errors)


def test_drift_notes_fire_only_on_large_moves():
    previous = {"dq_n_rows": 100_000, "dq_n_brands": 11, "dq_payout_mean_paid": 80.0}
    assert compare_to_previous({"dq_n_rows": 105_000, "dq_n_brands": 11, "dq_payout_mean_paid": 82.0}, previous) == []

    notes = compare_to_previous({"dq_n_rows": 40_000, "dq_n_brands": 11, "dq_payout_mean_paid": 80.0}, previous)
    assert len(notes) == 1 and "dq_n_rows" in notes[0]


def test_drift_comparison_survives_a_missing_previous_run():
    assert compare_to_previous({"dq_n_rows": 10}, {}) == []


# --------------------------------------------------------------------------
# Distillation maths
# --------------------------------------------------------------------------


def test_serving_grid_is_users_crossed_with_every_brand_except_other():
    rows = pd.DataFrame(
        {
            "client_name": ["x", "y", "x"],
            "campaign_id": [1, 1, 2],
            "page": ["p", "p", "q"],
        }
    )
    grid = build_serving_grid(rows, ["x", "y", "other"], ["campaign_id", "page", "client_name"], n_users=10)

    # 'other' is excluded by the vendored predictor, so it must not be taught.
    assert set(grid["client_name"]) == {"x", "y"}
    # Two distinct users (campaign/page pairs) x 2 brands.
    assert grid["_user_id"].nunique() == 2
    assert len(grid) == 4


def test_kendall_tau_is_1_for_identical_and_minus_1_for_reversed():
    order = ["a", "b", "c", "d"]
    assert _kendall_tau(order, order) == pytest.approx(1.0)
    assert _kendall_tau(order, list(reversed(order))) == pytest.approx(-1.0)


def test_spearman_handles_a_constant_series():
    assert np.isnan(_spearman(np.array([1.0, 1.0, 1.0]), np.array([1.0, 2.0, 3.0])))


class _ConstantClassifier:
    """P(lead) = 0.5 for every row, so expected payout is proportional to
    payout and the ranking comparison isolates the surrogate's error."""

    def predict_proba(self, X):
        return np.column_stack([np.full(len(X), 0.5), np.full(len(X), 0.5)])


def _grid(n_users=3, brands=("a", "b", "c")):
    rows = []
    for u in range(n_users):
        for b in brands:
            rows.append({"_user_id": u, "client_name": b, "campaign_id": u})
    return pd.DataFrame(rows)


def test_perfect_surrogate_scores_perfect_agreement():
    grid = _grid()
    teacher = np.array([30.0, 20.0, 10.0] * 3)
    metrics = evaluate_ranking_agreement(
        grid, ["campaign_id", "client_name"], teacher, teacher.copy(), _ConstantClassifier()
    )
    assert metrics["rank_top1_agreement"] == 1.0
    assert metrics["rank_kendall_tau_mean"] == pytest.approx(1.0)
    assert metrics["rank_top1_expected_payout_loss_pct"] == pytest.approx(0.0)


def test_reversed_surrogate_is_penalised_on_the_metric_that_matters():
    """Payout error alone can look modest while the order is inverted — the
    top-1 and revenue-loss metrics are what catch it."""
    grid = _grid()
    teacher = np.array([30.0, 20.0, 10.0] * 3)
    surrogate = np.array([10.0, 20.0, 30.0] * 3)
    metrics = evaluate_ranking_agreement(
        grid, ["campaign_id", "client_name"], teacher, surrogate, _ConstantClassifier()
    )
    assert metrics["rank_top1_agreement"] == 0.0
    assert metrics["rank_kendall_tau_mean"] == pytest.approx(-1.0)
    # Promoting the teacher's worst brand gives up 2/3 of the expected payout.
    assert metrics["rank_top1_expected_payout_loss_pct"] == pytest.approx(66.667, abs=0.01)


# --------------------------------------------------------------------------
# Ranking proxy metrics
# --------------------------------------------------------------------------


class _FixedPayout:
    """Payout is whatever the brand's name encodes, so the expected order is
    known up front and the metric arithmetic can be checked against it."""

    def __init__(self, by_brand: dict[str, float]) -> None:
        self._by_brand = by_brand

    def predict(self, X):
        return np.array([self._by_brand[b] for b in X["client_name"]])


def _ranking_fixtures(buyer_brand: str):
    """One session, three brands, a known sale."""
    columns = ["campaign_id", "client_name"]
    x_test = pd.DataFrame({"campaign_id": [1, 1, 1], "client_name": ["a", "b", "c"]})
    y_test = pd.DataFrame({"sold_to_client": [0, 0, 0], "payout": [0.0, 0.0, 0.0]})
    idx = ["a", "b", "c"].index(buyer_brand)
    y_test.loc[idx, "sold_to_client"] = 1
    y_test.loc[idx, "payout"] = 100.0
    return x_test, y_test, columns


def test_ranking_proxy_scores_a_correctly_ranked_buyer_at_one():
    from bl_ranking.ranking_eval import evaluate_ranking

    x_test, y_test, columns = _ranking_fixtures(buyer_brand="a")
    metrics = evaluate_ranking(
        x_test=x_test, y_test=y_test, brands=["a", "b", "c"], columns=columns,
        cb_model=_ConstantClassifier(),
        payout_predictor=_FixedPayout({"a": 30.0, "b": 20.0, "c": 10.0}),
    )
    assert metrics["rankproxy_recall_at_1"] == 1.0
    assert metrics["rankproxy_mrr"] == 1.0
    assert metrics["rankproxy_payout_captured_at_1_pct"] == 100.0
    assert metrics["rankproxy_distinct_buying_brands"] == 1


def test_ranking_proxy_reports_the_buyers_real_position():
    from bl_ranking.ranking_eval import evaluate_ranking

    x_test, y_test, columns = _ranking_fixtures(buyer_brand="c")
    metrics = evaluate_ranking(
        x_test=x_test, y_test=y_test, brands=["a", "b", "c"], columns=columns,
        cb_model=_ConstantClassifier(),
        payout_predictor=_FixedPayout({"a": 30.0, "b": 20.0, "c": 10.0}),
    )
    # The buyer was ranked last of three.
    assert metrics["rankproxy_recall_at_1"] == 0.0
    assert metrics["rankproxy_mean_rank_of_buyer"] == 3.0
    assert metrics["rankproxy_mrr"] == pytest.approx(1 / 3, abs=1e-4)
    assert metrics["rankproxy_payout_captured_at_1_pct"] == 0.0
    # Baselines must accompany the recalls, or they cannot be read.
    assert metrics["rankproxy_random_baseline_recall_at_1"] == pytest.approx(1 / 3, abs=1e-4)
    assert metrics["rankproxy_random_baseline_recall_at_3"] == 1.0


def test_ranking_proxy_ignores_the_other_bucket():
    """'other' is excluded by the vendored predictor, so it must never be
    ranked or counted as a candidate here either."""
    from bl_ranking.ranking_eval import evaluate_ranking

    x_test, y_test, columns = _ranking_fixtures(buyer_brand="a")
    metrics = evaluate_ranking(
        x_test=x_test, y_test=y_test, brands=["a", "b", "c", "other"], columns=columns,
        cb_model=_ConstantClassifier(),
        payout_predictor=_FixedPayout({"a": 30.0, "b": 20.0, "c": 10.0}),
    )
    assert metrics["rankproxy_n_brands_ranked"] == 3


# --------------------------------------------------------------------------
# Walk-forward pooling
# --------------------------------------------------------------------------


def test_wald_ci_brackets_the_estimate_and_narrows_with_n():
    from bl_ranking.ranking_eval import wald_ci

    lo, hi = wald_ci(89, 1000)
    assert lo < 0.089 < hi
    wide = hi - lo
    lo2, hi2 = wald_ci(890, 10000)  # same rate, 10x the data
    assert (hi2 - lo2) < wide / 2


def test_wald_ci_is_degenerate_at_zero_samples():
    from bl_ranking.ranking_eval import wald_ci

    lo, hi = wald_ci(0, 0)
    assert np.isnan(lo) and np.isnan(hi)


def test_pooling_uses_integer_hits_not_averaged_rates():
    """Folds cover unequal weeks. Averaging their recall rates would weight a
    quiet week the same as a busy one; pooling the raw hits does not."""
    from bl_ranking.train_pipeline import _pool_folds

    folds = [
        {"rankproxy_sales_evaluated": 100, "rankproxy_hits_at_1": 50, "rankproxy_hits_at_3": 90},
        {"rankproxy_sales_evaluated": 900, "rankproxy_hits_at_1": 90, "rankproxy_hits_at_3": 500},
    ]
    pooled = _pool_folds(folds, n_brands=10)

    assert pooled["rankproxy_pooled_sales"] == 1000
    # 140/1000, not the 0.30 that averaging 0.50 and 0.10 would give.
    assert pooled["rankproxy_pooled_recall_at_1"] == pytest.approx(0.14)
    assert pooled["rankproxy_pooled_folds"] == 2


def test_pooling_reports_whether_chance_is_excluded():
    from bl_ranking.train_pipeline import _pool_folds

    # 0.10 rate on 5,000 sales: the interval straddles the 1/10 baseline.
    inconclusive = _pool_folds(
        [{"rankproxy_sales_evaluated": 5000, "rankproxy_hits_at_1": 500, "rankproxy_hits_at_3": 1500}], n_brands=10
    )
    assert inconclusive["rankproxy_pooled_recall_at_1_beats_chance"] == 0

    # 0.30 on the same volume is far outside it.
    decisive = _pool_folds(
        [{"rankproxy_sales_evaluated": 5000, "rankproxy_hits_at_1": 1500, "rankproxy_hits_at_3": 3000}], n_brands=10
    )
    assert decisive["rankproxy_pooled_recall_at_1_beats_chance"] == 1


def test_pooling_no_sales_returns_nothing_rather_than_dividing_by_zero():
    from bl_ranking.train_pipeline import _pool_folds

    assert _pool_folds([{"rankproxy_sales_evaluated": 0}], n_brands=10) == {}
