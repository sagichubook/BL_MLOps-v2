"""The payout ladder: cache, admission control, fallback order.

Each of these pins a failure that is invisible in production -- a timeout that
does not bound wall-clock time, per-request state shared between requests, a
ladder that skips its best rung."""
from __future__ import annotations

import threading
import time

import numpy as np
import pandas as pd
import pytest

from bl_ranking.payout_transport import (
    HistoricalBrandPayout,
    PayoutResultCache,
    PayoutUnavailable,
    ResilientPayoutRegressor,
    StubPayoutRegressor,
    TabPFNCachedRegressor,
)


def _frame(client_names: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"client_name": client_names, "credit_score_num": [601] * len(client_names)})


def _resilient(primary, fallbacks=None, **kwargs) -> ResilientPayoutRegressor:
    defaults = dict(
        timeout_s=1.0,
        model_version="test-v1",
        max_concurrency=4,
        admission_wait_s=0.2,
        cache=None,
        primary_name="primary",
    )
    defaults.update(kwargs)
    if fallbacks is None:
        fallbacks = [("historical", HistoricalBrandPayout(global_mean_payout=7.0))]
    return ResilientPayoutRegressor(primary=primary, fallbacks=fallbacks, **defaults)


class _Ok:
    def __init__(self, value: float = 99.0) -> None:
        self.value = value
        self.calls = 0

    def predict(self, X):
        self.calls += 1
        return [self.value] * len(X)


class _Flaky:
    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def predict(self, X):
        raise self._exc


class _Slow:
    def __init__(self, sleep_s: float = 2.0) -> None:
        self.sleep_s = sleep_s

    def predict(self, X):
        time.sleep(self.sleep_s)
        return [1.0] * len(X)


# --------------------------------------------------------------------------
# Stub
# --------------------------------------------------------------------------


def test_stub_regressor_is_deterministic_and_in_range():
    stub = StubPayoutRegressor()
    X = _frame(["brandA", "brandB"])
    assert list(stub.predict(X)) == list(stub.predict(X))
    assert all(1.0 <= v <= 300.0 for v in stub.predict(X))


def test_stub_regressor_varies_by_row():
    preds = StubPayoutRegressor().predict(_frame(["a", "b", "c"]))
    assert len(set(preds)) > 1


# --------------------------------------------------------------------------
# Historical fallback table
# --------------------------------------------------------------------------


def test_historical_payout_records_counts_and_coverage():
    x = pd.DataFrame({"client_name": ["a", "a", "b"]})
    y = pd.DataFrame({"payout": [10.0, 20.0, 100.0]})
    hbp = HistoricalBrandPayout.compute(x, y)

    assert hbp.payout_for("a") == pytest.approx(15.0)
    assert hbp.n_by_brand == {"a": 2, "b": 1}
    assert hbp.payout_for("unseen") == pytest.approx(hbp.global_mean_payout)

    coverage = hbp.coverage_of(["a", "b", "c"])
    assert coverage["n_covered"] == 2
    assert coverage["uncovered_brands"] == ["c"]
    # Both real brands have < 30 observations, so both must be flagged thin
    # rather than quietly presented as reliable means.
    assert coverage["thin_brands"] == ["a", "b"]


def test_historical_payout_is_itself_a_predictor():
    """It sits on the ladder, so it must satisfy the same interface."""
    hbp = HistoricalBrandPayout(mean_payout_by_brand={"a": 5.0}, global_mean_payout=1.0)
    assert list(hbp.predict(_frame(["a", "zzz"]))) == [5.0, 1.0]


def test_historical_payout_roundtrip(tmp_path):
    hbp = HistoricalBrandPayout({"a": 15.0}, {"a": 4}, 15.0, source="full_payout_training_set")
    path = tmp_path / "fallback.json"
    hbp.save(path)
    loaded = HistoricalBrandPayout.load(path)
    assert loaded.mean_payout_by_brand == {"a": 15.0}
    assert loaded.n_by_brand == {"a": 4}
    assert loaded.source == "full_payout_training_set"


# --------------------------------------------------------------------------
# Result cache
# --------------------------------------------------------------------------


def test_cache_returns_a_copy_so_callers_cannot_corrupt_it():
    cache = PayoutResultCache(maxsize=4, ttl_s=60)
    X = _frame(["a", "b"])
    key = cache.key_for(X)
    cache.put(key, np.array([1.0, 2.0]))

    got = cache.get(key)
    got[0] = 999.0
    assert list(cache.get(key)) == [1.0, 2.0]


def test_cache_expires_entries():
    cache = PayoutResultCache(maxsize=4, ttl_s=0.05)
    key = cache.key_for(_frame(["a"]))
    cache.put(key, np.array([1.0]))
    assert cache.get(key) is not None
    time.sleep(0.08)
    assert cache.get(key) is None


def test_cache_evicts_least_recently_used():
    cache = PayoutResultCache(maxsize=2, ttl_s=60)
    keys = [cache.key_for(_frame([name])) for name in ("a", "b", "c")]
    for k in keys[:2]:
        cache.put(k, np.array([1.0]))
    cache.get(keys[0])          # touch 'a' so 'b' becomes the LRU entry
    cache.put(keys[2], np.array([1.0]))

    assert cache.get(keys[0]) is not None
    assert cache.get(keys[1]) is None


def test_cache_can_be_disabled_with_size_zero():
    cache = PayoutResultCache(maxsize=0, ttl_s=60)
    key = cache.key_for(_frame(["a"]))
    cache.put(key, np.array([1.0]))
    assert cache.get(key) is None


def test_resilient_serves_from_cache_without_calling_primary():
    primary = _Ok(42.0)
    cache = PayoutResultCache(maxsize=8, ttl_s=60)
    resilient = _resilient(primary, cache=cache)
    X = _frame(["brandA", "brandB"])

    resilient.reset_call_state()
    first = resilient.predict(X)
    resilient.reset_call_state()
    second = resilient.predict(X)

    assert list(first) == list(second)
    assert primary.calls == 1, "second identical request must not reach the primary"
    assert resilient.last_outcome.tier == "cache"
    assert resilient.last_call_used_fallback is False


# --------------------------------------------------------------------------
# The ladder
# --------------------------------------------------------------------------


def test_primary_success_is_reported_as_primary():
    resilient = _resilient(_Ok(99.0))
    resilient.reset_call_state()
    assert list(resilient.predict(_frame(["brandA"]))) == [99.0]
    assert resilient.last_outcome.tier == "primary"
    assert resilient.last_call_used_fallback is False


def test_error_drops_to_the_next_rung():
    hbp = HistoricalBrandPayout(mean_payout_by_brand={"brandA": 42.0}, global_mean_payout=10.0)
    triggered: list[str] = []
    resilient = _resilient(_Flaky(RuntimeError("HTTP 429")), [("historical", hbp)], on_event=triggered.append)

    resilient.reset_call_state()
    assert list(resilient.predict(_frame(["brandA", "unseen"]))) == [42.0, 10.0]
    assert resilient.last_call_used_fallback is True
    assert resilient.last_outcome.tier == "historical"
    assert len(triggered) == 1


def test_ladder_prefers_the_better_fallback_and_falls_through_if_it_breaks():
    """The surrogate rung must be tried before the coarse historical table —
    and if it is broken, the request must still be answered, not dropped."""
    good = _Ok(55.0)
    hbp = HistoricalBrandPayout(global_mean_payout=1.0)

    ok_ladder = _resilient(_Flaky(RuntimeError("down")), [("surrogate", good), ("historical", hbp)])
    ok_ladder.reset_call_state()
    assert list(ok_ladder.predict(_frame(["brandA"]))) == [55.0]
    assert ok_ladder.last_outcome.tier == "surrogate"

    broken_ladder = _resilient(
        _Flaky(RuntimeError("down")), [("surrogate", _Flaky(ValueError("bad model"))), ("historical", hbp)]
    )
    broken_ladder.reset_call_state()
    assert list(broken_ladder.predict(_frame(["brandA"]))) == [1.0]
    assert broken_ladder.last_outcome.tier == "historical"


def test_every_tier_failing_raises_rather_than_returning_nonsense():
    resilient = _resilient(_Flaky(RuntimeError("down")), [("broken", _Flaky(RuntimeError("also down")))])
    resilient.reset_call_state()
    with pytest.raises(PayoutUnavailable):
        resilient.predict(_frame(["brandA"]))


def test_timeout_falls_back():
    resilient = _resilient(_Slow(2.0), timeout_s=0.2)
    resilient.reset_call_state()
    assert list(resilient.predict(_frame(["brandA"]))) == [7.0]
    assert resilient.last_call_used_fallback is True


def test_timeout_actually_bounds_wall_clock_time():
    """Regression: a per-call `with ThreadPoolExecutor(...)` would block in
    shutdown(wait=True) until the hung primary returned, so the timeout would
    change which exception fired but not how long the caller waited.

    The bound is expressed relative to the primary's duration rather than as a
    small absolute number: the property under test is "the caller was released
    long before the primary finished", and an absolute threshold turns CPU
    contention on a busy CI box into a spurious failure.
    """
    primary_seconds = 4.0
    resilient = _resilient(_Slow(primary_seconds), timeout_s=0.2)
    resilient.reset_call_state()
    start = time.monotonic()
    resilient.predict(_frame(["brandA"]))
    elapsed = time.monotonic() - start
    assert elapsed < primary_seconds / 2, (
        f"predict() took {elapsed:.2f}s against a 0.2s timeout and a "
        f"{primary_seconds}s primary — the timeout is not bounding wall-clock time"
    )


def test_overload_sheds_to_fallback_instead_of_queueing():
    """The core admission-control property.

    With every hosted slot busy, an arriving request must degrade promptly
    rather than queue behind a ~5 s call and burn its whole budget waiting.
    One slot, two concurrent callers: the second must be served by the
    fallback within roughly the admission wait, not the primary's duration.
    """
    primary_seconds = 4.0
    resilient = _resilient(_Slow(primary_seconds), timeout_s=10.0, max_concurrency=1, admission_wait_s=0.1)

    outcomes: dict[str, object] = {}

    def occupy():
        resilient.reset_call_state()
        resilient.predict(_frame(["brandA"]))

    def arrive_late():
        time.sleep(0.3)  # let the first caller take the only slot
        resilient.reset_call_state()
        started = time.monotonic()
        result = resilient.predict(_frame(["brandA"]))
        outcomes["elapsed"] = time.monotonic() - started
        outcomes["tier"] = resilient.last_outcome.tier
        outcomes["value"] = list(result)

    t1 = threading.Thread(target=occupy)
    t2 = threading.Thread(target=arrive_late)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert outcomes["tier"] == "historical"
    assert outcomes["value"] == [7.0]
    # Relative to the primary's duration, for the same reason as the timeout
    # test above: the property is "it did not wait for the busy primary".
    assert outcomes["elapsed"] < primary_seconds / 2, (
        f"shed request waited {outcomes['elapsed']:.2f}s against a {primary_seconds}s primary — "
        "it queued instead of degrading promptly"
    )
    assert resilient.counters["shed"] == 1


# --------------------------------------------------------------------------
# Per-request state isolation
# --------------------------------------------------------------------------


def test_reset_call_state_clears_a_stale_outcome():
    """The vendored predict_() can return early without ever calling
    predict(); without a reset, that request would report whatever the
    previous request on the same thread left behind."""
    resilient = _resilient(_Flaky(RuntimeError("boom")))
    resilient.reset_call_state()
    resilient.predict(_frame(["brandA"]))
    assert resilient.last_call_used_fallback is True

    resilient.reset_call_state()
    assert resilient.last_outcome is None
    assert resilient.last_call_used_fallback is False


def test_outcome_is_isolated_per_thread():
    """One shared singleton serves every concurrent request, so a plain
    instance attribute would let one request read another's outcome."""
    hbp = HistoricalBrandPayout(mean_payout_by_brand={"brandA": 5.0, "brandB": 6.0}, global_mean_payout=1.0)

    class _DataDependent:
        def predict(self, X):
            time.sleep(0.1)
            if (X["client_name"] == "brandA").all():
                return [99.0] * len(X)
            raise RuntimeError("boom")

    resilient = _resilient(_DataDependent(), [("historical", hbp)], timeout_s=5.0)
    observed: dict[str, bool] = {}

    def run_ok():
        resilient.reset_call_state()
        resilient.predict(_frame(["brandA"]))
        observed["ok"] = resilient.last_call_used_fallback

    def run_failing():
        resilient.reset_call_state()
        time.sleep(0.02)
        resilient.predict(_frame(["brandB"]))
        observed["failing"] = resilient.last_call_used_fallback

    t1 = threading.Thread(target=run_ok)
    t2 = threading.Thread(target=run_failing)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert observed["ok"] is False
    assert observed["failing"] is True


# --------------------------------------------------------------------------
# Evicted hosted fit detection
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exc,expected",
    [
        (type("FittedModelNotFoundError", (Exception,), {})("gone"), True),
        (RuntimeError("Fitted model not found for id abc"), True),
        (RuntimeError("HTTP 429 Too Many Requests"), False),
        (ValueError("column mismatch"), False),
    ],
)
def test_evicted_fit_is_recognised(exc, expected):
    """A refit must be attempted for an evicted fit and NOT for a rate limit —
    refitting on a 429 would add load to an API already shedding it."""
    assert TabPFNCachedRegressor._looks_like_missing_fit(exc) is expected


def test_stats_expose_tier_counters_and_cache_hit_rate():
    cache = PayoutResultCache(maxsize=8, ttl_s=60)
    resilient = _resilient(_Ok(1.0), cache=cache)
    X = _frame(["brandA"])
    resilient.reset_call_state()
    resilient.predict(X)
    resilient.reset_call_state()
    resilient.predict(X)

    stats = resilient.stats()
    assert stats["payout_calls"]["primary"] == 1
    assert stats["payout_calls"]["cache"] == 1
    assert stats["payout_cache"]["hits"] == 1


class _BlocksThenAnswers:
    """Blocks the first `block_count` calls on a gate; answers the rest."""

    def __init__(self, gate: threading.Event, block_count: int) -> None:
        self._gate = gate
        self._remaining = block_count
        self._lock = threading.Lock()
        self.started = 0

    def predict(self, X):
        with self._lock:
            block = self._remaining > 0
            self._remaining -= 1
            self.started += 1
        if block:
            self._gate.wait(30)
        return [42.0] * len(X)


def test_a_timed_out_straggler_does_not_block_the_next_admitted_request():
    """A call that exceeds its deadline releases its slot, but its thread runs
    on until the hosted side answers. With the pool sized to the semaphore,
    the next admitted request queues behind that straggler and its deadline —
    counted from submit — expires while it waits, which is the exact
    behaviour admission control exists to prevent."""
    gate = threading.Event()
    primary = _BlocksThenAnswers(gate, block_count=2)
    resilient = _resilient(primary, timeout_s=0.3, max_concurrency=2, admission_wait_s=2.0)

    try:
        # Two calls occupy both slots and hang; both time out and fall back,
        # leaving two threads still parked inside the primary.
        stragglers = [threading.Thread(target=lambda: (resilient.reset_call_state(), resilient.predict(_frame(["a"]))))
                      for _ in range(2)]
        for t in stragglers:
            t.start()
        for t in stragglers:
            t.join(timeout=5)

        assert resilient.counters["timeout"] == 2

        # A fresh request must now reach the primary, not queue behind them.
        resilient.reset_call_state()
        result = resilient.predict(_frame(["a"]))
        assert resilient.last_outcome.tier == "primary", (
            f"served by {resilient.last_outcome.tier}: the request queued behind a timed-out "
            "straggler instead of getting a worker"
        )
        assert list(result) == [42.0]
    finally:
        gate.set()
