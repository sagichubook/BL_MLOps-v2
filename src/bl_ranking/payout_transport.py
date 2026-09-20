"""Payout leg: transports, cache, admission control, fallback ladder.

The vendored predictor fits a TabPFNRegressor inside ``load_models()``, which
``predict_()`` calls per request: 13 s, ~12 of it the fit. Hosted latency is
also flat in batch size (4.83 s for 11 rows, 4.72 s for 3,000), so batching is
no lever and no synchronous hosted call reaches a page-load budget.

Hence a ladder -- surrogate (local, 2 ms), hosted (~3 s, exact), historical
brand means (0.1 ms, coarse). ``TABPFN_TRANSPORT`` picks the primary; the rest
are fallbacks. Numbers: ``reports/payout_benchmark.json``.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

logger = logging.getLogger("bl_ranking.payout_transport")

SURROGATE_FILENAME = "payout_surrogate.cbm"
SURROGATE_META_FILENAME = "payout_surrogate_meta.json"


class PayoutPredictor(Protocol):
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


class PayoutUnavailable(RuntimeError):
    """Raised by a transport when it cannot answer; triggers the next rung."""


# ==========================================================================
# Hosted TabPFN transports
# ==========================================================================


class TabPFNCachedRegressor:
    """Hosted TabPFN, fitted once. ``save_model()`` persists only the
    server-side fit id (741 bytes), so a process starts without refitting.

    That id is not permanent: the hosted side drops a fit when its training
    data ages out, and predicting against a dead id raises. Without
    ``refit_from_context()`` one eviction means falling back forever.
    """

    def __init__(self, access_token: str, context: dict | None = None) -> None:
        from tabpfn_client import TabPFNRegressor, set_access_token

        set_access_token(access_token)
        self._token = access_token
        # ignore_pretraining_limits=True matches the vendored script exactly
        # (bl_models_train.py:290) — same model configuration, caching on.
        self._model = TabPFNRegressor(ignore_pretraining_limits=True, fit_mode="fit_with_cache")
        self._fitted = False
        self._context = context
        self._refit_lock = threading.Lock()
        self._last_refit = 0.0

    # -- lifecycle --------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: pd.Series) -> TabPFNCachedRegressor:
        self._model.fit(X, y)
        self._fitted = True
        self._context = {"x": X, "y": y}
        return self

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._model.save_model(str(path))

    @classmethod
    def load(cls, path: Path, access_token: str, context: dict | None = None) -> TabPFNCachedRegressor:
        from tabpfn_client import TabPFNRegressor, set_access_token

        set_access_token(access_token)
        obj = cls.__new__(cls)
        obj._token = access_token
        obj._model = TabPFNRegressor.load_model(str(path))
        obj._fitted = True
        obj._context = context
        obj._refit_lock = threading.Lock()
        obj._last_refit = 0.0
        return obj

    @property
    def model_id(self) -> str | None:
        return getattr(self._model, "model_id_", None)

    # -- prediction -------------------------------------------------------
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise PayoutUnavailable("TabPFNCachedRegressor.predict() called before fit()/load()")
        try:
            return np.asarray(self._model.predict(X))
        except Exception as exc:
            if self._looks_like_missing_fit(exc) and self.refit_from_context():
                logger.info("tabpfn_refit_succeeded_retrying_predict")
                return np.asarray(self._model.predict(X))
            raise

    @staticmethod
    def _looks_like_missing_fit(exc: Exception) -> bool:
        """Match on name/message: the client's exception has moved module
        between releases, so importing the symbol is not safe."""
        name = type(exc).__name__
        if "FittedModelNotFound" in name or "ModelNotFound" in name:
            return True
        text = str(exc).lower()
        return "not found" in text and ("model" in text or "fit" in text)

    def refit_from_context(self, cooldown_s: float = 60.0) -> bool:
        """Single-flight refit from the shipped context. Cooldown-guarded so
        an outage cannot turn every in-flight request into its own refit."""
        if self._context is None:
            return False
        with self._refit_lock:
            if time.monotonic() - self._last_refit < cooldown_s:
                return False
            self._last_refit = time.monotonic()
            try:
                logger.warning("tabpfn_refitting_from_context", extra={"n_rows": len(self._context["x"])})
                self._model.fit(self._context["x"], self._context["y"])
                self._fitted = True
                return True
            except Exception:
                logger.exception("tabpfn_refit_failed")
                return False



# ==========================================================================
# Local surrogate (distilled from TabPFN)
# ==========================================================================


class SurrogatePayoutRegressor:
    """Local CatBoost regressor distilled from TabPFN (see ``distill``).

    The only component that changes what a request is answered by, so its
    measured fidelity travels with the model version in
    ``payout_surrogate_meta.json``.
    """

    def __init__(self, model: Any, columns: list[str], meta: dict | None = None) -> None:
        self._model = model
        self._columns = columns
        self.meta = meta or {}

    @classmethod
    def load(cls, model_path: Path, columns: list[str], meta_path: Path | None = None) -> SurrogatePayoutRegressor:
        from catboost import CatBoostRegressor

        model = CatBoostRegressor(allow_writing_files=False, thread_count=1)
        model.load_model(str(model_path), format="cbm")
        meta = {}
        if meta_path is not None and meta_path.exists():
            meta = json.loads(meta_path.read_text())
        return cls(model, columns, meta)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self._model.predict(X[self._columns]))


# ==========================================================================
# Deterministic offline stub (tests, CI, token-free runs)
# ==========================================================================


class StubPayoutRegressor:
    """Deterministic function of the row, in the ~$1-$300 range. Unlike a
    real TabPFN fit it is exactly reproducible, so tests can assert on it."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> StubPayoutRegressor:
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        hashed = pd.util.hash_pandas_object(X, index=False).values
        digests = np.array(
            [int.from_bytes(hashlib.sha256(row.tobytes()).digest()[:4], "big") for row in hashed]
        )
        return 1.0 + (digests % 29900) / 100.0


# ==========================================================================
# Last-rung fallback: historical brand means
# ==========================================================================


@dataclass
class HistoricalBrandPayout:
    """Mean realised payout per brand, over the **full** payout population.

    Computing it from the 1,000-row TabPFN context instead covers 4 of 10
    brands and collapses the rest onto one global mean, flattening the order
    to P(lead) alone -- precisely when the system is already degraded.
    ``n_by_brand`` ships alongside so thin estimates stay visible.
    """

    mean_payout_by_brand: dict[str, float] = field(default_factory=dict)
    n_by_brand: dict[str, int] = field(default_factory=dict)
    global_mean_payout: float = 0.0
    source: str = "unknown"

    @classmethod
    def compute(
        cls,
        x_train_payout: pd.DataFrame,
        y_train_payout: pd.DataFrame,
        source: str = "full_payout_training_set",
    ) -> HistoricalBrandPayout:
        merged = x_train_payout[["client_name"]].copy()
        merged["payout"] = np.asarray(y_train_payout["payout"])
        grouped = merged.groupby("client_name")["payout"]
        return cls(
            mean_payout_by_brand={str(k): float(v) for k, v in grouped.mean().items()},
            n_by_brand={str(k): int(v) for k, v in grouped.size().items()},
            global_mean_payout=float(merged["payout"].mean()),
            source=source,
        )

    def payout_for(self, brand: str) -> float:
        return self.mean_payout_by_brand.get(brand, self.global_mean_payout)

    def coverage_of(self, brands: list[str]) -> dict[str, Any]:
        known = [b for b in brands if b in self.mean_payout_by_brand]
        thin = [b for b in known if self.n_by_brand.get(b, 0) < 30]
        return {
            "n_brands": len(brands),
            "n_covered": len(known),
            "coverage_pct": round(100.0 * len(known) / max(len(brands), 1), 1),
            "thin_brands": sorted(thin),
            "uncovered_brands": sorted(set(brands) - set(known)),
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """PayoutPredictor-shaped, so it is just another rung of the ladder."""
        return X["client_name"].map(self.payout_for).to_numpy(dtype=float)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "mean_payout_by_brand": self.mean_payout_by_brand,
                    "n_by_brand": self.n_by_brand,
                    "global_mean_payout": self.global_mean_payout,
                    "source": self.source,
                },
                indent=2,
            )
        )

    @classmethod
    def load(cls, path: Path) -> HistoricalBrandPayout:
        data = json.loads(path.read_text())
        return cls(
            mean_payout_by_brand=data["mean_payout_by_brand"],
            n_by_brand=data.get("n_by_brand", {}),
            global_mean_payout=data["global_mean_payout"],
            source=data.get("source", "unknown"),
        )


# ==========================================================================
# Result cache
# ==========================================================================


class PayoutResultCache:
    """TTL + LRU cache over ``predict()`` results.

    The frame carries ``from_start_to_register``, ``session_hour`` and
    ``campaign_id``, so organic users rarely collide: this absorbs retries and
    duplicate submits, not load. Hit rate is on ``/ready`` rather than assumed.
    The TTL is for correctness -- an entry outliving its weekly model would
    rank with a retired model's numbers.
    """

    def __init__(self, maxsize: int = 4096, ttl_s: float = 900.0) -> None:
        self.maxsize = maxsize
        self.ttl_s = ttl_s
        self._data: OrderedDict[str, tuple[float, np.ndarray]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key_for(X: pd.DataFrame) -> str:
        return hashlib.sha256(
            pd.util.hash_pandas_object(X, index=False).values.tobytes() + str(list(X.columns)).encode()
        ).hexdigest()

    def get(self, key: str) -> np.ndarray | None:
        if self.maxsize <= 0:
            return None
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                self.misses += 1
                return None
            stored_at, value = entry
            if time.monotonic() - stored_at > self.ttl_s:
                del self._data[key]
                self.misses += 1
                return None
            self._data.move_to_end(key)
            self.hits += 1
            return value.copy()

    def put(self, key: str, value: np.ndarray) -> None:
        if self.maxsize <= 0:
            return
        with self._lock:
            self._data[key] = (time.monotonic(), value.copy())
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            total = self.hits + self.misses
            return {
                "entries": len(self._data),
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": round(self.hits / total, 4) if total else None,
            }


# ==========================================================================
# The orchestrator
# ==========================================================================


@dataclass
class PayoutCallOutcome:
    tier: str
    reason: str | None = None
    duration_ms: float = 0.0


class ResilientPayoutRegressor:
    """Drop-in for ``model_tfm`` in the vendored ``prediction_expected_payout()``.

    Cache, then primary under an admission slot and a deadline, then each
    fallback rung. Outcome is thread-local because one instance is shared by
    every concurrent request; ``reset_call_state()`` must run before each one,
    since the vendored ``predict_()`` can return early without calling
    ``predict()`` and would inherit the previous outcome on that thread.
    """

    def __init__(
        self,
        primary: PayoutPredictor,
        fallbacks: list[tuple[str, PayoutPredictor]],
        timeout_s: float,
        model_version: str,
        max_concurrency: int = 8,
        admission_wait_s: float = 0.25,
        cache: PayoutResultCache | None = None,
        on_event: Any | None = None,
        primary_name: str = "primary",
    ) -> None:
        self._primary = primary
        self._primary_name = primary_name
        self._fallbacks = fallbacks
        self._timeout_s = timeout_s
        self._model_version = model_version
        self._cache = cache
        self._on_event = on_event
        self._local = threading.local()

        # The semaphore is the real concurrency bound. The pool is wider on
        # purpose: a call that exceeds its deadline releases its slot but its
        # thread keeps running until the hosted side answers, so a pool sized
        # to the semaphore would let the next admitted request queue behind
        # that straggler -- and its deadline, measured from submit, would
        # expire while it waited. Threads parked on a socket are cheap; the
        # headroom is what keeps admission and execution the same thing.
        self._admission = threading.Semaphore(max_concurrency)
        self._admission_wait_s = admission_wait_s
        self._pool = ThreadPoolExecutor(
            max_workers=max_concurrency * 4, thread_name_prefix="payout-call"
        )

        self.counters: dict[str, int] = {"primary": 0, "cache": 0, "shed": 0, "timeout": 0, "error": 0, "fallback": 0}
        self._counter_lock = threading.Lock()

    # -- per-request state -------------------------------------------------
    def reset_call_state(self) -> None:
        self._local.outcome = None

    @property
    def last_outcome(self) -> PayoutCallOutcome | None:
        return getattr(self._local, "outcome", None)

    @property
    def last_call_used_fallback(self) -> bool:
        outcome = self.last_outcome
        return bool(outcome and outcome.tier not in (self._primary_name, "cache"))

    def _bump(self, name: str) -> None:
        with self._counter_lock:
            self.counters[name] = self.counters.get(name, 0) + 1

    def _record(self, tier: str, reason: str | None, started: float) -> None:
        self._local.outcome = PayoutCallOutcome(
            tier=tier, reason=reason, duration_ms=round((time.monotonic() - started) * 1000, 2)
        )

    # -- the ladder --------------------------------------------------------
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        started = time.monotonic()

        cache_key = None
        if self._cache is not None:
            cache_key = self._cache.key_for(X)
            cached = self._cache.get(cache_key)
            if cached is not None and len(cached) == len(X):
                self._bump("cache")
                self._record("cache", None, started)
                return cached

        reason: str | None = None
        admitted = self._admission.acquire(timeout=self._admission_wait_s)
        if not admitted:
            # Deliberately do NOT queue. See the module docstring.
            reason = f"no hosted-call slot within {self._admission_wait_s}s"
            self._bump("shed")
        else:
            try:
                result = self._call_primary(X)
                if self._cache is not None and cache_key is not None:
                    self._cache.put(cache_key, result)
                self._bump("primary")
                self._record(self._primary_name, None, started)
                return result
            except FutureTimeoutError:
                reason = f"primary exceeded {self._timeout_s}s"
                self._bump("timeout")
            except Exception as exc:  # noqa: BLE001 - any failure drops a rung
                reason = f"{type(exc).__name__}: {exc}"
                self._bump("error")
            finally:
                self._admission.release()

        logger.warning(
            "payout_primary_unavailable",
            extra={
                "reason": reason,
                "model_version": self._model_version,
                "n_rows": len(X),
                "primary": self._primary_name,
            },
        )
        if self._on_event is not None:
            self._on_event(reason or "unknown")

        for name, predictor in self._fallbacks:
            try:
                result = predictor.predict(X)
                self._bump("fallback")
                self._record(name, reason, started)
                return np.asarray(result, dtype=float)
            except Exception:
                logger.exception("payout_fallback_failed", extra={"tier": name})
                continue

        raise PayoutUnavailable(f"every payout tier failed; last reason: {reason}")

    def _call_primary(self, X: pd.DataFrame) -> np.ndarray:
        """Primary on the bounded pool under a wall-clock deadline.

        The pool outlives the call: ``ThreadPoolExecutor.__exit__`` runs
        ``shutdown(wait=True)``, so a per-call ``with`` block would wait for a
        hung task anyway and the timeout would be cosmetic.
        """
        future = self._pool.submit(self._primary.predict, X)
        return np.asarray(future.result(timeout=self._timeout_s), dtype=float)

    # -- introspection -----------------------------------------------------
    def stats(self) -> dict[str, Any]:
        with self._counter_lock:
            counters = dict(self.counters)
        out: dict[str, Any] = {"payout_calls": counters, "primary": self._primary_name}
        if self._cache is not None:
            out["payout_cache"] = self._cache.stats()
        return out


# ==========================================================================
# Serving-startup factory
# ==========================================================================


def build_live_predictor(
    access_token: str,
    cached_model_path: Path,
    context_path: Path,
    use_cache_mode: bool = True,
) -> PayoutPredictor:
    """Prefer a saved cache reference (no data upload); otherwise fit once."""
    import joblib

    context = joblib.load(context_path)

    if not use_cache_mode:
        from tabpfn_client import TabPFNRegressor, set_access_token

        set_access_token(access_token)
        model = TabPFNRegressor(ignore_pretraining_limits=True)
        model.fit(context["x"], context["y"])
        return model

    if cached_model_path.exists():
        try:
            predictor = TabPFNCachedRegressor.load(cached_model_path, access_token, context=context)
            logger.info("tabpfn_cache_reference_loaded", extra={"model_id": predictor.model_id})
            return predictor
        except Exception:
            logger.exception("failed to load cached TabPFN reference %s; refitting from context", cached_model_path)

    regressor = TabPFNCachedRegressor(access_token, context=context)
    regressor.fit(context["x"], context["y"])
    try:
        regressor.save(cached_model_path)
    except Exception:
        logger.exception("failed to persist TabPFN cache reference to %s (non-fatal)", cached_model_path)
    return regressor
