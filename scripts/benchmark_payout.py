#!/usr/bin/env python3
"""Measure every payout transport against the same real artifacts.

This exists because the central claim of this system -- that the payout leg
was productised the efficient way -- is a claim about latency, and a latency
claim without a measured alternative is an assertion. It times, on identical
inputs:

* ``original``      what the vendored predictor does today: construct a
                    TabPFNRegressor, ``fit()`` the context, then ``predict()``
                    -- once per request;
* ``live_no_cache`` hosted, fitted once per process (the fit amortised away,
                    no server-side cache);
* ``live``          hosted with ``fit_mode="fit_with_cache"``;
* ``surrogate``     the local distilled model;
* ``cache_hit``     a repeat request served from the in-process cache.

Run:  python scripts/benchmark_payout.py --iterations 5 --out reports/payout_benchmark.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402

from bl_ranking.config import get_settings  # noqa: E402
from bl_ranking.payout_transport import (  # noqa: E402
    SURROGATE_FILENAME,
    SURROGATE_META_FILENAME,
    StubPayoutRegressor,
    SurrogatePayoutRegressor,
    TabPFNCachedRegressor,
)


def _percentiles(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "n": len(ordered),
        "min_ms": round(ordered[0], 1),
        "p50_ms": round(statistics.median(ordered), 1),
        "p95_ms": round(ordered[min(int(len(ordered) * 0.95), len(ordered) - 1)], 1),
        "max_ms": round(ordered[-1], 1),
        "mean_ms": round(statistics.fmean(ordered), 1),
    }


def _time_calls(fn, X, iterations: int) -> dict[str, float]:
    samples = []
    for _ in range(iterations):
        t0 = time.monotonic()
        fn(X)
        samples.append((time.monotonic() - t0) * 1000)
    return _percentiles(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", default=None, help="artifacts directory (default: settings.artifacts_path)")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--out", default="reports/payout_benchmark.json")
    parser.add_argument("--skip-original", action="store_true", help="skip the fit-per-request baseline (slow)")
    args = parser.parse_args()

    settings = get_settings()
    artifacts = Path(args.artifacts or settings.artifacts_path)
    context = joblib.load(artifacts / "payout_tfm_context.joblib")
    columns = list(context["columns"])

    all_clients = pd.read_csv(artifacts / "all_clients.csv")
    brands = [b for b in all_clients["client_name"].dropna().astype(str) if b != "other"]

    # One user against every brand: exactly the frame a /predict call builds.
    one_user = context["x"].iloc[[0]][columns].drop(columns=["client_name"])
    X = one_user.merge(pd.DataFrame({"client_name": brands}), how="cross")[columns]
    print(f"benchmark frame: {X.shape[0]} rows (1 user x {len(brands)} brands), {len(columns)} columns\n")

    results: dict[str, dict] = {"frame_rows": len(X), "iterations": args.iterations}
    token = settings.tabpfn_token

    # --- what the vendored predictor does per request -------------------
    if token and not args.skip_original:
        def original_call(frame):
            from tabpfn_client import TabPFNRegressor, set_access_token

            set_access_token(token)
            model = TabPFNRegressor(ignore_pretraining_limits=True)
            model.fit(context["x"], context["y"])
            return model.predict(frame)

        print("timing 'original' (fit + predict per call) ...")
        results["original_fit_per_request"] = _time_calls(original_call, X, max(2, args.iterations // 2))
        print(f"  {results['original_fit_per_request']}\n")

    # --- hosted, fitted once per process --------------------------------
    if token:
        from tabpfn_client import TabPFNRegressor, set_access_token

        print("timing 'live_no_cache' ...")
        set_access_token(token)
        local_model = TabPFNRegressor(ignore_pretraining_limits=True)
        local_model.fit(context["x"], context["y"])

        def _local_predict(frame):
            return local_model.predict(frame)

        results["live_no_cache"] = _time_calls(_local_predict, X, args.iterations)
        print(f"  {results['live_no_cache']}\n")

        print("timing 'live' (fit_with_cache) ...")
        cached = TabPFNCachedRegressor(token, context=context)
        t0 = time.monotonic()
        cached.fit(context["x"], context["y"])
        results["live_fit_seconds"] = round(time.monotonic() - t0, 2)
        results["live"] = _time_calls(cached.predict, X, args.iterations)
        print(f"  fit once: {results['live_fit_seconds']}s, then {results['live']}\n")

    # --- local surrogate -------------------------------------------------
    surrogate_path = artifacts / SURROGATE_FILENAME
    if surrogate_path.exists():
        print("timing 'surrogate' (local) ...")
        surrogate = SurrogatePayoutRegressor.load(surrogate_path, columns, artifacts / SURROGATE_META_FILENAME)
        surrogate.predict(X)  # warm
        results["surrogate"] = _time_calls(surrogate.predict, X, max(args.iterations, 50))
        results["surrogate_fidelity"] = {
            k: v for k, v in surrogate.meta.items() if k.startswith(("rank_", "payout_"))
        }
        print(f"  {results['surrogate']}\n  fidelity: {results['surrogate_fidelity']}\n")

    print("timing 'stub' ...")
    results["stub"] = _time_calls(StubPayoutRegressor().predict, X, max(args.iterations, 50))
    print(f"  {results['stub']}\n")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))

    print("=" * 68)
    print(f"{'transport':<28}{'p50 ms':>12}{'p95 ms':>12}")
    print("-" * 68)
    for key in ("original_fit_per_request", "live_no_cache", "live", "surrogate", "stub"):
        if key in results:
            print(f"{key:<28}{results[key]['p50_ms']:>12.1f}{results[key]['p95_ms']:>12.1f}")
    print("=" * 68)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
