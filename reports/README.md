# Measurements

Produced on this machine against the real 128,442-row `bl_full_data.csv` and the
real hosted TabPFN API. 2 vCPU, `WEB_CONCURRENCY=2`. Analysis in
[`../docs/part2.md`](../docs/part2.md).

- `payout_benchmark.json` — per-transport latency. `python scripts/benchmark_payout.py`
- `*_stats.csv` — end-to-end 45 s locust arms. `bash scripts/run_loadtests.sh`
