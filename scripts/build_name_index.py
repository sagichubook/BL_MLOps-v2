#!/usr/bin/env python3
"""Build the compact first-name gender index used by training and serving.

Run once per `names-dataset` version bump. The artifact is deterministic, is
verified against every name in the source dataset before it is written, and
is logged to MLflow by each production training run so the exact index a
model version was served with can be recovered.

    python scripts/build_name_index.py [--out artifacts/name_gender_index.json.gz]
"""
import argparse
import logging
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bl_ranking.name_index import INDEX_FILENAME, build_index  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None, help=f"output path (default: assets/{INDEX_FILENAME})")
    parser.add_argument("--no-verify", action="store_true", help="skip the exhaustive fidelity check (not recommended)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    out = Path(args.out) if args.out else Path("assets") / INDEX_FILENAME
    t0 = time.monotonic()
    index = build_index(verify=not args.no_verify)
    index.save(out)
    elapsed = time.monotonic() - t0
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    print(f"names with a gender answer: {index.n_names:,}")
    print(f"names-dataset version: {index.source_version}")
    print(f"build took {elapsed:.1f}s, peak RSS {peak_mb:.0f} MB (offline only)")


if __name__ == "__main__":
    main()
