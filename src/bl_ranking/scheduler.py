"""Weekly production training on a fixed wall-clock schedule.

Sunday 05:00 UTC. Computes the next occurrence, runs the job in-process,
logs the outcome, and sleeps until the next slot. Databricks handles
scheduling in production; this runs locally via Docker Compose.
"""
from __future__ import annotations

import argparse
import logging
import time
from datetime import UTC, datetime, timedelta

from bl_ranking.logging_utils import configure_logging

logger = logging.getLogger("bl_ranking.scheduler")

SUNDAY = 6
DEFAULT_HOUR = 5
DEFAULT_MINUTE = 0


def next_run_at(
    after: datetime,
    weekday: int = SUNDAY,
    hour: int = DEFAULT_HOUR,
    minute: int = DEFAULT_MINUTE,
) -> datetime:
    """First occurrence strictly after ``after`` (timezone-aware required)."""
    if after.tzinfo is None:
        raise ValueError("next_run_at() requires a timezone-aware datetime")

    candidate = after.replace(hour=hour, minute=minute, second=0, microsecond=0)
    days_ahead = (weekday - candidate.weekday()) % 7
    candidate += timedelta(days=days_ahead)
    if candidate <= after:
        candidate += timedelta(days=7)
    return candidate


def run_once() -> str | None:
    """Run the production training job, returning the registered version."""
    from bl_ranking.config import get_settings
    from bl_ranking.train_pipeline import run_training

    settings = get_settings()
    started = time.monotonic()
    try:
        version = run_training(
            mode="production",
            input_path=settings.data_path,
            input_file=settings.data_file,
            output_predictors_path=settings.artifacts_path,
        )
        logger.info(
            "scheduled_training_succeeded",
            extra={"model_version": version, "seconds": round(time.monotonic() - started, 1)},
        )
        return version
    except Exception:
        logger.exception("scheduled_training_failed", extra={"seconds": round(time.monotonic() - started, 1)})
        return None


def run_forever(weekday: int = SUNDAY, hour: int = DEFAULT_HOUR, minute: int = DEFAULT_MINUTE) -> None:
    """Sleep until each Sunday 05:00 occurrence, run, repeat."""
    while True:
        current = datetime.now(UTC)
        target = next_run_at(current, weekday, hour, minute)
        wait_s = max(0.0, (target - current).total_seconds())
        logger.info(
            "scheduler_waiting",
            extra={"next_run_utc": target.astimezone(UTC).isoformat(), "sleep_seconds": round(wait_s)},
        )
        time.sleep(wait_s)
        run_once()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run production training on a weekly schedule.")
    parser.add_argument("--hour", type=int, default=DEFAULT_HOUR)
    parser.add_argument("--minute", type=int, default=DEFAULT_MINUTE)
    parser.add_argument("--weekday", type=int, default=SUNDAY, help="0=Monday ... 6=Sunday")
    parser.add_argument("--now", action="store_true", help="run once immediately and exit")
    args = parser.parse_args()

    configure_logging()
    if args.now:
        run_once()
        return
    run_forever(weekday=args.weekday, hour=args.hour, minute=args.minute)


if __name__ == "__main__":
    main()
