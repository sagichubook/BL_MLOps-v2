"""The weekly schedule. Pins the instant itself, and that the Databricks
quartz expression and this scheduler name the same one."""
from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta

import pytest
import yaml

from bl_ranking import REPO_ROOT
from bl_ranking.scheduler import DEFAULT_HOUR, DEFAULT_MINUTE, SUNDAY, next_run_at


def _utc(y, m, d, hh=0, mm=0):
    return datetime(y, m, d, hh, mm, tzinfo=UTC)


def test_next_run_is_the_coming_sunday_at_0500_utc():
    assert next_run_at(_utc(2026, 9, 16, 12, 0)) == _utc(2026, 9, 20, 5, 0)


def test_earlier_on_the_same_sunday_still_fires_that_day():
    assert next_run_at(_utc(2026, 9, 20, 4, 59)) == _utc(2026, 9, 20, 5, 0)


def test_exactly_on_the_slot_moves_to_next_week_rather_than_refiring():
    assert next_run_at(_utc(2026, 9, 20, 5, 0)) == _utc(2026, 9, 27, 5, 0)


def test_after_the_slot_moves_to_next_week():
    assert next_run_at(_utc(2026, 9, 20, 5, 1)) == _utc(2026, 9, 27, 5, 0)


def test_every_result_is_a_sunday_at_the_configured_time():
    start = _utc(2026, 1, 1)
    for offset_hours in range(0, 24 * 40, 7):
        result = next_run_at(start + timedelta(hours=offset_hours))
        assert result.weekday() == SUNDAY
        assert (result.hour, result.minute, result.second) == (DEFAULT_HOUR, DEFAULT_MINUTE, 0)


def test_naive_datetime_is_rejected_rather_than_assumed():
    with pytest.raises(ValueError, match="timezone-aware"):
        next_run_at(datetime(2026, 9, 16, 12, 0))


def test_loop_sleeps_until_the_slot_and_does_not_double_fire(monkeypatch):
    """Verify the scheduler computes the correct wait and doesn't re-fire
    the same slot. Uses monkeypatch on time.sleep and datetime.now."""
    slept: list[float] = []
    ran: list[int] = []
    clock = {"now": _utc(2026, 9, 16, 12, 0)}

    def fake_sleep(seconds):
        slept.append(seconds)
        clock["now"] = clock["now"] + timedelta(seconds=seconds)

    def fake_now(tz=UTC):
        return clock["now"]

    # Patch run_forever's internals: time.sleep, datetime.now, and run_once.
    # We call run_forever in a loop ourselves to control iteration count.
    monkeypatch.setattr("bl_ranking.scheduler.time.sleep", fake_sleep)
    monkeypatch.setattr("bl_ranking.scheduler.datetime.now", fake_now)
    monkeypatch.setattr("bl_ranking.scheduler.run_once", lambda: ran.append(1))

    from bl_ranking.scheduler import run_forever

    # run_forever loops forever, so we need to limit it.
    # Override the while condition by raising after 2 runs.
    call_count = {"n": 0}

    def original_run_once():
        ran.append(1)
        call_count["n"] += 1

    def guarded_run_once():
        original_run_once()
        if call_count["n"] >= 2:
            raise KeyboardInterrupt

    monkeypatch.setattr("bl_ranking.scheduler.run_once", guarded_run_once)

    with pytest.raises(KeyboardInterrupt):
        run_forever()

    assert len(ran) == 2
    assert slept[0] == pytest.approx((3 * 24 + 17) * 3600)
    assert slept[1] == pytest.approx(7 * 24 * 3600)


def test_a_failing_job_does_not_kill_the_scheduler(monkeypatch, caplog):
    """One bad week must not silently end the weekly schedule."""
    def boom():
        raise RuntimeError("training exploded")

    monkeypatch.setattr("bl_ranking.train_pipeline.run_training", lambda **_: boom())
    from bl_ranking.scheduler import run_once

    with caplog.at_level("ERROR"):
        assert run_once() is None
    assert any("scheduled_training_failed" in r.message for r in caplog.records)


# --------------------------------------------------------------------------
# The two expressions of the schedule must agree
# --------------------------------------------------------------------------


def test_databricks_job_says_sunday_0500_in_utc():
    spec = yaml.safe_load((REPO_ROOT / "databricks" / "resources" / "bl_ranking_job.yml").read_text())
    schedule = spec["resources"]["jobs"]["bl_ranking_weekly_train"]["schedule"]
    assert schedule["timezone_id"] == "UTC"
    seconds, minutes, hours, _dom, _month, dow = schedule["quartz_cron_expression"].split()
    assert (seconds, minutes, hours, dow) == ("0", "0", "5", "SUN")


def test_compose_scheduler_service_runs_the_same_entry_point():
    compose = yaml.safe_load((REPO_ROOT / "docker-compose.yml").read_text())
    scheduler = compose["services"]["scheduler"]
    invocation = " ".join(scheduler.get("entrypoint") or []) + " " + " ".join(scheduler.get("command") or [])
    assert "bl_ranking.scheduler" in invocation
    assert scheduler["environment"]["TZ"] == "UTC"


def test_readme_documents_the_same_schedule():
    text = (REPO_ROOT / "README.md").read_text()
    assert re.search(r"Sunday\s+05:00\s+UTC", text), "README must state the schedule it actually runs"
