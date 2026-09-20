"""Input checks that gate every training job.

A weekly job that retrains on whatever CSV it finds will happily ship a model
built on a truncated export or a window with a hole in it. None of those
raise; all change what gets served. Thresholds are floors -- a breach means
"the export is wrong", not "the data drifted a little".
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("bl_ranking.data_quality")

REQUIRED_COLUMNS = [
    "session_id", "session_dt", "register_date", "client_name", "payout",
    "disposition", "disposition_source", "credit_score", "industry",
    "loan_amount", "monthly_revenue", "time_in_business", "business_type",
]


@dataclass
class DataQualityReport:
    metrics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def check_training_input(
    csv_path: str,
    min_rows: int = 10_000,
    min_days_covered: int = 21,
    max_missing_day_gap: int = 2,
    min_registered_share: float = 0.10,
) -> DataQualityReport:
    """Validate the training CSV. Errors block the run; warnings are logged."""
    report = DataQualityReport()

    header = pd.read_csv(csv_path, nrows=0)
    missing = [c for c in REQUIRED_COLUMNS if c not in header.columns]
    if missing:
        report.errors.append(f"input is missing required columns: {missing}")
        return report

    usecols = ["session_dt", "register_date", "client_name", "payout", "disposition"]
    frame = pd.read_csv(csv_path, usecols=usecols, low_memory=False)
    session_dt = pd.to_datetime(frame["session_dt"], errors="coerce")

    n_rows = len(frame)
    registered = frame["register_date"].notna().sum()
    days = session_dt.dt.normalize().dropna()
    n_days = int(days.nunique())
    span_days = int((days.max() - days.min()).days) + 1 if n_days else 0
    payout = pd.to_numeric(frame["payout"], errors="coerce").fillna(0.0)
    paid = payout[payout > 0]

    report.metrics = {
        "dq_n_rows": n_rows,
        "dq_n_days_present": n_days,
        "dq_span_days": span_days,
        "dq_registered_share": round(float(registered / n_rows), 4) if n_rows else 0.0,
        "dq_session_dt_unparseable": int(session_dt.isna().sum()),
        "dq_n_brands": int(frame["client_name"].dropna().nunique()),
        "dq_paid_rows": int((payout > 0).sum()),
        "dq_payout_mean_paid": round(float(paid.mean()), 4) if len(paid) else 0.0,
        "dq_payout_p99": round(float(np.percentile(paid, 99)), 2) if len(paid) else 0.0,
        "dq_payout_max": round(float(payout.max()), 2),
    }

    if n_rows < min_rows:
        report.errors.append(f"only {n_rows} rows (< {min_rows}); refusing to train on a probably-truncated export")
    if n_days and span_days < min_days_covered:
        report.errors.append(f"input covers {span_days} days (< {min_days_covered})")
    if n_rows and registered / n_rows < min_registered_share:
        report.errors.append(
            f"only {registered / n_rows:.1%} of rows have register_date (< {min_registered_share:.0%}); "
            "the trainable population would be far smaller than expected"
        )

    # A hole in the middle of the window silently reweights the time split.
    if n_days > 1:
        gaps = days.drop_duplicates().sort_values().diff().dt.days.dropna()
        worst = int(gaps.max()) if len(gaps) else 1
        report.metrics["dq_max_day_gap"] = worst
        if worst > max_missing_day_gap:
            report.warnings.append(f"largest gap between consecutive days is {worst} days")

    if session_dt.isna().sum():
        report.warnings.append(f"{int(session_dt.isna().sum())} rows have an unparseable session_dt")

    for line in report.warnings:
        logger.warning("data_quality_warning", extra={"detail": line})
    for line in report.errors:
        logger.error("data_quality_error", extra={"detail": line})
    return report


def compare_to_previous(current: dict[str, Any], previous: dict[str, Any], tolerance: float = 0.35) -> list[str]:
    """Flag run-over-run shifts big enough to mean 'the input changed', not
    'the world moved slightly'. Returns human-readable drift notes."""
    notes: list[str] = []
    watch = ["dq_n_rows", "dq_registered_share", "dq_n_brands", "dq_paid_rows", "dq_payout_mean_paid"]
    for key in watch:
        now, before = current.get(key), previous.get(key)
        if now is None or before is None or not isinstance(now, (int, float)) or not before:
            continue
        change = (now - before) / abs(before)
        if abs(change) > tolerance:
            notes.append(f"{key} moved {change:+.1%} vs the previous production run ({before} -> {now})")
    for line in notes:
        logger.warning("data_drift_warning", extra={"detail": line})
    return notes
