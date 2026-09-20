"""Structured JSON logging. Callers pass row counts, brand names, latencies
and model versions — never names, phone numbers or addresses."""
from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime

_RESERVED = set(logging.LogRecord(None, None, "", 0, "", (), None).__dict__) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED:
                payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    # The serving wrapper skips the vendored per-instance capture_warnings()
    # (deviation 4); this is the global equivalent.
    logging.captureWarnings(True)
