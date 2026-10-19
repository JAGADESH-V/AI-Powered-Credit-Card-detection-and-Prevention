"""Structured logging with redaction applied at the handler level.

Redaction lives here rather than at each call site, because the call site you
forget is exactly the one that leaks. Every record passes through the filter.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from contextvars import ContextVar
from datetime import UTC, datetime

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

_PAN_RE = re.compile(r"\b(?:\d[ -]*?){13,19}\b")
_SENSITIVE_KEYS = {
    "pan", "password", "secret", "token", "authorization", "api_key",
    "x-api-key", "jwt", "card_number", "cvv", "cvc",
}
_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
    "message", "asctime", "taskName",
}


def _redact(value: object) -> object:
    if isinstance(value, str):
        return _PAN_RE.sub(lambda m: "*" * 12 + re.sub(r"\D", "", m.group())[-4:], value)
    if isinstance(value, dict):
        return {
            k: ("[redacted]" if k.lower() in _SENSITIVE_KEYS else _redact(v))
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact(v) for v in value]
    return value


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": _redact(record.getMessage()),
            "request_id": request_id_var.get(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED:
                payload[key] = _redact(value)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())
    # Uvicorn's access log duplicates our own and can echo query strings.
    logging.getLogger("uvicorn.access").disabled = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
