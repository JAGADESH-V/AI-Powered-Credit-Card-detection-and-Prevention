"""Tamper-evident audit trail.

Every consequential action (a score being issued, an analyst overturning a
decision, a model being reloaded) is appended as a record whose hash covers
the previous record's hash. Deleting or editing any past entry breaks the
chain, and `verify_chain` will point at the exact record where it broke.

This is what makes the system defensible in a dispute or an audit: you can
prove the decision log has not been rewritten after the fact.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any

GENESIS = "0" * 64


@dataclass(frozen=True, slots=True)
class AuditRecord:
    sequence: int
    timestamp: str
    actor: str
    action: str
    resource: str
    metadata: dict[str, Any] = field(default_factory=dict)
    previous_hash: str = GENESIS
    record_hash: str = ""

    def compute_hash(self) -> str:
        body = {
            "sequence": self.sequence,
            "timestamp": self.timestamp,
            "actor": self.actor,
            "action": self.action,
            "resource": self.resource,
            "metadata": self.metadata,
            "previous_hash": self.previous_hash,
        }
        canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()


class AuditLog:
    """Append-only JSONL log. Swap the storage for an append-only S3 bucket
    with object lock, or a WORM table, in production."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self._lock = Lock()

    def _tail(self) -> tuple[int, str]:
        last_seq, last_hash = -1, GENESIS
        for record in self.read_all():
            last_seq, last_hash = record.sequence, record.record_hash
        return last_seq, last_hash

    def append(
        self,
        *,
        actor: str,
        action: str,
        resource: str,
        metadata: dict[str, Any] | None = None,
    ) -> AuditRecord:
        with self._lock:
            last_seq, last_hash = self._tail()
            draft = AuditRecord(
                sequence=last_seq + 1,
                timestamp=datetime.now(UTC).isoformat(),
                actor=actor,
                action=action,
                resource=resource,
                metadata=metadata or {},
                previous_hash=last_hash,
            )
            record = AuditRecord(**{**asdict(draft), "record_hash": draft.compute_hash()})
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(asdict(record), separators=(",", ":")) + "\n")
            return record

    def read_all(self) -> Iterator[AuditRecord]:
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield AuditRecord(**json.loads(line))

    def verify_chain(self) -> tuple[bool, str | None]:
        """Returns (is_intact, description_of_first_break)."""
        expected_previous = GENESIS
        for expected_sequence, record in enumerate(self.read_all()):
            if record.sequence != expected_sequence:
                return False, f"sequence gap at record {record.sequence}"
            if record.previous_hash != expected_previous:
                return False, f"broken link at record {record.sequence}"
            stored = record.record_hash
            if stored != AuditRecord(**{**asdict(record), "record_hash": ""}).compute_hash():
                return False, f"record {record.sequence} was modified"
            expected_previous = stored
        return True, None
