"""Append-only checkpointing for independently retryable query candidates."""

from __future__ import annotations

import os
import threading
from pathlib import Path

from .schemas import QuerySynthesisRecord

_LOCK = threading.Lock()


def load_records(path: Path) -> list[QuerySynthesisRecord]:
    if not path.exists():
        return []
    records: list[QuerySynthesisRecord] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(QuerySynthesisRecord.model_validate_json(line))
            except Exception as exc:
                raise ValueError(f"invalid query synthesis checkpoint at {path}:{line_number}: {exc}") from exc
    return records


def verify_fingerprint(records: list[QuerySynthesisRecord], fingerprint: str) -> None:
    incompatible = sorted(
        {record.synthesis_fingerprint for record in records if record.synthesis_fingerprint != fingerprint}
    )
    if incompatible:
        raise ValueError("query checkpoint contains incompatible synthesis fingerprint(s): " + ", ".join(incompatible))


def append_record(path: Path, record: QuerySynthesisRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(record.model_dump_json() + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def latest_by_query(
    records: list[QuerySynthesisRecord],
) -> dict[str, QuerySynthesisRecord]:
    latest: dict[str, QuerySynthesisRecord] = {}
    for record in records:
        current = latest.get(record.query_id)
        if current is None or record.attempt >= current.attempt:
            latest[record.query_id] = record
    return latest
