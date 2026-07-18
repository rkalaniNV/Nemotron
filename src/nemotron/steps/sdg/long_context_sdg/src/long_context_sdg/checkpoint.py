"""Append-only canonical checkpoint and resume index."""

from __future__ import annotations

import os
import threading
from collections.abc import Iterable
from pathlib import Path

from .schemas import CanonicalRecord

_LOCK = threading.Lock()


def load_records(path: Path) -> list[CanonicalRecord]:
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                records.append(CanonicalRecord.model_validate_json(line))
            except Exception as exc:
                raise ValueError(
                    f"invalid checkpoint record at {path}:{line_no}: {exc}"
                ) from exc
    return records


def verify_fingerprint(records: Iterable[CanonicalRecord], fingerprint: str) -> None:
    incompatible = sorted(
        {r.config_fingerprint for r in records if r.config_fingerprint != fingerprint}
    )
    if incompatible:
        raise ValueError(
            "checkpoint contains incompatible configuration fingerprint(s): "
            + ", ".join(incompatible)
        )


def completed_query_ids(
    records: Iterable[CanonicalRecord],
    *,
    retry_failed: bool,
    retry_quarantine: bool,
) -> set[str]:
    latest: dict[str, CanonicalRecord] = {}
    for record in records:
        latest[record.query_id] = record
    skip = set()
    for record in latest.values():
        retry = (record.status == "generation_failed" and retry_failed) or (
            record.status == "quarantine" and retry_quarantine
        )
        if not retry:
            skip.add(record.query_id)
    return skip


def append_record(path: Path, record: CanonicalRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = record.model_dump_json()
    with _LOCK:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())
