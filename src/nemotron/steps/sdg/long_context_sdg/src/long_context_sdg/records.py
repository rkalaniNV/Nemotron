"""Canonical record parsing and atomic final-artifact writing."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from .schemas import CanonicalRecord


def load_records(path: Path) -> list[CanonicalRecord]:
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(CanonicalRecord.model_validate_json(line))
            except Exception as exc:
                raise ValueError(f"invalid canonical record at {path}:{line_number}: {exc}") from exc
    return records


def write_records(path: Path, records: Iterable[CanonicalRecord]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(record.model_dump_json() + "\n")
                count += 1
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return count
