# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run manifest emitted by a curation step and read back by an auditor.

A manifest reconstructed after the fact from a step's own output is
self-consistent and proves nothing. Only the producer knows what it read, what
it meant to remove, and whether it reached the end. This module owns that
record so the producing step and the auditing step cannot drift apart.

The canonical JSON form is fixed here because ``config_hash`` must be
reproducible across machines and Python versions: sorted keys, no whitespace,
and no ASCII escaping.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

CANONICAL_JSON = "sorted-keys, separators=(',',':'), ensure_ascii=false"

#: Emitted when the producing stack cannot attribute a row disappearance to a
#: deliberate filter. Curator's writers carry file paths but not row counts
#: (``BaseWriter.process`` logs ``task.num_items`` and drops it), and no stage
#: reports how many rows it removed, so a wrapper cannot honestly declare a
#: per-filter breakdown. An auditor must treat this as "unknown", never as zero.
ATTRIBUTION_UNAVAILABLE = "unavailable"
ATTRIBUTION_DECLARED = "declared"


def utc_now() -> str:
    """Timestamp in RFC 3339 UTC, seconds resolution."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def canonical_json(obj: Any) -> str:
    """Serialize deterministically. The form is part of the manifest contract."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def json_safe(obj: Any) -> Any:
    """Replace non-finite floats with null, recursively.

    ``json.dumps`` writes bare ``NaN`` and ``Infinity``, which Python reads back
    and every strict parser rejects: they are not in RFC 8259, so a report
    containing one cannot be read by ``JSON.parse``, Go, or serde. A statistic
    over documents that all failed to score is genuinely absent, and ``null``
    says that in a form everything can read.

    Raising instead — ``allow_nan=False`` — would end a run over a figure the
    report is already equipped to describe as missing.
    """
    if isinstance(obj, float):
        return obj if obj == obj and abs(obj) != float("inf") else None
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def config_hash(config: dict[str, Any]) -> str:
    """Content hash of a resolved config, prefixed with its algorithm."""
    digest = hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def tool_revision() -> str:
    """Best available identifier for the code that produced a run.

    Prefers an explicitly injected revision so a container build can stamp its
    own commit; falls back to the installed package version.
    """
    injected = os.environ.get("NEMOTRON_TOOL_REVISION")
    if injected:
        return injected
    try:
        from importlib.metadata import version

        return f"nemotron {version('nemotron')}"
    except Exception:  # noqa: BLE001 - version lookup must never fail a run
        return "unknown"


def count_jsonl(paths: Iterable[str | Path], source_field: str | None = None) -> dict[str, Any]:
    """Count records in JSONL files, optionally tallying them per source.

    Delegates to :func:`integrity.scan_shard` so that "a row" means exactly one
    thing across the producer and the auditor. Two counters with two slightly
    different ideas of what counts — one treating an unparsable line as a row,
    the other not — would put a manifest and its audit permanently at odds and
    report the disagreement as a mismatch in the data.

    A truncated file still contributes its surviving rows rather than raising.
    """
    from nemotron.steps.curate.runtime import integrity

    file_count = 0
    row_count = 0
    per_source: Counter[str] = Counter()
    unparsable = 0

    for path in paths:
        p = Path(path)
        if not p.is_file():
            continue
        file_count += 1
        report = integrity.scan_shard(p, source_field=source_field, want_digest=False)
        row_count += report.row_count
        unparsable += report.bad_record_count
        per_source.update(report.per_source)

    counted: dict[str, Any] = {"file_count": file_count, "row_count": row_count}
    if source_field is not None:
        counted["per_source"] = dict(sorted(per_source.items()))
    if unparsable:
        counted["unparsable_rows"] = unparsable
    return counted


def build_manifest(
    *,
    step_id: str,
    config: dict[str, Any],
    started_at: str,
    input_glob: str,
    input_counts: dict[str, Any],
    output_counts: dict[str, Any],
    id_field: str | None = None,
    source_field: str | None = None,
    completed_at: str | None = None,
    duplicate_ids: str = "reject",
    declared: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a manifest.

    ``completed_at`` is passed only on the success path. Its absence is the
    signal that a run died before its write barrier, and an auditor is expected
    to treat that as a hard finding rather than infer completion from the files
    that happen to exist.
    """
    if declared is None:
        # The caller supplied no per-gate breakdown, so the honest record is a
        # measured delta labelled as measured. A caller that CAN attribute — one
        # holding Curator's per-stage counters — passes `declared` instead.
        delta = input_counts.get("row_count", 0) - output_counts.get("row_count", 0)
        declared = {
            "attribution": ATTRIBUTION_UNAVAILABLE,
            "rows_absent_from_output": delta,
            "filtered": None,
            "failed": None,
            "quarantined": None,
        }

    producer: dict[str, Any] = {
        "step_id": step_id,
        "tool_revision": tool_revision(),
        "config_hash": config_hash(config),
        "started_at": started_at,
    }
    if completed_at is not None:
        producer["completed_at"] = completed_at

    output: dict[str, Any] = dict(output_counts)
    if id_field is not None:
        output["id_field"] = id_field
    if source_field is not None:
        output["source_field"] = source_field

    return {
        "schema_version": SCHEMA_VERSION,
        "producer": producer,
        "input": {"glob": input_glob, **input_counts},
        "output": output,
        "declared": declared,
        "canonicalization": {"json": CANONICAL_JSON, "duplicate_ids": duplicate_ids},
    }


def validate_manifest(manifest: Any) -> list[str]:
    """Return a list of contract violations; empty means conformant.

    Shared by the producer, which validates before writing, and the auditor,
    which validates before trusting. Returning problems rather than raising lets
    an auditor report every fault in one pass.
    """
    problems: list[str] = []

    if not isinstance(manifest, dict):
        return [f"manifest must be a mapping, got {type(manifest).__name__}"]

    version = manifest.get("schema_version")
    if version != SCHEMA_VERSION:
        problems.append(f"schema_version must be {SCHEMA_VERSION}, got {version!r}")

    producer = manifest.get("producer")
    if not isinstance(producer, dict):
        problems.append("producer block is missing or not a mapping")
    else:
        for key in ("step_id", "tool_revision", "config_hash", "started_at"):
            if not producer.get(key):
                problems.append(f"producer.{key} is required")
        digest = producer.get("config_hash")
        if isinstance(digest, str) and not digest.startswith("sha256:"):
            problems.append("producer.config_hash must carry its algorithm prefix")

    for block in ("input", "output"):
        section = manifest.get(block)
        if not isinstance(section, dict):
            problems.append(f"{block} block is missing or not a mapping")
            continue
        for key in ("file_count", "row_count"):
            value = section.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                problems.append(f"{block}.{key} must be a non-negative integer, got {value!r}")

    declared = manifest.get("declared")
    if not isinstance(declared, dict):
        problems.append("declared block is missing or not a mapping")
    elif declared.get("attribution") not in (ATTRIBUTION_DECLARED, ATTRIBUTION_UNAVAILABLE):
        problems.append(
            f"declared.attribution must be {ATTRIBUTION_DECLARED!r} or "
            f"{ATTRIBUTION_UNAVAILABLE!r}, got {declared.get('attribution')!r}"
        )

    canon = manifest.get("canonicalization")
    if not isinstance(canon, dict):
        problems.append("canonicalization block is missing or not a mapping")
    else:
        if canon.get("json") != CANONICAL_JSON:
            problems.append("canonicalization.json does not match this schema's canonical form")
        if canon.get("duplicate_ids") not in ("reject", "multiset"):
            problems.append("canonicalization.duplicate_ids must be 'reject' or 'multiset'")

    return problems


def is_complete(manifest: dict[str, Any]) -> bool:
    """Whether the producer reached its write barrier."""
    producer = manifest.get("producer")
    return isinstance(producer, dict) and bool(producer.get("completed_at"))


def write_manifest(path: str | Path, manifest: dict[str, Any]) -> Path:
    """Validate, then write atomically so a reader never sees a partial file."""
    problems = validate_manifest(manifest)
    if problems:
        raise ValueError("refusing to write a non-conformant manifest: " + "; ".join(problems))

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(f".{destination.name}.tmp")
    payload = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(destination)
    return destination


def read_manifest(path: str | Path) -> dict[str, Any]:
    """Read a manifest without validating it, so an auditor can report on a bad one."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
