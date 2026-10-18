# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Local ingest: read each policy file, redact it, and cut it into source spans.

Each file ends in one of three states:

- ``safe``: redacted text and spans are stored. Pre-authored Policy Rule IR in YAML or
  JSON skips extraction and goes straight to the normalization queue.
- ``quarantined``: the privacy scan found a credential, or redaction left personal data
  behind. Nothing from the file is stored except finding kinds and line numbers.
- ``rejected``: the file cannot be used (unsupported format, unreadable, no text layer,
  malformed Policy Rule IR, or no policy text).

Ingest never calls a model or the network. Redaction always finishes before any later
stage can read a document.
"""

import dataclasses
import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from nemotron.steps.byob.runtime.benchmark_families.safety import documents, privacy
from nemotron.steps.byob.runtime.benchmark_families.safety.artifacts import write_atomic, write_json, write_jsonl
from nemotron.steps.byob.runtime.benchmark_families.safety.confidence import quote_in_span
from nemotron.steps.byob.runtime.benchmark_families.safety.config import _UniqueKeyLoader
from nemotron.steps.byob.runtime.benchmark_families.safety.documents import (
    STRUCTURED_FORMATS,
    DocumentError,
    Line,
    detect_format,
    extract,
)
from nemotron.steps.byob.runtime.benchmark_families.safety.ir import normalize_candidate
from nemotron.steps.byob.runtime.benchmark_families.safety.sources import SourceChangedError, SourceError
from nemotron.steps.byob.runtime.benchmark_families.safety.spans import build_document, link_glossary_terms

INGEST_VERSION = 3
MAX_REDACTED_NAME_CHARS = 120
EVIDENCE_SELF_AUTHORED = "self_authored"
EVIDENCE_DOCUMENT_NOT_FOUND = "document_not_found"
EVIDENCE_DOCUMENT_CHANGED = "document_changed"
EVIDENCE_SPAN_NOT_FOUND = "span_not_found"
EVIDENCE_QUOTE_FOUND = "quote_found"
EVIDENCE_QUOTE_NOT_FOUND = "quote_not_found"
SAFE = "safe"
QUARANTINED = "quarantined"
REJECTED = "rejected"
INVALID_POLICY_RULE_IR = "invalid_policy_rule_ir"
DUPLICATE_POLICY_ID = "duplicate_policy_id"
NO_POLICY_TEXT = "no_policy_text"
UNSUPPORTED_FORMAT = "unsupported_format"
INVALID_STRUCTURED_DOCUMENT = "invalid_structured_document"
INGEST_FAILED = "ingest_failed"
EXTRACTION_PENDING = "pending"
EXTRACTION_SKIPPED = "skipped"

CACHE_DIR = "ingest"
REDACTED_DIR = "redacted"
INVENTORY_FILE = "source_inventory.json"
INDEX_FILE = "source_index.json"
GLOSSARY_FILE = "glossary.json"
RULES_FILE = "preauthored_rules.jsonl"
QUARANTINE_FILE = "files.jsonl"
DOCUMENT_IDS_FILE = "document_ids.json"
# The ingest outputs that rule extraction reads. Prepare records their hashes so generate can refuse edited copies.
RULE_INPUT_FILES = (INDEX_FILE, GLOSSARY_FILE, RULES_FILE)


@dataclass
class IngestResult:
    files: list[dict[str, Any]]
    documents: list[dict[str, Any]] = field(default_factory=list)
    spans: list[dict[str, Any]] = field(default_factory=list)
    texts: dict[str, str] = field(default_factory=dict)
    glossary: dict[str, Any] = field(default_factory=dict)
    rules: list[dict[str, Any]] = field(default_factory=list)
    reused: int = 0

    def count(self, status: str) -> int:
        return sum(1 for row in self.files if row["status"] == status)

    @property
    def privacy_blocked(self) -> bool:
        """No usable document is left and at least one was held back for privacy."""
        return self.count(SAFE) == 0 and self.count(QUARANTINED) > 0

    def counts(self) -> dict[str, int]:
        return {
            "files": len(self.files),
            SAFE: self.count(SAFE),
            QUARANTINED: self.count(QUARANTINED),
            REJECTED: self.count(REJECTED),
            "duplicates": sum(1 for row in self.files if row["duplicate_of"]),
            "documents": len(self.documents),
            "spans": len(self.spans),
            "preauthored_rules": len(self.rules),
            "glossary_terms": len(self.glossary.get("terms", [])),
            "reused_from_cache": self.reused,
        }


def ingest_sources(
    input_dir: Path,
    sources: Mapping[str, str],
    cache_dir: Path,
    *,
    privacy_labels: Mapping[str, Sequence[str]] | None = None,
) -> IngestResult:
    """Process every source. ``sources`` is the scan result ``{relative path: sha256}``."""
    ids = document_ids(list(sources), _read_document_ids(cache_dir / DOCUMENT_IDS_FILE))
    cache = cache_dir / CACHE_DIR
    outcomes: dict[str, dict[str, Any]] = {}
    used: set[str] = set()
    reused = 0
    for relative in sorted(sources):
        key = _cache_key(relative, ids[relative], sources[relative], privacy_labels)
        used.add(f"{key}.json")
        outcome = _read_cache(cache / f"{key}.json", relative, sources[relative])
        if outcome is not None:
            reused += 1
        else:
            outcome = _process_source(
                input_dir,
                relative,
                sources[relative],
                ids[relative],
                privacy_labels=privacy_labels,
            )
            write_json(cache / f"{key}.json", _cache_entry(relative, sources[relative], outcome))
        outcomes[relative] = outcome
    if cache.is_dir():
        for stale in cache.glob("*.json"):
            if stale.name not in used:
                stale.unlink(missing_ok=True)
    return _assemble(outcomes, reused)


def _process_source(
    input_dir: Path,
    relative: str,
    sha256: str,
    document_id: str,
    *,
    privacy_labels: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Read and ingest one source. Oversized files are refused before their bytes are loaded."""
    limit = documents.MAX_SOURCE_BYTES
    try:
        size = (input_dir / relative).stat().st_size
    except OSError as exc:
        raise SourceError(f"cannot read {relative}: {exc}") from exc
    if size > limit:
        row = _row(relative, sha256, document_id)
        row["format"] = detect_format(relative)
        return _rejected(row, "file_too_large", f"files larger than {limit} bytes are not read")
    data = _read_source(input_dir, relative, sha256)
    try:
        return process_file(relative, data, document_id, privacy_labels=privacy_labels)
    except Exception as exc:
        # One damaged or unexpected file must not discard the rest of the directory.
        return _rejected(
            _row(relative, sha256, document_id), INGEST_FAILED, f"{type(exc).__name__} while reading this file"
        )


def process_file(
    relative: str,
    data: bytes,
    document_id: str,
    *,
    privacy_labels: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Ingest one file's bytes. The returned outcome holds only redacted text."""
    row = _row(relative, hashlib.sha256(data).hexdigest(), document_id)
    fmt = detect_format(relative)
    row["format"] = fmt
    if fmt is None:
        return _rejected(row, UNSUPPORTED_FORMAT, f"{PurePosixPath(relative).suffix or 'no extension'} is not read")
    try:
        extracted = extract(data, fmt)
    except DocumentError as exc:
        return _rejected(row, exc.reason, exc.detail)
    scan = privacy.assess([line.text for line in extracted.lines], additional_labels=privacy_labels)
    row["findings"] = [finding.to_dict() for finding in scan.findings]
    row["redactions"] = scan.redactions
    if not scan.safe:
        row.update(status=QUARANTINED, reason=scan.reason, detail=_privacy_detail(scan))
        return {"file": row}
    lines = tuple(dataclasses.replace(line, text=text) for line, text in zip(extracted.lines, scan.lines))
    text = "\n".join(line.text for line in lines)
    if fmt in STRUCTURED_FORMATS:
        try:
            parsed = _parse_structured(text, fmt)
        except ValueError as exc:
            return _rejected(row, INVALID_STRUCTURED_DOCUMENT, str(exc))
        rules = _rule_candidates(parsed)
        if rules is not None:
            return _preauthored(row, rules)
    try:
        structured = build_document(document_id, lines, markdown_headings=extracted.markdown_headings)
    except DocumentError as exc:
        return _rejected(row, exc.reason, exc.detail)
    if not structured.spans:
        return _rejected(row, NO_POLICY_TEXT, "the file has headings or markup but no policy sentences")
    row.update(status=SAFE, extraction=EXTRACTION_PENDING, span_count=len(structured.spans))
    return {
        "file": row,
        "document": _document(row, structured.sections, lines),
        "text": structured.text,
        "spans": structured.spans,
        "glossary": structured.glossary,
    }


def document_ids(paths: Mapping[str, Any] | list[str], previous: Mapping[str, str] | None = None) -> dict[str, str]:
    """Readable document IDs that do not change when another file is added.

    The stem is used while it is free (``policy.md`` is ``policy``). An ID already recorded for
    that path is kept, so adding ``policy.txt`` names the new file ``policy.txt`` and leaves
    ``policy.md`` as ``policy``. On a first run with both files, the first path in sorted order
    keeps the stem.
    """
    ordered = sorted(paths)
    previous = previous or {}
    ids: dict[str, str] = {}
    used: set[str] = set()
    for path in ordered:
        old = previous.get(path)
        if isinstance(old, str) and _usable_id(old) and old not in used:
            ids[path] = old
            used.add(old)
    for path in ordered:
        if path in ids:
            continue
        for candidate in _id_candidates(path):
            if candidate not in used:
                ids[path] = candidate
                used.add(candidate)
                break
        else:
            digest = hashlib.sha256(path.encode("utf-8")).hexdigest()[:8]
            label = f"{_id_candidates(path)[-1]}-{digest}"
            while label in used:
                label += "x"
            ids[path] = label
            used.add(label)
    return ids


def write_artifacts(run_dir: Path, result: IngestResult) -> None:
    """Write every ingest output. The caller writes the cache index and run record after this."""
    stage_cache = run_dir / "stage_cache"
    write_jsonl(run_dir / "quarantine" / QUARANTINE_FILE, result.files)
    write_json(
        stage_cache / INVENTORY_FILE,
        {"ingest_version": INGEST_VERSION, "files": result.files, "counts": result.counts()},
    )
    write_json(
        stage_cache / INDEX_FILE,
        {"ingest_version": INGEST_VERSION, "documents": result.documents, "spans": result.spans},
    )
    write_json(stage_cache / GLOSSARY_FILE, {"ingest_version": INGEST_VERSION, **result.glossary})
    write_jsonl(stage_cache / RULES_FILE, result.rules)
    write_json(
        stage_cache / DOCUMENT_IDS_FILE,
        {"ids": {row["path"]: row["document_id"] for row in result.files}},
    )
    redacted = stage_cache / REDACTED_DIR
    keep = set()
    for document in result.documents:
        name = _redacted_name(document["document_id"])
        keep.add(name)
        path = redacted / name
        text = result.texts[document["document_id"]]
        if not path.is_file() or path.read_text(encoding="utf-8") != text:
            write_atomic(path, text)
    if redacted.is_dir():
        for stale in redacted.glob("*.txt"):
            if stale.name not in keep:
                stale.unlink(missing_ok=True)


def rule_input_digests(stage_cache: Path) -> dict[str, str | None]:
    """sha256 of each rule input file in ``stage_cache``, or None when the file is absent."""
    digests: dict[str, str | None] = {}
    for name in RULE_INPUT_FILES:
        try:
            digests[name] = hashlib.sha256((stage_cache / name).read_bytes()).hexdigest()
        except FileNotFoundError:
            digests[name] = None
    return digests


def _redacted_name(document_id: str) -> str:
    # ``~`` cannot appear in a document id, so ``a/b`` and ``a~b`` cannot land on one file.
    # A deep path is shortened and keeps a hash, so the name stays under file-system limits.
    flat = document_id.replace("/", "~")
    if len(flat) > MAX_REDACTED_NAME_CHARS:
        digest = hashlib.sha256(document_id.encode("utf-8")).hexdigest()[:16]
        flat = f"{flat[: MAX_REDACTED_NAME_CHARS - 17]}-{digest}"
    return flat + ".txt"


def _assemble(outcomes: dict[str, dict[str, Any]], reused: int) -> IngestResult:
    result = IngestResult(files=[], reused=reused)
    first_by_sha: dict[str, str] = {}
    rule_owner: dict[str, str] = {}
    entries: list[dict[str, Any]] = []
    aliases: dict[str, str] = {}
    for relative in sorted(outcomes):
        outcome = outcomes[relative]
        row = dict(outcome["file"])
        if row["status"] == SAFE:
            original = first_by_sha.setdefault(row["sha256"], relative)
            if original != relative:
                aliases[row["document_id"]] = outcomes[original]["file"]["document_id"]
                row["duplicate_of"] = original
                row["span_count"] = 0
                row["rule_count"] = 0
                result.files.append(row)
                continue
            clash = sorted(rule["policy_id"] for rule in outcome.get("rules", []) if rule["policy_id"] in rule_owner)
            if clash:
                detail = "; ".join(f"policy_id {pid!r} is already defined in {rule_owner[pid]}" for pid in clash)
                result.files.append(_rejected(row, DUPLICATE_POLICY_ID, detail)["file"])
                continue
            for rule in outcome.get("rules", []):
                rule_owner[rule["policy_id"]] = relative
            result.rules.extend(outcome.get("rules", []))
            if "document" in outcome:
                result.documents.append(outcome["document"])
                result.spans.extend(outcome["spans"])
                result.texts[outcome["document"]["document_id"]] = outcome["text"]
                entries.extend(outcome["glossary"])
        result.files.append(row)
    result.glossary = _merge_glossary(entries)
    link_glossary_terms(result.spans, [term["key"] for term in result.glossary["terms"]])
    result.rules = [_with_evidence(rule, result, aliases) for rule in result.rules]
    return result


def _with_evidence(rule_row: dict[str, Any], result: IngestResult, aliases: Mapping[str, str]) -> dict[str, Any]:
    """Check a pre-authored rule's provenance against the redacted documents of this run.

    The rule is not changed and not routed here. ``evidence`` tells routing what was found:
    the cited document and span exist, and the quote appears verbatim in that span.
    """
    provenance = rule_row["rule"]["provenance"]
    cited = provenance["document_id"]
    if cited == rule_row["document_id"]:
        return {**rule_row, "evidence": {"status": EVIDENCE_SELF_AUTHORED}}
    document_id = aliases.get(cited, cited)
    document = next((item for item in result.documents if item["document_id"] == document_id), None)
    if document is None:
        return {**rule_row, "evidence": {"status": EVIDENCE_DOCUMENT_NOT_FOUND, "document_id": cited}}
    evidence: dict[str, Any] = {"document_id": document_id, "document_sha256": document["sha256"]}
    stated_sha = provenance.get("document_sha256")
    if stated_sha is not None and stated_sha != document["sha256"]:
        return {**rule_row, "evidence": {**evidence, "status": EVIDENCE_DOCUMENT_CHANGED}}
    spans = [span for span in result.spans if span["document_id"] == document_id]
    span_id = provenance.get("source_span_id")
    if span_id is not None:
        spans = [span for span in spans if span["source_span_id"] == span_id]
        if not spans:
            return {**rule_row, "evidence": {**evidence, "status": EVIDENCE_SPAN_NOT_FOUND, "source_span_id": span_id}}
    found = [span["source_span_id"] for span in spans if quote_in_span(provenance["source_quote"], span["text"])]
    status = EVIDENCE_QUOTE_FOUND if found else EVIDENCE_QUOTE_NOT_FOUND
    return {**rule_row, "evidence": {**evidence, "status": status, "span_ids": found}}


def _merge_glossary(entries: list[dict[str, Any]]) -> dict[str, Any]:
    by_key: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        by_key.setdefault(entry["key"], []).append(entry)
    terms = []
    conflicts = []
    for key in sorted(by_key):
        group = by_key[key]
        terms.append({**group[0], "also_defined_in": [entry["source_span_id"] for entry in group[1:]]})
        definitions = {" ".join(entry["definition"].casefold().split()) for entry in group}
        if len(definitions) > 1:
            conflicts.append(
                {
                    "key": key,
                    "definitions": [
                        {"source_span_id": entry["source_span_id"], "definition": entry["definition"]}
                        for entry in group
                    ],
                }
            )
    return {"terms": terms, "conflicts": conflicts}


def _row(relative: str, sha256: str, document_id: str) -> dict[str, Any]:
    return {
        "path": relative,
        "sha256": sha256,
        "status": None,
        "document_id": document_id,
        "format": None,
        "reason": None,
        "detail": None,
        "extraction": None,
        "span_count": 0,
        "rule_count": 0,
        "findings": [],
        "redactions": {},
        "duplicate_of": None,
    }


def _rejected(row: dict[str, Any], reason: str, detail: str) -> dict[str, Any]:
    row.update(status=REJECTED, reason=reason, detail=detail, extraction=None, span_count=0, rule_count=0)
    return {"file": row}


def _privacy_detail(scan: privacy.PrivacyResult) -> str:
    kinds = sorted({finding.kind for finding in scan.findings if finding.category == privacy.CREDENTIAL})
    if scan.reason == privacy.SECRET_PATTERN:
        return f"credential found ({', '.join(kinds)}); remove it from the file and prepare again"
    return "personal data remained after redaction; remove it from the file and prepare again"


def _document(row: dict[str, Any], sections: list[dict[str, Any]], lines: tuple[Line, ...]) -> dict[str, Any]:
    pages = sorted({line.page for line in lines if line.page is not None})
    return {
        "document_id": row["document_id"],
        "path": row["path"],
        "sha256": row["sha256"],
        "format": row["format"],
        "redacted_text": f"{REDACTED_DIR}/{_redacted_name(row['document_id'])}",
        "page_count": len(pages) or None,
        "sections": [
            {key: section[key] for key in ("section", "title", "heading_path")}
            | {"span_count": len(section["span_ids"])}
            for section in sections
        ],
    }


def _preauthored(row: dict[str, Any], candidates: list[Any]) -> dict[str, Any]:
    if not candidates:
        return _rejected(row, INVALID_POLICY_RULE_IR, "policy_rules is empty")
    errors: list[str] = []
    rules: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(candidates):
        where = f"rule {index + 1}"
        if not isinstance(raw, dict):
            errors.append(f"{where}: must be a mapping")
            continue
        raw = _with_source(raw, row)
        result = normalize_candidate(raw)
        if result.rule is None:
            errors.extend(f"{where}: {message}" for message in result.errors)
            continue
        policy_id = result.rule.policy_id
        if policy_id in seen:
            errors.append(f"{where}: policy_id {policy_id!r} appears more than once in this file")
            continue
        seen.add(policy_id)
        rules.append(
            {
                "policy_id": policy_id,
                "source_path": row["path"],
                "document_id": row["document_id"],
                "source_sha256": row["sha256"],
                "extraction": EXTRACTION_SKIPPED,
                "rule": result.rule.model_dump(mode="json"),
                "flags": [dataclasses.asdict(flag) for flag in result.flags],
            }
        )
    if errors:
        return _rejected(row, INVALID_POLICY_RULE_IR, "; ".join(errors))
    row.update(status=SAFE, extraction=EXTRACTION_SKIPPED, rule_count=len(rules))
    return {"file": row, "rules": rules}


def _with_source(raw: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    provenance = raw.get("provenance")
    if not isinstance(provenance, dict):
        return raw
    filled = dict(provenance)
    filled.setdefault("document_id", row["document_id"])
    # The file's own hash describes the file only when the rule cites the file itself.
    if filled["document_id"] == row["document_id"]:
        filled.setdefault("document_sha256", row["sha256"])
    return {**raw, "provenance": filled}


def _rule_candidates(parsed: Any) -> list[Any] | None:
    """The rule list when the file is Policy Rule IR, otherwise None (the file is read as text)."""
    if isinstance(parsed, dict):
        if "policy_rules" in parsed:
            rules = parsed["policy_rules"]
            return rules if isinstance(rules, list) else [rules]
        return [parsed] if _rule_like(parsed) else None
    if isinstance(parsed, list) and parsed and all(_rule_like(item) for item in parsed):
        return parsed
    return None


def _rule_like(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and isinstance(value.get("rule"), dict)
        and ("policy_id" in value or "policy_id_candidate" in value)
    )


def _parse_structured(text: str, fmt: str) -> Any:
    if fmt == "json":
        try:
            return json.loads(text, object_pairs_hook=_unique_pairs, parse_constant=_reject_constant)
        except json.JSONDecodeError as exc:
            raise ValueError(f"not valid JSON: {exc.msg} at line {exc.lineno}") from exc
    try:
        if any(isinstance(event, yaml.AliasEvent) for event in yaml.parse(text)):
            raise ValueError("YAML aliases are not accepted in policy files")
        return yaml.load(text, Loader=_UniqueKeyLoader)  # noqa: S506 - _UniqueKeyLoader extends SafeLoader
    except yaml.YAMLError as exc:
        raise ValueError(f"not valid YAML: {' '.join(str(exc).split())}") from exc


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for key, value in pairs:
        if key in data:
            raise ValueError(f"duplicate JSON key {key!r}")
        data[key] = value
    return data


def _reject_constant(name: str) -> Any:
    raise ValueError(f"JSON constant {name} is not allowed")


def _read_source(input_dir: Path, relative: str, expected: str) -> bytes:
    try:
        data = (input_dir / relative).read_bytes()
    except OSError as exc:
        raise SourceError(f"cannot read {relative}: {exc}") from exc
    if hashlib.sha256(data).hexdigest() != expected:
        raise SourceChangedError(f"{relative} changed while prepare was reading it; run stage=prepare again.")
    return data


def _cache_key(
    relative: str,
    document_id: str,
    sha256: str,
    privacy_labels: Mapping[str, Sequence[str]] | None = None,
) -> str:
    labels = json.dumps(privacy_labels or {}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    payload = f"safety-ingest-v{INGEST_VERSION}\0{relative}\0{document_id}\0{sha256}\0{labels}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_entry(relative: str, sha256: str, outcome: dict[str, Any]) -> dict[str, Any]:
    return {"ingest_version": INGEST_VERSION, "path": relative, "sha256": sha256, "outcome": outcome}


def _read_cache(path: Path, relative: str, sha256: str) -> dict[str, Any] | None:
    """A cached outcome for exactly this file version, or None. A damaged entry is recomputed."""
    try:
        entry = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not (
        isinstance(entry, dict)
        and entry.get("ingest_version") == INGEST_VERSION
        and entry.get("path") == relative
        and entry.get("sha256") == sha256
        and isinstance(entry.get("outcome"), dict)
        and isinstance(entry["outcome"].get("file"), dict)
        and entry["outcome"]["file"].get("sha256") == sha256
    ):
        return None
    return entry["outcome"]


def _read_document_ids(path: Path) -> dict[str, str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    ids = data.get("ids") if isinstance(data, dict) else None
    if not isinstance(ids, dict):
        return {}
    return {key: value for key, value in ids.items() if isinstance(key, str) and isinstance(value, str)}


def _usable_id(value: str) -> bool:
    return bool(value) and len(value) <= 200 and not any(char in value for char in "#~\\\n")


def _id_candidates(path: str) -> list[str]:
    pure = PurePosixPath(path)
    stem = _slug_path(pure.with_suffix("").as_posix())
    full = _slug_path(path)
    return [stem] if stem == full else [stem, full]


def _slug_path(path: str) -> str:
    parts = []
    for part in path.split("/"):
        folded = unicodedata.normalize("NFKD", part.replace("đ", "d").replace("Đ", "D"))
        ascii_part = "".join(char for char in folded if not unicodedata.combining(char))
        ascii_part = ascii_part.encode("ascii", "ignore").decode().casefold()
        slug = re.sub(r"[^a-z0-9._-]+", "-", ascii_part).strip("-.")
        parts.append(slug or "document")
    return "/".join(parts)
