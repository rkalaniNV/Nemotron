# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Grounded glossary and cross-reference annotations after ingest.

Python keeps punctuation definitions and ``§`` / ``第`` citations. The model may add
a definition or a cross-reference only when its evidence is an exact part of the span
and a citation target appears in that evidence. It never changes section labels or span ids.
A failed pass keeps the ingest glossary.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from nemotron.steps.byob.runtime.benchmark_families.safety.artifacts import write_json
from nemotron.steps.byob.runtime.benchmark_families.safety.confidence import quote_in_span
from nemotron.steps.byob.runtime.benchmark_families.safety.ingest import _merge_glossary
from nemotron.steps.byob.runtime.benchmark_families.safety.paraphrase import Completer
from nemotron.steps.byob.runtime.benchmark_families.safety.spans import (
    _MAX_TERM_WORDS,
    _definition,
    cross_reference,
    link_glossary_terms,
    sections_from_spans,
    term_key,
)

PROMPT_VERSION = 2
CACHE_DIR = "annotate"
REPORT_FILE = "annotate_report.json"
GLOSSARY_FILE = "annotated_glossary.json"
ANNOTATE_OUTPUTS = (REPORT_FILE, GLOSSARY_FILE)
ANNOTATE_TEMPERATURE = 0.0

SYSTEM_PROMPT = """You mark definitions and section citations in one policy span.

The JSON payload has the span text and the numbered section labels of that document.
Do not rewrite the span. Copy evidence exactly from text.

is_definition is true only when the span defines a term. term_evidence and
definition_evidence must be copied from text.

cross_references lists citations of other numbered sections. evidence is copied from
text. target_section is one of numbered_sections and must appear in evidence. Prefer
the longest matching label, so 4.3 not 4 when the evidence names 4.3.

If the span is not a definition and cites no section, return is_definition false and
an empty cross_references list. Do not guess."""

PROMPT = "Span as JSON:\n{{ payload }}\n\nReturn grounded definitions and cross-references."


class AnnotatedReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence: str = Field(min_length=1)
    target_section: str = Field(min_length=1)


class SpanAnnotation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    is_definition: bool = False
    term_evidence: str = ""
    definition_evidence: str = ""
    cross_references: list[AnnotatedReference] = Field(default_factory=list)


def apply_annotations(
    spans: Sequence[Mapping[str, Any]],
    terms: Sequence[Mapping[str, Any]],
    conflicts: Sequence[Mapping[str, Any]],
    extractor: Completer | None,
    cache_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Add grounded glossary entries and citations. Prepare artifacts are not rewritten."""
    report = {
        "status": "skipped",
        "requested": 0,
        "cached": 0,
        "definitions": 0,
        "cross_references": 0,
        "skipped_reason": None,
        "glossary_source": "ingest",
        "failures": [],
    }
    copied = json.loads(json.dumps(list(spans)))
    if extractor is None or not callable(getattr(extractor, "complete", None)):
        report["skipped_reason"] = "no_model" if extractor is None else "extractor_cannot_complete"
        return copied, list(terms), list(conflicts), report

    identity = dict(getattr(extractor, "identity", {}) or {})
    stored_params = identity.get("inference_parameters")
    params = dict(stored_params) if isinstance(stored_params, Mapping) else {}
    params["temperature"] = ANNOTATE_TEMPERATURE
    identity = {**identity, "inference_parameters": params}
    folder = cache_dir / CACHE_DIR
    folder.mkdir(parents=True, exist_ok=True)
    jobs = _jobs(copied, identity, folder)
    if not jobs:
        report["skipped_reason"] = "no_work"
        return copied, list(terms), list(conflicts), report

    replies: dict[str, Any] = {}
    pending: list[tuple[dict[str, Any], Path, dict[str, Any]]] = []
    batch: Mapping[str, Any] | None = None
    for span, path, payload in jobs:
        cached = _read_cache(path)
        if cached is not None:
            replies[span["source_span_id"]] = cached
            report["cached"] += 1
            continue
        pending.append((span, path, payload))
    if pending:
        rows = [
            {
                "request_id": span["source_span_id"],
                "payload": json.dumps(payload, ensure_ascii=False, sort_keys=True),
            }
            for span, _path, payload in pending
        ]
        report["requested"] = len(rows)
        try:
            raw = extractor.complete(
                rows,
                system_prompt=SYSTEM_PROMPT,
                prompt=PROMPT,
                output_format=SpanAnnotation,
                prefix="safety-annotate-",
                inference_parameters=params,
            )
        except Exception:
            report["skipped_reason"] = "model_unavailable"
            raw = None
        if isinstance(raw, Mapping):
            batch = raw
            replies.update(raw)
            for span, path, _payload in pending:
                parsed = _parse(replies.get(span["source_span_id"]))
                if parsed is not None:
                    write_json(path, {"prompt_version": PROMPT_VERSION, **parsed.model_dump()})
        elif raw is not None:
            report["skipped_reason"] = report.get("skipped_reason") or "invalid_batch_reply"
    if report["cached"] or batch is not None:
        report["status"] = "applied"

    added = []
    for span, _path, _payload in jobs:
        parsed = _parse(replies.get(span["source_span_id"]))
        if parsed is None:
            continue
        entry = _grounded_definition(span, parsed)
        if entry is not None:
            added.append(entry)
            report["definitions"] += 1
        report["cross_references"] += _grounded_references(span, parsed, copied)
    merged = _merge_entries(terms, added)
    link_glossary_terms(copied, [term["key"] for term in merged["terms"]])
    if added:
        report["glossary_source"] = "ingest_and_model" if terms else "model"
    return copied, merged["terms"], merged["conflicts"], report


def _jobs(
    spans: Sequence[dict[str, Any]], identity: Mapping[str, Any], folder: Path
) -> list[tuple[dict[str, Any], Path, dict[str, Any]]]:
    labels_by_doc: dict[str, list[str]] = {}
    jobs = []
    for span in spans:
        document_id = str(span.get("document_id") or "")
        if document_id not in labels_by_doc:
            labels_by_doc[document_id] = [item["section"] for item in sections_from_spans(spans, document_id)]
        labels = labels_by_doc[document_id]
        if not _is_candidate(span, labels):
            continue
        payload = {
            "text": span.get("text"),
            "section": span.get("section"),
            "numbered_sections": labels,
        }
        path = folder / f"{_cache_key(str(span.get('source_span_id') or ''), payload, identity)}.json"
        jobs.append((span, path, payload))
    return jobs


def _is_candidate(span: Mapping[str, Any], labels: Sequence[str]) -> bool:
    text = span.get("text")
    if not isinstance(text, str) or not text.strip():
        return False
    if _definition(text):
        return bool(_citation_labels(text, span.get("section"), labels))
    stripped = text.strip()
    if any(mark in stripped for mark in ":：") or stripped[:1] in "\"“「『'‘":
        return True
    return bool(_citation_labels(text, span.get("section"), labels))


def _citation_labels(text: str, section: Any, labels: Sequence[str]) -> list[str]:
    found = []
    for label in labels:
        if label == section:
            continue
        if len(label) == 1 and label.isalpha():
            continue
        if _label_in_text(label, text):
            found.append(label)
    return found


def _label_in_text(label: str, text: str) -> bool:
    """True when ``label`` appears as its own number, so ``4`` does not match ``4.3``."""
    if not label:
        return False
    return re.search(rf"(?<![\d.]){re.escape(label)}(?!\.?\d)", text) is not None


def _grounded_definition(span: Mapping[str, Any], parsed: SpanAnnotation) -> dict[str, Any] | None:
    if not parsed.is_definition:
        return None
    text = str(span.get("text") or "")
    term = parsed.term_evidence.strip()
    definition = parsed.definition_evidence.strip()
    if not term or not definition:
        return None
    if not quote_in_span(term, text) or not quote_in_span(definition, text):
        return None
    if len(term.split()) > _MAX_TERM_WORDS:
        return None
    return {
        "term": term,
        "key": term_key(term),
        "definition": definition.rstrip(".;。．").strip(),
        "document_id": span.get("document_id"),
        "source_span_id": span.get("source_span_id"),
    }


def _grounded_references(span: dict[str, Any], parsed: SpanAnnotation, spans: Sequence[Mapping[str, Any]]) -> int:
    numbered = sections_from_spans(spans, str(span.get("document_id") or ""))
    labels = {item["section"] for item in numbered}
    existing = list(span.get("cross_references") or [])
    seen = {item.get("target_section") for item in existing if isinstance(item, Mapping)}
    added = 0
    text = str(span.get("text") or "")
    for item in parsed.cross_references:
        evidence = item.evidence.strip()
        target = item.target_section.strip()
        if not evidence or not quote_in_span(evidence, text) or target not in labels:
            continue
        if not _label_in_text(target, evidence):
            continue
        if any(other.startswith(f"{target}.") and _label_in_text(other, evidence) for other in labels):
            continue
        ref = cross_reference(span, evidence, target, numbered)
        if ref is None or ref["target_section"] in seen:
            continue
        existing.append(ref)
        seen.add(ref["target_section"])
        added += 1
    span["cross_references"] = existing
    return added


def _merge_entries(existing: Sequence[Mapping[str, Any]], added: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    seen_spans: set[tuple[str, str]] = set()
    for term in existing:
        if not isinstance(term, Mapping) or not isinstance(term.get("key"), str):
            continue
        base = {
            "term": term.get("term"),
            "key": term["key"],
            "definition": term.get("definition"),
            "document_id": term.get("document_id"),
            "source_span_id": term.get("source_span_id"),
        }
        rows.append(dict(base))
        seen_spans.add((str(base["key"]), str(base["source_span_id"])))
        for span_id in term.get("also_defined_in") or []:
            if not isinstance(span_id, str):
                continue
            rows.append({**base, "source_span_id": span_id})
            seen_spans.add((str(base["key"]), span_id))
    for entry in added:
        marker = (str(entry.get("key")), str(entry.get("source_span_id")))
        if marker in seen_spans:
            continue
        rows.append(dict(entry))
        seen_spans.add(marker)
    return _merge_glossary(rows)


def _parse(reply: Any) -> SpanAnnotation | None:
    if reply is None:
        return None
    if isinstance(reply, SpanAnnotation):
        return reply
    if isinstance(reply, BaseModel):
        reply = reply.model_dump()
    if isinstance(reply, str):
        try:
            reply = json.loads(reply)
        except json.JSONDecodeError:
            return None
    try:
        return SpanAnnotation.model_validate(reply)
    except ValidationError:
        return None


def _cache_key(span_id: str, payload: Mapping[str, Any], identity: Mapping[str, Any]) -> str:
    material = {
        "prompt_version": PROMPT_VERSION,
        "system_prompt": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "prompt": hashlib.sha256(PROMPT.encode("utf-8")).hexdigest(),
        "model": dict(identity),
        "span_id": span_id,
        "payload": dict(payload),
    }
    return hashlib.sha256(json.dumps(material, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _read_cache(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or data.get("prompt_version") != PROMPT_VERSION:
        return None
    parsed = _parse(data)
    return parsed.model_dump() if parsed is not None else None
