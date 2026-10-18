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

"""Evidence windows: the only document text an extraction model sees.

A window is one clause (the target span) with its heading path, the clause before and after
it in the same section, and the glossary definitions the clause uses. The whole document is
never sent, and every window is built from redacted text.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from nemotron.steps.byob.runtime.benchmark_families.safety.confidence import quote_in_span
from nemotron.steps.byob.runtime.benchmark_families.safety.spans import structural_definition


@dataclass(frozen=True)
class EvidenceWindow:
    document_id: str
    document_sha256: str
    section: str
    heading_path: tuple[str, ...]
    target: Mapping[str, Any]
    neighbors: tuple[Mapping[str, Any], ...]
    glossary: tuple[Mapping[str, str], ...]
    grounding_texts: tuple[str, ...] = ()

    @property
    def window_id(self) -> str:
        return self.target["source_span_id"]

    def payload(self) -> dict[str, Any]:
        """What the model receives. Neighbors and definitions are context, not quote sources."""
        return {
            "document_id": self.document_id,
            "heading_path": list(self.heading_path),
            "clause": {"source_span_id": self.window_id, "text": self.target["text"]},
            "neighboring_clauses": [
                {"source_span_id": span["source_span_id"], "text": span["text"]} for span in self.neighbors
            ],
            "glossary": [dict(entry) for entry in self.glossary],
        }


def build_windows(
    documents: Sequence[Mapping[str, Any]],
    spans: Sequence[Mapping[str, Any]],
    glossary_terms: Sequence[Mapping[str, Any]],
) -> list[EvidenceWindow]:
    """One window per span of every safe text document, in document and span order.

    Raises ValueError when the ingest artifacts are not the shape prepare writes.
    """
    _mappings(documents, "document")
    _mappings(spans, "span")
    _mappings(glossary_terms, "glossary term")
    for document in documents:
        _text_fields(document, ("document_id", "sha256"), "document")
    for term in glossary_terms:
        _text_fields(term, ("key", "term", "definition"), "glossary term")
    seen: set[str] = set()
    for span in spans:
        _text_fields(span, ("source_span_id", "document_id", "section", "text"), "span")
        heading = span.get("heading_path")
        if not isinstance(heading, (list, tuple)) or not all(isinstance(item, str) for item in heading):
            raise ValueError(f"span {span['source_span_id']} has no heading path")
        if span["source_span_id"] in seen:
            raise ValueError(f"duplicate span id {span['source_span_id']}")
        seen.add(span["source_span_id"])
        terms = span.get("glossary_terms", [])
        if not isinstance(terms, list) or not all(isinstance(item, str) for item in terms):
            raise ValueError(f"span {span['source_span_id']} has a glossary_terms value that is not a list of terms")
    known = {document["document_id"] for document in documents}
    unknown = sorted({span["document_id"] for span in spans} - known)
    if unknown:
        raise ValueError("spans cite documents that are not in the index: " + ", ".join(unknown))
    sha_by_document = {document["document_id"]: document["sha256"] for document in documents}
    definitions = {term["key"]: term for term in glossary_terms}
    structural_by_document: dict[str, list[tuple[str, str, str]]] = {}
    for span in spans:
        definition = structural_definition(span)
        if definition is not None:
            term, text = definition
            structural_by_document.setdefault(span["document_id"], []).append((span["source_span_id"], term, text))
    by_section: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    positions = []
    for span in spans:
        siblings = by_section.setdefault((span["document_id"], span["section"]), [])
        positions.append(len(siblings))
        siblings.append(span)
    windows = []
    for span, index in zip(spans, positions):
        siblings = by_section[(span["document_id"], span["section"])]
        neighbors = tuple(siblings[position] for position in (index - 1, index + 1) if 0 <= position < len(siblings))
        glossary = tuple(
            {"term": definitions[key]["term"], "definition": definitions[key]["definition"]}
            for key in span.get("glossary_terms", [])
            if key in definitions
        )
        structural = tuple(
            text
            for span_id, term, text in structural_by_document.get(span["document_id"], [])
            if span_id != span["source_span_id"] and quote_in_span(term, span["text"])
        )
        windows.append(
            EvidenceWindow(
                document_id=span["document_id"],
                document_sha256=sha_by_document.get(span["document_id"], ""),
                section=span["section"],
                heading_path=tuple(span["heading_path"]),
                target=span,
                neighbors=neighbors,
                glossary=glossary,
                grounding_texts=tuple(item["text"] for item in neighbors) + structural,
            )
        )
    return windows


def _mappings(items: Sequence[Any], name: str) -> None:
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        raise ValueError(f"{name}s must be a list")
    if any(not isinstance(item, Mapping) for item in items):
        raise ValueError(f"every {name} must be an object")


def _text_fields(item: Mapping[str, Any], fields: tuple[str, ...], name: str) -> None:
    missing = [field for field in fields if not isinstance(item.get(field), str) or not item[field].strip()]
    if missing:
        raise ValueError(f"a {name} is missing {', '.join(missing)}")
