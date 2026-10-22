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

"""Source spans, sections, glossary, and same-document cross-references.

A span is one sentence (or one table row) of redacted document text. Its ID is
``<document_id>#<section>#span-NN``, where NN counts spans inside that section. The ID
depends only on the document's own text, so an unchanged file keeps its IDs. Character
offsets index the redacted document text that ingest stores beside the spans.
"""

import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from nemotron.steps.byob.runtime.benchmark_families.safety.confidence import fold_latin, latin_token
from nemotron.steps.byob.runtime.benchmark_families.safety.documents import DocumentError, Line

PREAMBLE = "preamble"
MAX_SPAN_CHARS = 1200
MAX_CROSS_REFERENCE_TARGETS = 50

# A roman numeral, non-empty. "IIII" does not match; "IV" and "III" do.
_ROMAN = r"(?=[IVXLCDM])M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})"
_NUMBER = rf"(?:\d+(?:\.\d+)*|(?:{_ROMAN})(?:\.\d+)*)"
_DOTTED_NUMBER = r"\d+\.\d+(?:\.\d+)*"
_ATX = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
# Dotted numbers are section labels (`4.3 Title`). A single `1.` stays a list marker.
_DOTTED_HEADING = re.compile(rf"^(?P<number>{_DOTTED_NUMBER})(?:[.):](?=\s|$)|[ \t]+(?=\S)|[ \t]+[-–—](?=\s)|$)", re.I)
_STRUCTURAL_HEADING = re.compile(
    r"^(?:第\s*(?P<cjk>\d{1,6})\s*(?P<unit>[条章节編编])|§\s*(?P<section>\d+(?:\.\d+)*))\s*(?P<title>\S.*)?$"
)
# Labels of already-detected headings may start with one unlisted word (`Điều 1.`, `4. Restricted`).
# Arabic numbers may be followed by a space (`4.1 Transfers`). Roman numerals need `.` / `)` / `:`
# so "Phạm vi áp dụng" is not read as section VI.
_TITLE_ARABIC = re.compile(r"^(?:[^\W\d_]+[ \t]+)?(\d+(?:\.\d+)*)(?:[.):](?=\s|$)|[ \t]+(?=\S)|$)")
_TITLE_ROMAN = re.compile(
    rf"^(?:[^\W\d_]+[ \t]+)?({_ROMAN})(?:\.\d+)*(?:[.):](?=\s|$)|$)",
    re.I,
)
_NUMBERED_LABEL = re.compile(rf"^{_NUMBER}$", re.I)
_STRUCTURAL_CROSS_REF = re.compile(r"(?:§\s*(?P<section>\d+(?:\.\d+)*)|第\s*(?P<cjk>\d{1,6})\s*[条章节編编])")
_TABLE_DIVIDER = re.compile(r"^\|?[\s:|-]*-[\s:|-]*\|?$")
# A single letter marks a list only in lowercase, so "A. Nguyen" and "I. Chương" stay sentences.
_LIST_MARKER = re.compile(r"(?:[-*+•▪◦]|\d{1,3}[.)]|[a-z][.)]|\((?:\d{1,3}|[A-Za-z])\))[ \t]+")
_QUOTE_MARKER = re.compile(r"(?:>[ \t]?)+")
_SENTENCE_END = re.compile(r"(?:[.!?…]+[\"'”’)\]]*(?=\s|$)|[。！？]+[\"'”’）)\]]*)")
_QUOTED_COLON = re.compile(r"^[“\"'‘「『]([^”\"'’」』]{1,80})[”\"'’」』]\s*[:：]\s*(\S.*)$")
_BOLD_DEFINITION = re.compile(r"^\*\*([^*]{1,80})\*\*\s*[:\-–—]\s+(\S.*)$")
_MAX_TERM_WORDS = 8


@dataclass
class StructuredDocument:
    text: str
    spans: list[dict[str, Any]] = field(default_factory=list)
    sections: list[dict[str, Any]] = field(default_factory=list)
    glossary: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class _Heading:
    level: int
    title: str
    section: str


def build_document(document_id: str, lines: Sequence[Line], *, markdown_headings: bool) -> StructuredDocument:
    """Split redacted lines into sections and sentence spans."""
    text = "\n".join(line.text for line in lines)
    document = StructuredDocument(text)
    sections: dict[str, dict[str, Any]] = {}
    counters: dict[str, int] = {}
    used_labels = {PREAMBLE}
    stack: list[_Heading] = []
    offset = 0
    in_fence = False
    for number, line in enumerate(lines, start=1):
        start = offset
        offset += len(line.text) + 1
        stripped = line.text.strip()
        if stripped.startswith(("```", "~~~")):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        heading = _heading(line, stripped, markdown_headings)
        if heading is not None:
            level, title, body_start = heading
            while stack and stack[-1].level >= level:
                stack.pop()
            parent = stack[-1].section if stack else None
            label = _allocate_label(_preferred_label(title, stack, number), parent, used_labels)
            stack.append(_Heading(level, title, label))
            _section(sections, stack)
            if body_start is None:
                continue
            ranges = _body_ranges(line.text, body_start)
        else:
            ranges = _body_ranges(line.text, None)
        if not ranges:
            continue
        section = _section(sections, stack)
        for begin, end, kind in ranges:
            counters[section["section"]] = counters.get(section["section"], 0) + 1
            span_id = f"{document_id}#{section['section']}#span-{counters[section['section']]:02d}"
            section["span_ids"].append(span_id)
            document.spans.append(
                {
                    "source_span_id": span_id,
                    "document_id": document_id,
                    "section": section["section"],
                    "heading_path": list(section["heading_path"]),
                    "page": line.page,
                    "line": number,
                    "kind": kind,
                    "char_start": start + begin,
                    "char_end": start + end,
                    "text": line.text[begin:end],
                }
            )
    if in_fence:
        raise DocumentError("unclosed_code_fence", "a code fence starts with ``` or ~~~ and is never closed")
    document.sections = list(sections.values())
    _link_cross_references(document)
    document.glossary = _glossary(document)
    _link_glossary_terms(document)
    return document


def _heading(line: Line, stripped: str, markdown_headings: bool) -> tuple[int, str, int | None] | None:
    """Return (level, title, start of body text on the same line or None)."""
    if not stripped:
        return None
    if line.heading_level is not None:
        return line.heading_level, stripped, None
    if markdown_headings:
        match = _ATX.match(stripped)
        if match:
            return len(match.group(1)), match.group(2).strip(), None
    dotted = _DOTTED_HEADING.match(stripped)
    if dotted:
        number = dotted.group("number")
        level = 1 + number.count(".")
        lead = len(line.text) - len(line.text.lstrip())
        return level, stripped, lead + dotted.end()
    structural = _STRUCTURAL_HEADING.match(stripped)
    if structural:
        if structural.group("cjk"):
            number = structural.group("cjk")
            level = 1 if structural.group("unit") in "章編编" else 2
        else:
            number = structural.group("section") or ""
            level = 1 + number.count(".")
        title = f"{number} {(structural.group('title') or '').strip()}".strip()
        return level, title, None
    return None


def _preferred_label(title: str, stack: list[_Heading], line_number: int) -> str:
    """The label this heading would like. A repeated one is disambiguated later."""
    numbered = _title_number(title)
    if numbered:
        return _canonical_number(numbered)
    slug = _slug(title) or f"heading-{line_number}"
    for parent in reversed(stack):
        if _title_number(parent.title):
            return f"{parent.section}.{slug}"
    return slug


def _title_number(title: str) -> str:
    for pattern in (_TITLE_ARABIC, _TITLE_ROMAN):
        match = pattern.match(title)
        if match:
            return match.group(1)
    return ""


def _allocate_label(preferred: str, parent: str | None, used: set[str]) -> str:
    """Keep a unique number as itself. A repeat becomes ``<parent>/<number>``, then ``~2``."""
    if preferred not in used:
        used.add(preferred)
        return preferred
    base = f"{parent}/{preferred}" if parent else preferred
    if parent and base not in used:
        used.add(base)
        return base
    suffix = 2
    while f"{base}~{suffix}" in used:
        suffix += 1
    label = f"{base}~{suffix}"
    used.add(label)
    return label


def _canonical_number(raw: str) -> str:
    head, dot, tail = raw.partition(".")
    if head[:1].isalpha():
        head = head.upper()
    return f"{head}{dot}{tail}" if dot else head


def _section(sections: dict[str, dict[str, Any]], stack: list[_Heading]) -> dict[str, Any]:
    label = stack[-1].section if stack else PREAMBLE
    if label not in sections:
        sections[label] = {
            "section": label,
            "title": stack[-1].title if stack else "",
            "heading_path": [heading.title for heading in stack],
            "span_ids": [],
        }
    return sections[label]


def _body_ranges(text: str, start: int | None) -> list[tuple[int, int, str]]:
    begin = len(text) - len(text.lstrip()) if start is None else start
    end = len(text.rstrip())
    while begin < end and text[begin] in " \t":
        begin += 1
    if begin >= end:
        return []
    kind = "paragraph"
    body = text[begin:end]
    if body.startswith("|"):
        if _TABLE_DIVIDER.match(body):
            return []
        begin += 1
        if text[end - 1] == "|" and end - 1 > begin:
            end -= 1
        kind = "table_row"
    else:
        for marker, marker_kind in ((_QUOTE_MARKER, "quote"), (_LIST_MARKER, "list_item")):
            match = marker.match(text, begin, end)
            if match:
                begin = match.end()
                kind = marker_kind
    if kind == "table_row":
        pieces = [(begin, end)]
    else:
        pieces = _sentences(text, begin, end)
    ranges: list[tuple[int, int, str]] = []
    for piece_start, piece_end in pieces:
        for chunk_start, chunk_end in _cap(text, piece_start, piece_end):
            chunk_start, chunk_end = _trim(text, chunk_start, chunk_end)
            if any(char.isalnum() for char in text[chunk_start:chunk_end]):
                ranges.append((chunk_start, chunk_end, kind))
    return ranges


def _sentences(text: str, begin: int, end: int) -> list[tuple[int, int]]:
    pieces: list[tuple[int, int]] = []
    cursor = begin
    for match in _SENTENCE_END.finditer(text, begin, end):
        following = match.end()
        while following < end and text[following].isspace():
            following += 1
        if following >= end:
            break
        word = re.search(r"([\w.]+)$", text[cursor : match.start()])
        token = word.group(1).rstrip(".") if word else ""
        if _abbreviation(token):
            continue
        pieces.append((cursor, match.end()))
        cursor = following
    pieces.append((cursor, end))
    return pieces


def _cap(text: str, begin: int, end: int) -> list[tuple[int, int]]:
    chunks: list[tuple[int, int]] = []
    while end - begin > MAX_SPAN_CHARS:
        cut = text.rfind(" ", begin, begin + MAX_SPAN_CHARS)
        if cut <= begin:
            cut = begin + MAX_SPAN_CHARS
        chunks.append((begin, cut))
        begin = cut
    chunks.append((begin, end))
    return chunks


def _trim(text: str, begin: int, end: int) -> tuple[int, int]:
    while begin < end and text[begin].isspace():
        begin += 1
    while end > begin and text[end - 1].isspace():
        end -= 1
    return begin, end


def _link_cross_references(document: StructuredDocument) -> None:
    numbered = [section for section in document.sections if _is_numbered(section["section"])]
    for span in document.spans:
        references = []
        seen: set[str] = set()
        for match in _STRUCTURAL_CROSS_REF.finditer(span["text"]):
            raw = match.group("section") or match.group("cjk")
            if not raw:
                continue
            target = _canonical_number(raw)
            if target in seen:
                continue
            seen.add(target)
            ref = cross_reference(span, match.group(0), target, numbered)
            if ref is not None:
                references.append(ref)
        span["cross_references"] = references


def sections_from_spans(spans: Sequence[Mapping[str, Any]], document_id: str) -> list[dict[str, Any]]:
    """Numbered sections reconstructed from stored spans, including their span ids."""
    grouped: dict[str, list[str]] = {}
    for span in spans:
        if span.get("document_id") != document_id:
            continue
        label = span.get("section")
        span_id = span.get("source_span_id")
        if not isinstance(label, str) or not isinstance(span_id, str):
            continue
        grouped.setdefault(label, []).append(span_id)
    return [{"section": label, "span_ids": ids} for label, ids in grouped.items() if _is_numbered(label)]


def cross_reference(
    span: Mapping[str, Any],
    evidence: str,
    target_section: str,
    numbered: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Resolve one citation onto numbered sections of the same document. Does not change span ids."""
    target = _canonical_number(target_section)
    if not target or target == span.get("section"):
        return None
    exact = [section for section in numbered if section.get("section") == target]
    targets = []
    if exact:
        targets = [
            span_id
            for section in numbered
            if section.get("section") == target or str(section.get("section")).startswith(f"{target}.")
            for span_id in section.get("span_ids") or []
            if span_id != span.get("source_span_id")
        ]
    return {
        "text": evidence,
        "target_section": target,
        "resolved": bool(targets),
        "target_span_ids": targets[:MAX_CROSS_REFERENCE_TARGETS],
    }


def _is_numbered(label: str) -> bool:
    return _NUMBERED_LABEL.match(label) is not None


def _glossary(document: StructuredDocument) -> list[dict[str, Any]]:
    entries = []
    for span in document.spans:
        found = _definition(span["text"])
        if found is None:
            continue
        term, definition = found
        entries.append(
            {
                "term": term,
                "key": term_key(term),
                "definition": definition,
                "document_id": span["document_id"],
                "source_span_id": span["source_span_id"],
            }
        )
    return entries


def structural_definition(span: Mapping[str, Any]) -> tuple[str, str] | None:
    """A definition recognized by structure, not by a language-specific verb.

    A table row with two or more cells, or a bold term followed by a colon or dash,
    returns ``(term, span text)``. The span text is what evidence grounding searches.
    """
    text = span.get("text")
    if not isinstance(text, str):
        return None
    stripped = text.strip()
    if span.get("kind") == "table_row":
        cells = [cell.strip() for cell in stripped.split("|") if cell.strip()]
        if len(cells) >= 2 and 0 < len(cells[0].split()) <= _MAX_TERM_WORDS:
            return cells[0], stripped
    match = _BOLD_DEFINITION.match(stripped)
    if match is None:
        return None
    term = match.group(1).strip()
    if term and len(term.split()) <= _MAX_TERM_WORDS and match.group(2).strip():
        return term, stripped
    return None


def _definition(text: str) -> tuple[str, str] | None:
    """A definition recognized by punctuation, not by a verb in any language."""
    stripped = text.strip()
    for pattern in (_QUOTED_COLON, _BOLD_DEFINITION):
        match = pattern.match(stripped)
        if match is None:
            continue
        term = match.group(1).strip().strip("*").strip()
        definition = match.group(2).strip().rstrip(".;。．").strip()
        if term and definition and len(term.split()) <= _MAX_TERM_WORDS:
            return term, definition
    return None


def _link_glossary_terms(document: StructuredDocument) -> None:
    link_glossary_terms(document.spans, [entry["key"] for entry in document.glossary])


def link_glossary_terms(spans: Sequence[dict[str, Any]], keys: Sequence[str]) -> None:
    """Set ``glossary_terms`` on each span to the defined terms it uses.

    Spaced or Latin terms, including diacritics, match as whole tokens. Unsegmented
    terms match as substrings.
    """
    ordered = list(dict.fromkeys(keys))
    for span in spans:
        folded = term_key(span["text"])
        matched = [key for key in ordered if _term_in_text(key, folded)]
        # Prefer a defined compound over a shorter unsegmented term contained in it.
        span["glossary_terms"] = sorted(
            key
            for key in matched
            if not any(
                key != other and _unsegmented_term(key) and _unsegmented_term(other) and key in other
                for other in matched
            )
        )


def term_key(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def _term_in_text(key: str, folded: str) -> bool:
    if not key:
        return False
    if re.search(r"\s", key) or latin_token(key):
        return re.search(rf"(?<!\w){re.escape(key)}(?!\w)", folded) is not None
    if sum(char.isalnum() for char in key) < 2:
        return False
    return key in folded


def _unsegmented_term(key: str) -> bool:
    return not re.search(r"\s", key) and not latin_token(key)


def _abbreviation(token: str) -> bool:
    """A dotted initialism (`e.g.`, `v.v.`) or a single letter (`A.`) is not a sentence end."""
    if not token:
        return False
    if "." in token and any(char.isalpha() for char in token):
        return True
    return len(token) == 1 and token.isalpha()


def _slug(text: str) -> str:
    """ASCII-fold Latin letters. Keep letters that have no ASCII form, such as CJK."""
    chars = []
    for char in fold_latin(text):
        lowered = char.casefold()
        if lowered.isascii() and lowered.isalnum():
            chars.append(lowered)
        elif not char.isascii() and char.isalnum():
            chars.append(lowered)
        else:
            chars.append("-")
    return re.sub(r"-+", "-", "".join(chars)).strip("-")[:48].strip("-")
