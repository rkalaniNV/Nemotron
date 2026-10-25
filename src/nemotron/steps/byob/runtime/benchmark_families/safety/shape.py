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

"""Quote-punctuation shape checks. Python owns them; the model does not.

A slash list in one grammatical slot is a valid ``any_of``. Combining two slots
(verbs with objects, or two quoted slash-groups joined by another operator) into
one flat ``any_of`` or ``all_of`` of every token is not. The check uses slash
characters, Unicode comma/semicolon names, and short-span cardinality, not
source-language keyword lists.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any

from nemotron.steps.byob.runtime.benchmark_families.safety.confidence import canonical_text
from nemotron.steps.byob.runtime.benchmark_families.safety.contrast import letter_count
from nemotron.steps.byob.runtime.benchmark_families.safety.ir import PolicyRule

SLASH_CHARS = "/\u2044\u2215\uff0f"
MAX_ALT_WORDS = 4
MAX_SHORT_WORDS = 3
MAX_SHORT_LETTERS = 24
HIGH_CARDINALITY = 6
CROSS_SLOT = "flattens slash alternatives from more than one slot into one operator"
SLASH_WITH_RESIDUE = "flattens slash alternatives together with other short tokens into one operator"
TOKENIZED_LIST = "tokenizes the clause into short alternatives with no shared frame"
MISSING_EVIDENCE = "stores a multi-leg operator without grounded condition evidence"

_SLASH = re.compile(f"[{re.escape(SLASH_CHARS)}]")
_URL = re.compile(r"https?://[^\s]+|www\.[^\s]+", re.IGNORECASE)
# Hyphen and apostrophe stay inside one alternative (red-team, don't).
_TOKEN = re.compile(r"\w+(?:[\u002d\u2010\u2011'’]\w+)*", re.UNICODE)
_SLOT_QUOTES = "\"'\u201c\u201d\u2018\u2019\u00ab\u00bb\u300c\u300d\u300e\u300f"
# Same-length non-space filler so a masked URL is a walk barrier, not a glue gap.
_URL_MASK = "\ufffc"


def slash_alternative_groups(quote: str) -> tuple[tuple[str, ...], ...]:
    """Slash-joined alternative lists in ``quote``, excluding URL paths.

    Consecutive slashes with only a short alternative between them are one group.
    Unquoted first and last alternatives are capped at the longest interior
    alternative so a surrounding clause is not absorbed. A quote-delimited first
    alternative keeps the whole slot. Groups shorter than two alternatives are
    dropped.
    """
    if not isinstance(quote, str) or not quote.strip():
        return ()
    groups: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()
    for run in _slash_runs(_mask_urls(quote)):
        if len(run) < 2:
            continue
        key = tuple(canonical_text(part).casefold() for part in run)
        if key in seen:
            continue
        seen.add(key)
        groups.append(run)
    return tuple(groups)


def flattened_condition_reason(rule: PolicyRule) -> str:
    """Why a stored operator flattened punctuation slots, or empty when the shape is usable.

    A ``shared_frame`` covers only legs whose evidence is that frame or a residue
    after it. Leftover short tokens still flatten. Evidence that itself contains a
    slash is already a slot-level bundle. Membership is slash-adjacent prefixes
    and suffixes, not whole-clause splits. Missing condition evidence on a
    multi-leg group is review, not a pass.
    """
    quote = rule.provenance.source_quote
    grounded = rule.extensions.get("grounded_evidence") if isinstance(rule.extensions, Mapping) else None
    blob = grounded if isinstance(grounded, Mapping) else {}
    evidence = blob.get("conditions") if isinstance(blob.get("conditions"), Mapping) else {}
    frames = blob.get("shared_frame") if isinstance(blob.get("shared_frame"), Mapping) else {}
    adjacent = _slash_adjacent_alts(_mask_urls(quote))
    reasons: list[str] = []
    for name, group in rule.rule.conditions().items():
        if group.operator not in {"any_of", "all_of"} or len(group.legs) < 2:
            continue
        frame = frames.get(name)
        frame_text = frame.strip() if isinstance(frame, str) else ""
        detail = _flatten_group(group.legs, evidence, adjacent, frame_text)
        if detail:
            reasons.append(f"{name} {detail}")
    return "; ".join(reasons)


def _flatten_group(
    legs: Sequence[str],
    evidence: Mapping[str, Any],
    adjacent: Sequence[tuple[int, str]],
    frame: str = "",
) -> str:
    matched: set[int] = set()
    unmatched_short = 0
    short_count = 0
    slot_level = 0
    missing = 0
    for leg in legs:
        text = _leg_evidence(leg, evidence)
        if not text:
            missing += 1
            unmatched_short += 1
            continue
        if _contains_slash(text):
            slot_level += 1
            continue
        if frame and _frame_residue(text, frame):
            continue
        hits = [run_id for run_id, alt in adjacent if _same_span(text, alt)]
        if hits:
            matched.update(hits)
        elif _short_token(text):
            unmatched_short += 1
        if _short_token(text):
            short_count += 1
    if missing and not matched:
        return MISSING_EVIDENCE
    if len(matched) >= 2:
        return CROSS_SLOT
    if matched and unmatched_short:
        return SLASH_WITH_RESIDUE
    if short_count >= HIGH_CARDINALITY and short_count == len(legs) and slot_level == 0 and not matched:
        return TOKENIZED_LIST
    return ""


def _frame_residue(text: str, frame: str) -> bool:
    """True when evidence is the shared frame or continues after it as a residue."""
    needle = canonical_text(frame).casefold()
    hay = canonical_text(text).casefold()
    if not needle or not hay:
        return False
    if hay == needle:
        return True
    if not hay.startswith(needle):
        return False
    rest = hay[len(needle) :]
    return not rest or not rest[0].isalnum()


def _mask_urls(quote: str) -> str:
    return _URL.sub(lambda match: _URL_MASK * len(match.group(0)), quote)


def _slash_clusters(text: str) -> list[list[int]]:
    positions = [match.start() for match in _SLASH.finditer(text)]
    if not positions:
        return []
    clusters: list[list[int]] = [[positions[0]]]
    for pos in positions[1:]:
        if _connected(text, clusters[-1][-1], pos):
            clusters[-1].append(pos)
        else:
            clusters.append([pos])
    return clusters


def _slash_runs(text: str) -> list[tuple[str, ...]]:
    return [run for cluster in _slash_clusters(text) if (run := _run_alternatives(text, cluster))]


def _connected(text: str, left: int, right: int) -> bool:
    gap = text[left + 1 : right]
    if _gap_breaks_run(gap):
        return False
    return len(_TOKEN.findall(gap)) <= MAX_ALT_WORDS


def _gap_breaks_run(gap: str) -> bool:
    """True when the gap contains a slot joiner, a quote, or a Unicode comma/semicolon."""
    for char in gap:
        if char in _SLOT_QUOTES or char == "+":
            return True
        if _comma_or_semicolon(char):
            return True
    return False


def _comma_or_semicolon(char: str) -> bool:
    try:
        name = unicodedata.name(char)
    except ValueError:
        return False
    return "COMMA" in name or "SEMICOLON" in name


def _run_alternatives(text: str, slashes: Sequence[int]) -> tuple[str, ...]:
    betweens: list[str] = []
    for index, pos in enumerate(slashes[:-1]):
        between = _TOKEN.findall(text[pos + 1 : slashes[index + 1]])
        if not between or len(between) > MAX_ALT_WORDS:
            return ()
        betweens.append(" ".join(between))
    left_walk = _left_token_walk(text, slashes[0])
    if not left_walk:
        return ()
    first_left = [token for token, _start, _end in reversed(left_walk)]
    if not _preceded_by_quote(text, left_walk[-1][1]):
        first_left = first_left[-max((len(part.split()) for part in betweens), default=1) :]
    parts = [" ".join(first_left), *betweens]
    cap = max(len(part.split()) for part in parts)
    last_right = _right_words(text, slashes[-1])[:cap]
    if not last_right:
        return ()
    parts.append(" ".join(last_right))
    if len(parts) < 2:
        return ()
    return tuple(parts)


def _preceded_by_quote(text: str, index: int) -> bool:
    cursor = index - 1
    while cursor >= 0 and text[cursor].isspace():
        cursor -= 1
    return cursor >= 0 and text[cursor] in _SLOT_QUOTES


def _slash_adjacent_alts(text: str) -> list[tuple[int, str]]:
    """``(run_id, span)`` for every 1..N word prefix/suffix touching a slash."""
    found: list[tuple[int, str]] = []
    seen: set[tuple[int, str]] = set()
    for run_id, cluster in enumerate(_slash_clusters(text)):
        for pos in cluster:
            for span in (*_left_spans(text, pos), *_right_spans(text, pos)):
                key = (run_id, canonical_text(span).casefold())
                if not key[1] or key in seen:
                    continue
                seen.add(key)
                found.append((run_id, span))
    return found


def _right_words(text: str, pos: int) -> list[str]:
    return [token for token, _start, _end in _right_token_walk(text, pos)]


def _left_spans(text: str, pos: int) -> list[str]:
    walked = _left_token_walk(text, pos)
    return [" ".join(token for token, _s, _e in reversed(walked[: index + 1])) for index in range(len(walked))]


def _right_spans(text: str, pos: int) -> list[str]:
    walked = _right_token_walk(text, pos)
    return [" ".join(token for token, _s, _e in walked[: index + 1]) for index in range(len(walked))]


def _left_token_walk(text: str, pos: int) -> list[tuple[str, int, int]]:
    picked: list[tuple[str, int, int]] = []
    edge = pos
    for token, start, end in reversed(_tokens(text)):
        if end > pos:
            continue
        if not _space_gap(text, end, edge):
            break
        picked.append((token, start, end))
        edge = start
        if len(picked) >= MAX_ALT_WORDS:
            break
    return picked


def _right_token_walk(text: str, pos: int) -> list[tuple[str, int, int]]:
    picked: list[tuple[str, int, int]] = []
    edge = pos + 1
    for token, start, end in _tokens(text):
        if start < pos:
            continue
        if not _space_gap(text, edge, start):
            break
        picked.append((token, start, end))
        edge = end
        if len(picked) >= MAX_ALT_WORDS:
            break
    return picked


def _tokens(text: str) -> list[tuple[str, int, int]]:
    return [(match.group(0), match.start(), match.end()) for match in _TOKEN.finditer(text)]


def _space_gap(text: str, start: int, end: int) -> bool:
    if start > end:
        return False
    if start == end:
        return True
    return text[start:end].isspace()


def _leg_evidence(leg: str, evidence: Mapping[str, Any]) -> str:
    value = evidence.get(leg) if isinstance(leg, str) else None
    return value.strip() if isinstance(value, str) else ""


def _contains_slash(text: str) -> bool:
    return any(char in text for char in SLASH_CHARS)


def _short_token(text: str) -> bool:
    if _contains_slash(text) or letter_count(text) > MAX_SHORT_LETTERS:
        return False
    return len(canonical_text(text).split()) <= MAX_SHORT_WORDS


def _same_span(left: str, right: str) -> bool:
    return canonical_text(left).casefold() == canonical_text(right).casefold()
