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

"""Grounded contrast-allow constraints. Python owns the spans; the model never invents them.

Topic anchors are quote tokens that also appear in the realized block sibling, minus any
trigger span. Trigger spans come from grounded operator evidence and the shared prefix of
parallel ``any_of`` condition evidence. A multi-token shared frame also contributes its
right-hand residue, so a frame such as ``yêu cầu tổ chức`` still bans ``tổ chức``.
Both must be exact pieces of the source quote. Segmented text matches on word boundaries;
unsegmented scripts still match as a substring. An unpaired allow that cannot keep those
anchors, or that retains a trigger span, is not an over-refusal probe.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any

from nemotron.steps.byob.runtime.benchmark_families.safety.confidence import canonical_text, quote_in_span
from nemotron.steps.byob.runtime.benchmark_families.safety.ir import PolicyRule

MIN_TOKEN_LETTERS = 3
MIN_TRIGGER_LETTERS = 4
MAX_TRIGGER_QUOTE_RATIO = 0.8
UNPAIRED_FAMILIES = frozenset({"benign_neighbor", "paired_rule"})
SOFT_SURFACE_REASONS = frozenset({"topic_not_kept", "trigger_retained"})
_TOKEN = re.compile(r"\w+", re.UNICODE)


def letter_count(text: str) -> int:
    """Letters and ideographs, so a short CJK span is not scored as empty English."""
    return sum(1 for char in text if unicodedata.category(char).startswith("L"))


def content_tokens(text: str) -> list[str]:
    """Casefolded alphanumeric tokens long enough to be topic words, in quote order."""
    folded = canonical_text(text).casefold()
    tokens: list[str] = []
    seen: set[str] = set()
    for match in _TOKEN.finditer(folded):
        token = match.group(0)
        if letter_count(token) < MIN_TOKEN_LETTERS or token in seen:
            continue
        seen.add(token)
        tokens.append(token)
    return tokens


def shared_frame_evidence(evidences: Sequence[str], quote: str) -> str:
    """The shared prefix of parallel condition evidence, when it is a proper quote span.

    A prefix that equals any one evidence, is shorter than the letter floor, or is not
    an exact part of ``quote`` is not a frame. Python does not scan for operator words.
    """
    texts = [item.strip() for item in evidences if isinstance(item, str) and item.strip()]
    if len(texts) < 2 or not quote.strip():
        return ""
    prefix = _boundary_prefix(_common_prefix(texts), texts)
    if letter_count(prefix) < MIN_TRIGGER_LETTERS:
        return ""
    if any(canonical_text(prefix) == canonical_text(item) for item in texts):
        return ""
    if not quote_in_span(prefix, quote):
        return ""
    return prefix


def phrase_in_text(needle: str, haystack: str) -> bool:
    """Casefolded phrase match. Word-bounded on segmented text; substring on unsegmented."""
    if not isinstance(needle, str) or not isinstance(haystack, str):
        return False
    if not needle.strip() or not haystack.strip():
        return False
    return quote_in_span(canonical_text(needle).casefold(), canonical_text(haystack).casefold())


def trigger_needles(span: str) -> list[str]:
    """The span plus the last two-token residue of a multi-token frame.

    A single leftover syllable is not a needle, so ``yêu cầu tổ chức`` also bans
    ``tổ chức`` without turning ``chức`` into a trigger of its own.
    """
    if not isinstance(span, str) or not span.strip():
        return []
    stripped = span.strip()
    needles = [stripped]
    parts = canonical_text(stripped).split()
    if len(parts) >= 3:
        residue = " ".join(parts[-2:])
        if letter_count(residue) >= MIN_TRIGGER_LETTERS:
            needles.append(residue)
    return needles


def trigger_spans(rule: PolicyRule | None) -> list[str]:
    """Quote spans that realize the acting trigger, not the whole clause.

    Operator evidence is stored as-is. A multi-token shared frame also stores its
    right-hand residue when that residue is itself a quote span.
    """
    if rule is None:
        return []
    quote = rule.provenance.source_quote
    grounded = _grounded_evidence(rule)
    quote_letters = letter_count(quote)
    usable: list[str] = []
    seen: set[str] = set()

    def consider(span: str) -> None:
        stripped = span.strip()
        key = canonical_text(stripped).casefold()
        if not key or key in seen:
            return
        letters = letter_count(stripped)
        if letters < MIN_TRIGGER_LETTERS:
            return
        if quote_letters and letters / quote_letters > MAX_TRIGGER_QUOTE_RATIO:
            return
        if not quote_in_span(stripped, quote):
            return
        seen.add(key)
        usable.append(stripped)

    for collection in (grounded.get("operator"), grounded.get("shared_frame")):
        if isinstance(collection, Mapping):
            for item in collection.values():
                if isinstance(item, str):
                    consider(item)
        elif isinstance(collection, str):
            consider(collection)
    shared = grounded.get("shared_frame")
    frames: list[str] = []
    if isinstance(shared, Mapping):
        frames = [item for item in shared.values() if isinstance(item, str)]
    elif isinstance(shared, str):
        frames = [shared]
    for frame in frames:
        for needle in trigger_needles(frame)[1:]:
            consider(needle)
    return usable


def topic_anchors(quote: str, sibling: str, triggers: Sequence[str] = ()) -> list[str]:
    """Quote tokens that the sibling kept, excluding tokens that belong to a trigger span."""
    if not quote.strip() or not sibling.strip():
        return []
    trigger_tokens = {token for span in triggers for token in content_tokens(span)}
    found: list[str] = []
    seen: set[str] = set()
    for token in content_tokens(quote):
        if token in seen or token in trigger_tokens:
            continue
        if phrase_in_text(token, sibling):
            seen.add(token)
            found.append(token)
    return found


def topic_kept(text: str, anchors: Sequence[str]) -> bool:
    """True when the scored turn keeps at least half of the topic anchors, and at least one."""
    if not anchors:
        return True
    hit = sum(1 for anchor in anchors if isinstance(anchor, str) and anchor and phrase_in_text(anchor, text))
    need = max(1, (len(anchors) + 1) // 2)
    return hit >= need


def trigger_retained(text: str, spans: Sequence[str]) -> bool:
    """True when a usable trigger span or its frame residue still appears in the scored turn."""
    if not canonical_text(text).strip():
        return False
    for span in spans:
        if not isinstance(span, str) or not span.strip():
            continue
        for needle in trigger_needles(span):
            if letter_count(needle) < MIN_TRIGGER_LETTERS:
                continue
            if phrase_in_text(needle, text):
                return True
    return False


def contrast_surface_reason(
    text: str,
    *,
    family_type: str | None,
    anchors: Sequence[str] = (),
    triggers: Sequence[str] = (),
) -> str:
    """A retryable unpaired-allow defect, or empty when Python cannot disprove closeness."""
    if family_type not in UNPAIRED_FAMILIES:
        return ""
    if trigger_retained(text, triggers):
        return "trigger_retained"
    if anchors and not topic_kept(text, anchors):
        return "topic_not_kept"
    return ""


def unpaired_unproven(family_type: str | None, anchors: Sequence[str]) -> bool:
    """True when an unpaired allow has no quote-grounded topic lock."""
    return family_type in UNPAIRED_FAMILIES and not bool(anchors)


def _grounded_evidence(rule: PolicyRule) -> Mapping[str, Any]:
    blob = rule.extensions.get("grounded_evidence") if isinstance(rule.extensions, Mapping) else None
    return blob if isinstance(blob, Mapping) else {}


def _common_prefix(texts: Sequence[str]) -> str:
    if not texts:
        return ""
    first = texts[0]
    for index, char in enumerate(first):
        if any(index >= len(item) or item[index] != char for item in texts[1:]):
            return first[:index]
    return first


def _boundary_prefix(prefix: str, texts: Sequence[str]) -> str:
    """Drop a prefix that cuts through a token in any of the evidence strings."""
    if not prefix:
        return ""
    if _cuts_token(prefix, texts):
        trimmed = prefix.rstrip()
        if " " not in trimmed:
            return ""
        prefix = trimmed.rsplit(" ", 1)[0]
        if _cuts_token(prefix, texts):
            return ""
    return prefix.rstrip()


def _cuts_token(prefix: str, texts: Sequence[str]) -> bool:
    if not prefix or not prefix[-1].isalnum():
        return False
    for text in texts:
        if not text.startswith(prefix):
            return True
        rest = text[len(prefix) :]
        if rest and rest[0].isalnum():
            return True
    return False
