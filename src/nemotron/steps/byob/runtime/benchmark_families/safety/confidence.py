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

"""Extraction confidence and the routing it drives.

Python computes the one published ``confidence`` from checks it can observe. A model
self-score is never an input. ``confidence`` estimates whether a rule was read correctly
from its source passage; it is not a measure of legal correctness or model accuracy.

``operator_explicitness`` reads ``operator_source`` already stored on the rule. Whether a
condition quote appears in the source is a separate pass-or-fail check, not a weighted
signal. Section labels are not scored.
"""

import math
import re
import unicodedata
from dataclasses import dataclass, replace

from nemotron.steps.byob.runtime.benchmark_families.safety.ir import ConditionGroup, PolicyRule

CONFIDENCE_WEIGHTS = {
    "quote_match": 0.36,
    "schema_completeness": 0.28,
    "pass_agreement": 0.28,
    "operator_explicitness": 0.08,
}
# A multi-leg ``default_any_of`` scores 0 on ``operator_explicitness``. Weight 0.08 keeps
# that rule at 0.92 when every other signal is 1, so the guessed operator is still not,
# by itself, a reason for review.
ELIGIBLE_THRESHOLD = 0.90
DECISION_THRESHOLD = 0.70
MIN_LATIN_SNIPPET_LETTERS = 6
MIN_UNSEGMENTED_SNIPPET_LETTERS = 2
_WHITESPACE = re.compile(r"\s+")
# đ, ø, ł and similar letters do not decompose under NFKD. The Unicode name still names the base.
_LATIN_BASE = re.compile(r"^LATIN (SMALL|CAPITAL) LETTER ([A-Z])(?: WITH .+)?$")


@dataclass(frozen=True)
class ConfidenceSignals:
    """Observable checks, each a score from 0 to 1.

    ``pass_agreement`` is None when only one extraction pass ran, and then counts as 1.
    """

    quote_match: float
    schema_completeness: float
    operator_explicitness: float
    pass_agreement: float | None = None

    def __post_init__(self) -> None:
        for name in CONFIDENCE_WEIGHTS:
            value = getattr(self, name)
            if value is None and name == "pass_agreement":
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a number from 0 to 1")
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be a number from 0 to 1; received {value}")


@dataclass(frozen=True)
class Thresholds:
    eligible: float = ELIGIBLE_THRESHOLD
    decision: float = DECISION_THRESHOLD

    def __post_init__(self) -> None:
        if not 0 <= self.decision <= self.eligible <= 1:
            raise ValueError("thresholds must satisfy 0 <= decision <= eligible <= 1")


@dataclass(frozen=True)
class RoutingResult:
    rule: PolicyRule
    reasons: tuple[str, ...]


def compute_confidence(signals: ConfidenceSignals) -> float:
    """Weighted sum with the locked weights, rounded to four decimals.

    A missing ``pass_agreement`` counts as 1.
    """
    total = 0.0
    for name, weight in CONFIDENCE_WEIGHTS.items():
        value = getattr(signals, name)
        total += weight * (1.0 if value is None else float(value))
    return round(total, 4)


_EXPLICIT_OPERATORS = frozenset({"explicit", "model_evidence", "single_condition"})


def operator_explicitness(rule: PolicyRule) -> float:
    """Share of condition groups whose operator was stated or did not need one.

    ``explicit``, grounded ``model_evidence``, and ``single_condition`` score 1. A one-leg
    ``default_any_of`` also scores 1: a single positive condition has the same meaning under
    either positive operator. A multi-leg ``default_any_of`` scores 0 because the operator
    was guessed. An unresolved group scores 0. The scorer reads the stored operator and does
    not look for words such as "đồng thời" or "may".
    """
    groups = tuple(rule.rule.conditions().values())
    total = len(groups) + len(rule.unresolved_conditions)
    if total == 0:
        return 1.0
    stated = sum(_operator_is_explicit(group) for group in groups)
    return round(stated / total, 4)


def _operator_is_explicit(group: ConditionGroup) -> bool:
    if group.operator_source in _EXPLICIT_OPERATORS:
        return True
    return group.operator_source == "default_any_of" and len(group.legs) == 1


def quote_in_span(quote: str, span_text: str) -> bool:
    """True when the quote appears verbatim in the span.

    Unicode NFC, curly quotes, zero-width characters, and runs of whitespace are
    normalized. Latin text, including diacritics, must fall on word boundaries, so a
    short quote does not match inside a longer word. A quote of at least two letters
    in a script that does not separate words may match as a substring. One changed
    word, letter, or diacritic is still a mismatch.
    """
    if not isinstance(quote, str) or not isinstance(span_text, str):
        raise TypeError("quote and span must be strings")
    needle = canonical_text(quote)
    if not needle:
        return False
    haystack = canonical_text(span_text)
    if re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", haystack) is not None:
        return True
    if sum(char.isalnum() for char in needle) < 2 or not unsegmented_text(needle):
        return False
    return needle in haystack


def latin_token(key: str) -> bool:
    """True when every letter is Latin, so the text uses spaces or word boundaries."""
    if not key:
        return False
    for char in key:
        if char.isdigit() or char in "'_-":
            continue
        if not _latin_letter(char):
            return False
    return True


def unsegmented_text(text: str) -> bool:
    """True when the text has letters that do not use spaces as word boundaries."""
    return bool(text) and not any(char.isspace() for char in text) and not latin_token(text)


def snippet_letter_floor(text: str) -> int:
    """Minimum letters for a LanguageCheck excerpt of this text."""
    needle = canonical_text(text) if text else ""
    if unsegmented_text(needle):
        return MIN_UNSEGMENTED_SNIPPET_LETTERS
    return MIN_LATIN_SNIPPET_LETTERS


def fold_latin(text: str) -> str:
    """ASCII-fold Latin letters. Combining marks drop; strokes such as đ and ø become d and o."""
    chars: list[str] = []
    for char in unicodedata.normalize("NFKD", text):
        if unicodedata.combining(char):
            continue
        chars.append(_latin_base(char))
    return "".join(chars)


def _latin_letter(char: str) -> bool:
    if not unicodedata.category(char).startswith("L"):
        return False
    try:
        return unicodedata.name(char).startswith("LATIN")
    except ValueError:
        return False


def _latin_base(char: str) -> str:
    try:
        name = unicodedata.name(char)
    except ValueError:
        return char
    match = _LATIN_BASE.fullmatch(name)
    if match is None:
        return char
    letter = match.group(2)
    return letter.lower() if match.group(1) == "SMALL" else letter


def route_rule(
    rule: PolicyRule,
    signals: ConfidenceSignals,
    *,
    quote_found: bool,
    has_conflict: bool = False,
    thresholds: Thresholds | None = None,
    decision_reasons: tuple[str, ...] = (),
    quarantine_reasons: tuple[str, ...] = (),
) -> RoutingResult:
    """Store the computed confidence and route the rule.

    - Quarantined: the quote is not in its span, confidence is below the decision
      threshold, the rule is not testable, or the caller passes a quarantine reason.
    - needs_decision: confidence is between the thresholds, or an operator is
      unresolved, a field is unsupported, a conflict exists, the rule is only
      partially testable, or the caller passes a decision reason. A ``block_if``
      stored as default ``any_of`` is not a reason. A caller may also pass a
      flatten reason when a multi-leg operator mixes slash-list slots.
    - eligible_for_generation: everything else.

    A rejected rule stays rejected; routing does not revive a reviewer's decision.
    """
    if not isinstance(quote_found, bool) or not isinstance(has_conflict, bool):
        raise ValueError("quote_found and has_conflict must be true or false")
    if not all(isinstance(reason, str) for reason in (*decision_reasons, *quarantine_reasons)):
        raise ValueError("routing reasons must be text")
    thresholds = thresholds or Thresholds()
    # ``quote_found`` is the deterministic evidence check and therefore the only
    # authority for the exact-match component. Do not let a stale/model score disagree.
    signals = replace(signals, quote_match=1.0 if quote_found else 0.0)
    confidence = compute_confidence(signals)
    quarantine: list[str] = list(quarantine_reasons)
    decision: list[str] = list(decision_reasons)
    if not quote_found:
        quarantine.append("source quote is not in the cited span")
    if confidence < thresholds.decision:
        quarantine.append(f"confidence {confidence} is below {thresholds.decision}")
    if rule.testability in {"needs_clarification", "non_testable", "rejected"}:
        quarantine.append(f"testability is {rule.testability}")
    if thresholds.decision <= confidence < thresholds.eligible:
        decision.append(f"confidence {confidence} is below {thresholds.eligible}")
    if rule.unresolved_conditions:
        decision.append("operator not stated for " + ", ".join(sorted(rule.unresolved_conditions)))
    if rule.unsupported_fields:
        decision.append("unsupported inferred fields: " + ", ".join(rule.unsupported_fields))
    if has_conflict:
        decision.append("conflicts with another rule")
    if rule.testability == "partially_testable":
        decision.append("rule is only partially testable")
    if rule.testability is None:
        decision.append("testability has not been assessed")

    if rule.status == "rejected":
        status, reasons = "rejected", ["rejected by a reviewer"]
    elif quarantine:
        status, reasons = "quarantined", quarantine + decision
    elif decision:
        status, reasons = "needs_decision", decision
    else:
        status, reasons = "eligible_for_generation", []
    routed = PolicyRule.model_validate({**rule.model_dump(), "confidence": confidence, "status": status})
    return RoutingResult(routed, tuple(reasons))


def route_rule_from_span(
    rule: PolicyRule,
    signals: ConfidenceSignals,
    span_text: str,
    *,
    has_conflict: bool = False,
    thresholds: Thresholds | None = None,
) -> RoutingResult:
    """Verify the rule quote against its span and route it in one operation."""
    found = quote_in_span(rule.provenance.source_quote, span_text)
    return route_rule(
        rule,
        signals,
        quote_found=found,
        has_conflict=has_conflict,
        thresholds=thresholds,
    )


_INVISIBLE = str.maketrans("", "", "\u200b\u200c\u200d\ufeff\u2060")
_QUOTES = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u00ab": '"', "\u00bb": '"'})


def canonical_text(text: str) -> str:
    """NFC, straight quotes, no zero-width characters, single spaces."""
    text = unicodedata.normalize("NFC", text).translate(_QUOTES).translate(_INVISIBLE)
    return _WHITESPACE.sub(" ", text).strip()
