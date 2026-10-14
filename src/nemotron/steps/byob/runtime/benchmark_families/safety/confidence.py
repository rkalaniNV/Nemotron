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
"""

import math
import re
import unicodedata
from dataclasses import dataclass, replace

from nemotron.steps.byob.runtime.benchmark_families.safety.ir import PolicyRule

CONFIDENCE_WEIGHTS = {
    "quote_match": 0.25,
    "schema_completeness": 0.20,
    "pass_agreement": 0.20,
    "definition_resolution": 0.15,
    "cross_reference_resolution": 0.10,
    "low_ambiguity": 0.10,
}
ELIGIBLE_THRESHOLD = 0.90
DECISION_THRESHOLD = 0.70
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class ConfidenceSignals:
    """Observable checks, each a score from 0 to 1.

    ``pass_agreement`` is None when only one extraction pass ran, and
    ``cross_reference_resolution`` is None when the passage has no cross-reference.
    Both then count as 1.0, because there is nothing to disagree with or resolve.
    """

    quote_match: float
    schema_completeness: float
    definition_resolution: float
    low_ambiguity: float
    pass_agreement: float | None = None
    cross_reference_resolution: float | None = None

    def __post_init__(self) -> None:
        for name in CONFIDENCE_WEIGHTS:
            value = getattr(self, name)
            if value is None and name in {"pass_agreement", "cross_reference_resolution"}:
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
    """Weighted sum with the locked weights, rounded to four decimals."""
    total = 0.0
    for name, weight in CONFIDENCE_WEIGHTS.items():
        value = getattr(signals, name)
        total += weight * (1.0 if value is None else float(value))
    return round(total, 4)


def quote_in_span(quote: str, span_text: str) -> bool:
    """True when the quote appears verbatim in the span.

    Unicode NFC, curly quotes, zero-width characters, and runs of whitespace are
    normalized. The quote must fall on word boundaries, so a short quote does not match
    inside a longer word. One changed word, letter, or diacritic is still a mismatch.
    """
    if not isinstance(quote, str) or not isinstance(span_text, str):
        raise TypeError("quote and span must be strings")
    needle = canonical_text(quote)
    if not needle:
        return False
    return re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", canonical_text(span_text)) is not None


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
      stored as default ``any_of`` is not a reason.
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
