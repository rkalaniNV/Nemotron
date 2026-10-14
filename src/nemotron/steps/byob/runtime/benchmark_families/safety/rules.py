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

"""From evidence windows and pre-authored IR to routed Policy Rule IR.

Stages 3 to 5 of the safety family: candidate extraction, normalization and routing, and
conflict checks. The model only proposes. Python verifies every quote and condition against
the redacted source, stores operators, assigns stable IDs and rubric severity, computes the
one published confidence, and decides which rules may generate cases.
"""

import dataclasses
import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from nemotron.steps.byob.runtime.benchmark_families.safety import extract as extraction
from nemotron.steps.byob.runtime.benchmark_families.safety.artifacts import write_json, write_jsonl, write_yaml
from nemotron.steps.byob.runtime.benchmark_families.safety.confidence import (
    CONFIDENCE_WEIGHTS,
    DECISION_THRESHOLD,
    ELIGIBLE_THRESHOLD,
    ConfidenceSignals,
    canonical_text,
    compute_confidence,
    quote_in_span,
    route_rule,
)
from nemotron.steps.byob.runtime.benchmark_families.safety.config import ResolvedConfig
from nemotron.steps.byob.runtime.benchmark_families.safety.conflicts import detect_conflicts
from nemotron.steps.byob.runtime.benchmark_families.safety.coverage import COVERAGE_OUTPUTS
from nemotron.steps.byob.runtime.benchmark_families.safety.extract import (
    ExtractedCandidate,
    ExtractionError,
    PassOutcome,
)
from nemotron.steps.byob.runtime.benchmark_families.safety.ingest import GLOSSARY_FILE, INDEX_FILE
from nemotron.steps.byob.runtime.benchmark_families.safety.ingest import RULES_FILE as PREAUTHORED_FILE
from nemotron.steps.byob.runtime.benchmark_families.safety.ir import (
    CONDITION_CLASSES,
    PolicyRule,
    RuleValidationError,
    detect_operator,
    normalize_candidate,
    validate_rule,
)
from nemotron.steps.byob.runtime.benchmark_families.safety.status import RunStatus, abort
from nemotron.steps.byob.runtime.benchmark_families.safety.windows import EvidenceWindow, build_windows

EXTRACTED = "extracted"
PRE_AUTHORED = "pre_authored"
CANDIDATES_FILE = "candidate_policy_rules.jsonl"
UNCERTAINTIES_FILE = "extraction_uncertainties.jsonl"
POLICY_RULES_FILE = "policy_rules.jsonl"
ELIGIBLE_FILE = "policy_rules_eligible.yaml"
FLAGS_FILE = "policy_flags.jsonl"
ROUTING_LOG_FILE = "policy_routing_log.json"
GRAPH_FILE = "policy_graph.json"
CONFLICT_FILE = "conflict_report.json"
RULE_OUTPUTS = (
    CANDIDATES_FILE,
    UNCERTAINTIES_FILE,
    POLICY_RULES_FILE,
    ELIGIBLE_FILE,
    FLAGS_FILE,
    ROUTING_LOG_FILE,
    GRAPH_FILE,
    CONFLICT_FILE,
)
SECOND_PASS_PROFILES = frozenset({"regulated", "high_assurance"})
ALLOW_SUFFIX = "-ALLOW"
CRITICAL_CATEGORIES = frozenset({"fraud", "personal_harm", "legal_prohibition", "irreversible_tool_action"})
_BINDING = (
    "must",
    "shall",
    "required",
    "prohibited",
    "never",
    "phải",
    "bắt buộc",
    "nghiêm cấm",
    "tuyệt đối không",
    "không được phép",
)
_CONDITIONAL = (
    "only when",
    "only if",
    "unless",
    "except when",
    "except if",
    "chỉ khi",
    "chỉ chặn khi",
    "trừ khi",
    "ngoại trừ",
    "trừ trường hợp",
)
_HEDGES = (
    "may",
    "might",
    "where appropriate",
    "as appropriate",
    "if possible",
    "as needed",
    "có thể",
    "should",
    "nên",
    "reasonable",
    "tùy trường hợp",
    "khi cần thiết",
    "nếu phù hợp",
)
_VIOLATION_ACTIONS = frozenset({"block", "caution", "rewrite", "redact"})


@dataclass
class Draft:
    """One rule before routing, with the evidence routing needs."""

    rule: PolicyRule
    origin: str
    quote_found: bool
    signals: ConfidenceSignals
    expected_action: str
    flags: list[dict[str, Any]] = field(default_factory=list)
    decision_reasons: list[str] = field(default_factory=list)
    quarantine_reasons: list[str] = field(default_factory=list)
    glossary_terms: tuple[str, ...] = ()
    window_id: str | None = None
    order: int = 0
    passes: int = 1

    @property
    def quote_key(self) -> str:
        return canonical_text(self.rule.provenance.source_quote).casefold()

    @property
    def shape(self) -> tuple[Any, ...]:
        mapped = _condition_map(self)
        return (self.expected_action, tuple(mapped), tuple(mapped.values()))

    def _legs(self, name: str) -> tuple[str, ...]:
        group = self.rule.rule.conditions().get(name)
        return group.legs if group is not None else tuple(self.rule.unresolved_conditions.get(name, ()))


@dataclass
class RuleStageResult:
    counts: dict[str, int]
    extraction: dict[str, Any]

    @property
    def usable(self) -> bool:
        return self.counts["eligible_for_generation"] + self.counts["needs_decision"] > 0


def run_rule_stages(config: ResolvedConfig, run_dir: Path) -> RuleStageResult:
    """Extract, normalize, check conflicts, route, and write every rule artifact."""
    stage_cache = run_dir / "stage_cache"
    index = _load_json(stage_cache / INDEX_FILE, run_dir)
    glossary = _load_json(stage_cache / GLOSSARY_FILE, run_dir)
    preauthored = _load_jsonl(stage_cache / PREAUTHORED_FILE, run_dir)
    documents, spans = index.get("documents"), index.get("spans")
    terms, glossary_conflicts = glossary.get("terms", []), glossary.get("conflicts", [])
    if not all(isinstance(value, list) for value in (documents, spans, terms, glossary_conflicts)):
        raise abort(
            RunStatus.INVALID_RUN_ARTIFACT, run_dir, "stage_cache ingest files are incomplete; run stage=prepare."
        )
    _check_glossary_conflicts(glossary_conflicts, run_dir)
    # Drop the previous routing result first, so a failed run cannot leave rules from older code.
    for name in (*RULE_OUTPUTS, *COVERAGE_OUTPUTS):
        (stage_cache / name).unlink(missing_ok=True)
    try:
        windows = build_windows(documents, spans, terms)
    except (TypeError, ValueError) as exc:
        raise abort(
            RunStatus.INVALID_RUN_ARTIFACT, run_dir, f"{INDEX_FILE} is damaged: {exc} Run stage=prepare."
        ) from exc
    uncertainties: list[dict[str, Any]] = []
    extracted, log = _extract(windows, config, run_dir, uncertainties)
    spans_by_id = {span["source_span_id"]: span for span in spans}
    drafts = [_preauthored_draft(row, spans_by_id, config, run_dir) for row in preauthored]
    taken = {draft.rule.policy_id for draft in drafts}
    for draft in _assign_ids(extracted, uncertainties):
        if draft.rule.policy_id in taken:
            uncertainties.append(
                _uncertainty(draft.window_id, "duplicate_policy_id", draft.rule.provenance.source_quote)
            )
            continue
        taken.add(draft.rule.policy_id)
        drafts.append(draft)
    rules = {draft.rule.policy_id: draft.rule for draft in drafts}
    conflicts = detect_conflicts(
        list(rules.values()),
        term_usage={draft.rule.policy_id: draft.glossary_terms for draft in drafts},
        glossary_conflicts=glossary_conflicts,
    )
    decision_by_id, quarantine_by_id = conflicts.reasons(rules)
    routed = []
    for draft in sorted(drafts, key=lambda item: item.rule.policy_id):
        policy_id = draft.rule.policy_id
        result = route_rule(
            draft.rule,
            draft.signals,
            quote_found=draft.quote_found,
            decision_reasons=tuple(draft.decision_reasons + decision_by_id.get(policy_id, [])),
            quarantine_reasons=tuple(draft.quarantine_reasons + quarantine_by_id.get(policy_id, [])),
        )
        routed.append((draft, result.rule, list(result.reasons)))
    counts = _counts(routed, len(conflicts.conflicts), len(uncertainties), len(windows))
    _write(stage_cache, routed, extracted, uncertainties, conflicts, glossary_conflicts, log, counts)
    return RuleStageResult(counts, log)


# Extraction


def _extract(
    windows: Sequence[EvidenceWindow], config: ResolvedConfig, run_dir: Path, uncertainties: list[dict[str, Any]]
) -> tuple[list[Draft], dict[str, Any]]:
    log: dict[str, Any] = {"windows": len(windows), "model": None, "passes": []}
    if not windows:
        return [], log
    try:
        extractor = extraction.make_extractor(config, run_dir)
    except ValueError as exc:
        raise abort(RunStatus.INVALID_CONFIG, run_dir, str(exc)) from exc
    if extractor is None:
        raise abort(
            RunStatus.INVALID_CONFIG,
            run_dir,
            f"generation_model_config is required to extract rules from {len(windows)} policy clauses. "
            "Add it, or supply the rules as Policy Rule IR in YAML or JSON.",
        )
    log["model"] = dict(extractor.identity)
    cache = run_dir / "stage_cache"
    first = _run_pass(windows, extractor, cache, 1, run_dir, fatal=True)
    log["passes"].append(_pass_log(1, first))
    by_window: dict[str, list[Draft]] = {}
    order = 0
    for window in windows:
        if window.window_id in first.errors:
            uncertainties.append(_uncertainty(window.window_id, "malformed_response", first.errors[window.window_id]))
            continue
        drafts = _window_drafts(window, first.candidates.get(window.window_id, []), config, uncertainties)
        for draft in drafts:
            draft.order = order
            order += 1
        by_window[window.window_id] = drafts
    second_windows = [
        window
        for window in windows
        if window.window_id in by_window
        and (
            config.assurance["profile"] in SECOND_PASS_PROFILES
            or any(
                draft.quote_found and compute_confidence(draft.signals) < ELIGIBLE_THRESHOLD
                for draft in by_window[window.window_id]
            )
        )
    ]
    if second_windows:
        second = _run_pass(second_windows, extractor, cache, 2, run_dir, fatal=False)
        log["passes"].append(_pass_log(2, second))
        for window in second_windows:
            drafts = by_window[window.window_id]
            if window.window_id in second.errors:
                _unconfirmed(drafts, uncertainties, window.window_id, second.errors[window.window_id])
                continue
            other = _window_drafts(window, second.candidates.get(window.window_id, []), config, [])
            _compare_passes(drafts, other, uncertainties, window.window_id)
    return [draft for window in windows for draft in by_window.get(window.window_id, [])], log


def _run_pass(
    windows: Sequence[EvidenceWindow], extractor: Any, cache: Path, index: int, run_dir: Path, *, fatal: bool
) -> PassOutcome:
    """Run one pass. Only pass one stops the run; a failed second pass sends its rules to review."""
    try:
        outcome = extraction.run_pass(windows, extractor, cache, index)
    except ExtractionError as exc:
        if fatal:
            raise abort(RunStatus.MODEL_UNAVAILABLE, run_dir, str(exc)) from exc
        return PassOutcome(errors={window.window_id: str(exc) for window in windows})
    if fatal and not outcome.candidates and outcome.errors:
        raise abort(RunStatus.MODEL_UNAVAILABLE, run_dir, "Every reply in this extraction pass was malformed.")
    return outcome


def _unconfirmed(drafts: Sequence[Draft], uncertainties: list[dict[str, Any]], window_id: str, detail: str) -> None:
    """A second pass that did not answer. The first-pass rules stay, but they need a person."""
    uncertainties.append(_uncertainty(window_id, "second_pass_failed", detail))
    for draft in drafts:
        draft.signals = dataclasses.replace(draft.signals, pass_agreement=0.0)
        draft.passes = 2
        draft.decision_reasons.append("the second extraction pass did not confirm this rule")


def _pass_log(index: int, outcome: Any) -> dict[str, Any]:
    return {
        "pass": index,
        "windows": len(outcome.candidates) + len(outcome.errors),
        "cached": outcome.cached,
        "requested": outcome.requested,
        "malformed": len(outcome.errors),
        "not_cached": outcome.uncached,
    }


def _compare_passes(
    first: list[Draft], second: list[Draft], uncertainties: list[dict[str, Any]], window_id: str
) -> None:
    """Score pass agreement per rule. Pass two never saw pass one's answer."""
    remaining = list(second)
    for draft in first:
        match = next((other for other in remaining if other.quote_key == draft.quote_key), None)
        if match is not None:
            remaining.remove(match)
        agreement = 0.0 if match is None else _agreement(draft, match)
        draft.signals = dataclasses.replace(draft.signals, pass_agreement=agreement)
        draft.passes = 2
    for other in remaining:
        uncertainties.append(_uncertainty(window_id, "second_pass_only", other.rule.provenance.source_quote))


def _condition_map(draft: Draft) -> dict[str, frozenset[str]]:
    names = set(draft.rule.rule.conditions()) | set(draft.rule.unresolved_conditions)
    return {name: frozenset(draft._legs(name)) for name in sorted(names)}


def _agreement(first: Draft, second: Draft) -> float:
    """Share of action, condition classes, and condition names that the two passes share."""
    left, right = _condition_map(first), _condition_map(second)
    checks = (
        first.expected_action == second.expected_action,
        set(left) == set(right),
        left == right,
    )
    return round(sum(checks) / len(checks), 4)


def _window_drafts(
    window: EvidenceWindow,
    raw_candidates: Sequence[Any],
    config: ResolvedConfig,
    uncertainties: list[dict[str, Any]],
) -> list[Draft]:
    drafts = []
    for raw in raw_candidates:
        try:
            candidate = ExtractedCandidate.model_validate(raw)
        except ValidationError as exc:
            detail = "; ".join(f"{'.'.join(map(str, error['loc']))}: {error['msg']}" for error in exc.errors()[:3])
            uncertainties.append(_uncertainty(window.window_id, "malformed_candidate", detail))
            continue
        draft = _candidate_draft(window, candidate, config, uncertainties)
        if draft is not None:
            drafts.append(draft)
    return drafts


def _candidate_draft(
    window: EvidenceWindow,
    candidate: ExtractedCandidate,
    config: ResolvedConfig,
    uncertainties: list[dict[str, Any]],
) -> Draft | None:
    clause = window.target["text"]
    quote = candidate.source_quote.strip()
    if not quote:
        uncertainties.append(_uncertainty(window.window_id, "no_quote", candidate.title))
        return None
    found = quote_in_span(quote, clause)
    if not found and any(quote_in_span(quote, span["text"]) for span in window.neighbors):
        # The neighbor's own window extracts this rule.
        uncertainties.append(_uncertainty(window.window_id, "quote_in_neighbor", quote))
        return None
    sources = [clause, *(entry["definition"] for entry in window.glossary)]
    flags: list[dict[str, Any]] = []
    unsupported: list[str] = []
    grounded = total = 0
    body: dict[str, Any] = {name: _text(getattr(candidate, name)) for name in ("actor", "action", "object")}
    for name in CONDITION_CLASSES:
        legs, found_legs, all_legs = _legs(
            getattr(candidate.conditions, name), sources, f"rule.{name}", unsupported, flags
        )
        grounded, total = grounded + found_legs, total + all_legs
        if legs:
            body[name] = legs
    exceptions, found_legs, all_legs = _legs(candidate.exceptions, sources, "exceptions", unsupported, flags)
    grounded, total = grounded + found_legs, total + all_legs
    if not any(name in body for name in CONDITION_CLASSES):
        uncertainties.append(_uncertainty(window.window_id, "no_conditions", quote))
        return None
    expected = candidate.expected_action
    produced = {
        "block_if": expected if expected in _VIOLATION_ACTIONS else "block",
        "escalate_if": "escalate",
        "require_confirmation_if": "require_confirmation",
        "allow_if": "allow",
    }
    actions = {produced[name] for name in CONDITION_CLASSES if name in body}
    decision_reasons = []
    if expected not in actions:
        decision_reasons.append(f"expected_action {expected} is not produced by any of its conditions")
    disabled = sorted((actions | {expected}) - set(config.proposals["decision_actions"]))
    if disabled:
        decision_reasons.append(f"{', '.join(disabled)} is not enabled in proposals.decision_actions")
    raw = {
        "policy_id": "PENDING",
        "title": candidate.title.strip() or None,
        "scope": {
            "domains": [config.proposals["domain"]],
            "jurisdictions": [config.proposals["jurisdiction"]],
            "targets": list(config.proposals["moderation_targets"]),
        },
        "rule": body,
        "exceptions": exceptions,
        "decision": {"violation_action": produced["block_if"]},
        "severity": "low",
        "provenance": {
            "document_id": window.document_id,
            "document_sha256": window.document_sha256 or None,
            "section": window.section,
            "source_span_id": window.window_id,
            "source_quote": quote,
        },
        "testability": candidate.testability,
        "unsupported_fields": unsupported,
    }
    normalized = normalize_candidate(raw)
    if normalized.rule is None:
        uncertainties.append(_uncertainty(window.window_id, "invalid_rule", "; ".join(normalized.errors)))
        return None
    rule = normalized.rule
    flags.extend(dataclasses.asdict(flag) for flag in normalized.flags)
    stored = {group.operator for group in rule.rule.conditions().values()}
    if candidate.operator_hint is not None and candidate.operator_hint not in stored:
        flags.append(
            _flag(
                "operator_hint_ignored", f"the model suggested {candidate.operator_hint}; the quote does not state it"
            )
        )
    severity, basis = assign_severity(rule, quote, expected, candidate.risk_category)
    rule = _replace(rule, severity=severity, severity_basis=basis)
    signals = ConfidenceSignals(
        quote_match=1.0 if found else 0.0,
        schema_completeness=_completeness(rule, expected in actions),
        definition_resolution=round(grounded / total, 4) if total else 1.0,
        low_ambiguity=_low_ambiguity(quote),
        cross_reference_resolution=_reference_score(window.target.get("cross_references")),
    )
    return Draft(
        rule=rule,
        origin=EXTRACTED,
        quote_found=found,
        signals=signals,
        expected_action=expected,
        flags=flags,
        decision_reasons=decision_reasons,
        glossary_terms=_term_list(window.target.get("glossary_terms")),
        window_id=window.window_id,
    )


def _legs(
    items: Sequence[Any], sources: Sequence[str], where: str, unsupported: list[str], flags: list[dict[str, Any]]
) -> tuple[list[str], int, int]:
    """Condition names, plus how many of them the clause or a definition states."""
    names: list[str] = []
    grounded = 0
    for item in items:
        readable = _ascii_snake(item.name)
        name = readable or condition_name(item.name)
        if name is None:
            flags.append(_flag("condition_name_unusable", f"{item.name!r} cannot be a condition name", where))
            if f"{where}.unusable_name" not in unsupported:
                unsupported.append(f"{where}.unusable_name")
            continue
        if name in names:
            continue
        names.append(name)
        found = bool(item.evidence.strip()) and any(quote_in_span(item.evidence, source) for source in sources)
        problems = []
        if readable is None:
            problems.append(
                _flag("condition_name_hashed", f"{item.name!r} has no ASCII form and was stored as {name}", where)
            )
        if not found:
            problems.append(
                _flag("condition_not_grounded", f"{name} is stated neither in the clause nor in a definition", where)
            )
        if problems:
            flags.extend(problems)
            unsupported.append(f"{where}.{name}")
        else:
            grounded += 1
    return names, grounded, len(names)


# Pre-authored IR


def _preauthored_draft(
    row: Mapping[str, Any], spans_by_id: Mapping[str, Mapping[str, Any]], config: ResolvedConfig, run_dir: Path
) -> Draft:
    try:
        rule = validate_rule(row.get("rule"))
    except RuleValidationError as exc:
        raise abort(
            RunStatus.INVALID_RUN_ARTIFACT, run_dir, f"{PREAUTHORED_FILE} holds an invalid rule; run stage=prepare."
        ) from exc
    evidence = row.get("evidence") if isinstance(row.get("evidence"), dict) else {}
    status = evidence.get("status")
    found = status in ("self_authored", "quote_found")
    span_ids = evidence.get("span_ids") if isinstance(evidence.get("span_ids"), list) else []
    spans = [spans_by_id[span_id] for span_id in span_ids if isinstance(span_id, str) and span_id in spans_by_id]
    raw_flags = row.get("flags", [])
    if not isinstance(raw_flags, list) or any(not isinstance(item, dict) for item in raw_flags):
        raise abort(
            RunStatus.INVALID_RUN_ARTIFACT,
            run_dir,
            f"{PREAUTHORED_FILE} has a flags value that is not a list of objects.",
        )
    flags = list(raw_flags)
    updates: dict[str, Any] = {}
    if rule.testability is None:
        # A person wrote these conditions; there is no extraction to doubt.
        updates["testability"] = "testable"
        flags.append(_flag("testability_from_author", "pre-authored rule without testability is read as testable"))
    if rule.severity_basis is None:
        updates["severity_basis"] = "source:pre_authored"
    if updates:
        rule = _replace(rule, **updates)
    actions = _rule_actions(rule)
    decision_reasons = []
    disabled = sorted(actions - set(config.proposals["decision_actions"]))
    if disabled:
        decision_reasons.append(f"{', '.join(disabled)} is not enabled in proposals.decision_actions")
    signals = ConfidenceSignals(
        quote_match=1.0 if found else 0.0,
        schema_completeness=_completeness(rule, True),
        definition_resolution=1.0,
        low_ambiguity=_low_ambiguity(rule.provenance.source_quote),
        cross_reference_resolution=_reference_score(_references(spans)),
    )
    if found:
        provenance_reasons: list[str] = []
    elif status is None:
        provenance_reasons = ["provenance was not checked"]
    else:
        provenance_reasons = [f"provenance check failed: {status}"]
    return Draft(
        rule=rule,
        origin=PRE_AUTHORED,
        quote_found=found,
        signals=signals,
        expected_action=next((action for action in ("escalate", "block", "allow") if action in actions), "block"),
        flags=flags,
        decision_reasons=decision_reasons,
        quarantine_reasons=provenance_reasons,
        glossary_terms=tuple(sorted({term for span in spans for term in span.get("glossary_terms", [])})),
    )


# IDs, pairing, severity, and signals


def stable_policy_id(document_id: str, section: str, quote: str, action: str) -> str:
    """Same document, section, quote, and action give the same ID on every run."""
    material = "\x1f".join((document_id, section, canonical_text(quote).casefold(), action))
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:8].upper()
    return f"{_id_part(document_id)}-{_id_part(section)}-{digest}"


def _assign_ids(drafts: Sequence[Draft], uncertainties: list[dict[str, Any]]) -> list[Draft]:
    """Stable IDs, then paired allow rules named after the rule they relax."""
    kept: list[Draft] = []
    by_id: dict[str, Draft] = {}
    for draft in drafts:
        provenance = draft.rule.provenance
        policy_id = stable_policy_id(
            provenance.document_id, provenance.section or "", provenance.source_quote, draft.expected_action
        )
        previous = by_id.get(policy_id)
        if previous is not None:
            if previous.shape != draft.shape:
                previous.decision_reasons.append("another candidate with the same quote states a different rule")
            uncertainties.append(_uncertainty(draft.window_id, "duplicate_candidate", provenance.source_quote))
            continue
        draft.rule = _replace(draft.rule, policy_id=policy_id)
        by_id[policy_id] = draft
        kept.append(draft)
    sections: dict[tuple[str, str], list[Draft]] = {}
    for draft in kept:
        sections.setdefault((draft.rule.provenance.document_id, draft.rule.provenance.section or ""), []).append(draft)
    suffixes: dict[str, int] = {}
    for group in sections.values():
        group.sort(key=lambda item: item.order)
        acting = [draft for draft in group if not _is_allow_only(draft)]
        for draft in group:
            if not acting or not _is_allow_only(draft):
                continue
            earlier = [other for other in acting if other.order < draft.order]
            partner = (earlier[-1] if earlier else acting[0]).rule.policy_id
            suffixes[partner] = suffixes.get(partner, 0) + 1
            count = suffixes[partner]
            policy_id = f"{partner}{ALLOW_SUFFIX}" + (f"-{count}" if count > 1 else "")
            draft.rule = _replace(draft.rule, policy_id=policy_id, paired_with=partner)
    return kept


def _is_allow_only(draft: Draft) -> bool:
    names = set(draft.rule.rule.conditions()) | set(draft.rule.unresolved_conditions)
    return draft.expected_action == "allow" and names == {"allow_if"}


def assign_severity(rule: PolicyRule, quote: str, expected_action: str, risk_category: str) -> tuple[str, str]:
    """The rubric; the first matching level wins. A model severity is never an input."""
    if expected_action == "allow":
        return "low", "rubric:explicit_allow"
    if risk_category in CRITICAL_CATEGORIES and _says(quote, _BINDING):
        return "critical", f"rubric:binding_language+{risk_category}"
    if expected_action == "escalate":
        return "high", "rubric:escalation"
    acting = [group for name, group in rule.rule.conditions().items() if name != "allow_if"]
    conditional = (
        bool(rule.exceptions)
        or any(group.operator == "none_of" or (group.operator == "all_of" and len(group.legs) > 1) for group in acting)
        or _says(quote, _CONDITIONAL)
    )
    if expected_action == "block" and not conditional:
        return "high", "rubric:explicit_prohibition"
    return "medium", "rubric:conditional_restriction"


def condition_name(text: str) -> str | None:
    """A snake_case condition name.

    Latin text is folded to ASCII (``Bỏ qua`` becomes ``bo_qua``). A name with no ASCII
    letters, such as a Chinese term, is kept as ``c_`` plus a stable hash so the condition
    is not dropped. None only when the text is empty.
    """
    readable = _ascii_snake(text)
    if readable:
        return readable
    folded = unicodedata.normalize("NFC", text).strip()
    if not folded:
        return None
    return "c_" + hashlib.sha256(folded.casefold().encode("utf-8")).hexdigest()[:8]


def _ascii_snake(text: str) -> str | None:
    folded = unicodedata.normalize("NFKD", text.replace("đ", "d").replace("Đ", "D"))
    ascii_text = "".join(char for char in folded if not unicodedata.combining(char)).encode("ascii", "ignore").decode()
    name = re.sub(r"[^a-z0-9]+", "_", ascii_text.casefold()).strip("_")[:128].strip("_")
    return name or None


def _completeness(rule: PolicyRule, action_produced: bool) -> float:
    checks = [
        rule.rule.actor != "unspecified",
        rule.rule.action != "unspecified",
        rule.rule.object != "unspecified",
        bool(rule.rule.conditions()),
        rule.testability is not None,
        action_produced,
    ]
    return round(sum(checks) / len(checks), 4)


def _low_ambiguity(quote: str) -> float:
    if detect_operator(quote) == "mixed":
        return 0.0
    return 0.5 if _says(quote, _HEDGES) else 1.0


def _rule_actions(rule: PolicyRule) -> set[str]:
    produced = {
        "block_if": rule.decision.violation_action,
        "escalate_if": rule.decision.critical_action or "escalate",
        "require_confirmation_if": "require_confirmation",
        "allow_if": "allow",
    }
    return {produced[name] for name in rule.rule.conditions()}


def _says(text: str, phrases: Sequence[str]) -> bool:
    folded = canonical_text(text).casefold()
    return any(
        re.search(rf"(?<!\w){re.escape(canonical_text(phrase).casefold())}(?!\w)", folded) for phrase in phrases
    )


def _id_part(text: str) -> str:
    folded = unicodedata.normalize("NFKD", text.replace("đ", "d").replace("Đ", "D"))
    ascii_text = "".join(char for char in folded if not unicodedata.combining(char)).encode("ascii", "ignore").decode()
    return re.sub(r"[^A-Z0-9]+", "-", ascii_text.upper()).strip("-")[:40].strip("-") or "X"


def _text(value: str) -> str:
    return value.strip() or "unspecified"


def _replace(rule: PolicyRule, **updates: Any) -> PolicyRule:
    return PolicyRule.model_validate({**rule.model_dump(), **updates})


def _flag(code: str, detail: str, field_name: str | None = None) -> dict[str, Any]:
    return {"code": code, "detail": detail, "field": field_name}


def _uncertainty(window_id: str | None, reason: str, detail: str | None) -> dict[str, Any]:
    return {"window_id": window_id, "reason": reason, "detail": detail}


# Artifacts


def _counts(routed: Sequence[tuple[Draft, PolicyRule, list[str]]], conflicts: int, uncertainties: int, windows: int):
    counts = {status: 0 for status in ("eligible_for_generation", "needs_decision", "quarantined", "rejected")}
    for _, rule, _ in routed:
        counts[rule.status] += 1
    return {
        "windows": windows,
        "extracted_rules": sum(1 for draft, _, _ in routed if draft.origin == EXTRACTED),
        "pre_authored_rules": sum(1 for draft, _, _ in routed if draft.origin == PRE_AUTHORED),
        **counts,
        "unresolved_conflicts": conflicts,
        "extraction_uncertainties": uncertainties,
    }


def _write(
    stage_cache: Path,
    routed: Sequence[tuple[Draft, PolicyRule, list[str]]],
    extracted: Sequence[Draft],
    uncertainties: Sequence[dict[str, Any]],
    conflicts: Any,
    glossary_conflicts: Sequence[Any],
    log: dict[str, Any],
    counts: dict[str, int],
) -> None:
    rows = [
        {
            "policy_id": rule.policy_id,
            "origin": draft.origin,
            "status": rule.status,
            "confidence": rule.confidence,
            "reasons": reasons,
            "flags": draft.flags,
            "rule": rule.model_dump(mode="json", exclude_none=True),
        }
        for draft, rule, reasons in routed
    ]
    write_jsonl(
        stage_cache / CANDIDATES_FILE,
        [
            {
                "policy_id": draft.rule.policy_id,
                "source_span_id": draft.window_id,
                "quote_found": draft.quote_found,
                "passes": draft.passes,
                "signals": dataclasses.asdict(draft.signals),
                "rule": draft.rule.model_dump(mode="json", exclude_none=True),
            }
            for draft in sorted(extracted, key=lambda item: item.rule.policy_id)
        ],
    )
    write_jsonl(stage_cache / UNCERTAINTIES_FILE, uncertainties)
    write_jsonl(stage_cache / POLICY_RULES_FILE, rows)
    write_jsonl(
        stage_cache / FLAGS_FILE,
        [{k: v for k, v in row.items() if k != "rule"} for row in rows if row["status"] != "eligible_for_generation"],
    )
    write_yaml(
        stage_cache / ELIGIBLE_FILE,
        {"rules": [row["rule"] for row in rows if row["status"] == "eligible_for_generation"]},
    )
    write_json(stage_cache / GRAPH_FILE, {"nodes": conflicts.graph})
    write_json(
        stage_cache / CONFLICT_FILE,
        {"conflicts": conflicts.conflicts, "glossary_conflicts": list(glossary_conflicts)},
    )
    write_json(
        stage_cache / ROUTING_LOG_FILE,
        {
            "prompt_version": extraction.PROMPT_VERSION,
            "extraction": log,
            "thresholds": {"eligible": ELIGIBLE_THRESHOLD, "decision": DECISION_THRESHOLD},
            "weights": CONFIDENCE_WEIGHTS,
            "counts": counts,
        },
    )


def _load_json(path: Path, run_dir: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise abort(
            RunStatus.INVALID_RUN_ARTIFACT, run_dir, f"{path.name} cannot be read; run stage=prepare."
        ) from exc
    if not isinstance(data, dict):
        raise abort(RunStatus.INVALID_RUN_ARTIFACT, run_dir, f"{path.name} must hold an object; run stage=prepare.")
    return data


def _references(spans: Sequence[Mapping[str, Any]]) -> list[Any]:
    refs: list[Any] = []
    for span in spans:
        items = span.get("cross_references", [])
        if isinstance(items, list):
            refs.extend(items)
    return refs


def _reference_score(items: Any) -> float | None:
    """Share of cross-references that resolve. None when the clause cites nothing."""
    if not isinstance(items, list):
        return None
    refs = [item for item in items if isinstance(item, dict)]
    if not refs:
        return None
    return round(sum(1 for item in refs if item.get("resolved") is True) / len(refs), 4)


def _term_list(items: Any) -> tuple[str, ...]:
    if not isinstance(items, list):
        return ()
    return tuple(item for item in items if isinstance(item, str))


def _check_glossary_conflicts(entries: Sequence[Any], run_dir: Path) -> None:
    for entry in entries:
        definitions = entry.get("definitions") if isinstance(entry, dict) else None
        valid = isinstance(entry, dict) and isinstance(entry.get("key"), str) and isinstance(definitions, list)
        valid = valid and all(
            isinstance(item, dict) and isinstance(item.get("source_span_id"), str) for item in definitions
        )
        if not valid:
            raise abort(
                RunStatus.INVALID_RUN_ARTIFACT,
                run_dir,
                f"{GLOSSARY_FILE} has a conflict entry that is not a term plus its definitions. Run stage=prepare.",
            )


def _load_jsonl(path: Path, run_dir: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise abort(RunStatus.INVALID_RUN_ARTIFACT, run_dir, f"{path.name} is missing; run stage=prepare.") from exc
    except (OSError, UnicodeDecodeError) as exc:
        raise abort(
            RunStatus.INVALID_RUN_ARTIFACT, run_dir, f"{path.name} cannot be read; run stage=prepare."
        ) from exc
    rows = []
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise abort(
                RunStatus.INVALID_RUN_ARTIFACT, run_dir, f"{path.name} is damaged; run stage=prepare."
            ) from exc
        if not isinstance(row, dict):
            raise abort(RunStatus.INVALID_RUN_ARTIFACT, run_dir, f"{path.name} is damaged; run stage=prepare.")
        rows.append(row)
    return rows
