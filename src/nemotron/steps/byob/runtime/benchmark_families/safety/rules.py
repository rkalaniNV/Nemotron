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
from nemotron.steps.byob.runtime.benchmark_families.safety.actions import ACTION_ORDER
from nemotron.steps.byob.runtime.benchmark_families.safety.artifacts import write_json, write_jsonl, write_yaml
from nemotron.steps.byob.runtime.benchmark_families.safety.confidence import (
    CONFIDENCE_WEIGHTS,
    DECISION_THRESHOLD,
    ELIGIBLE_THRESHOLD,
    ConfidenceSignals,
    canonical_text,
    compute_confidence,
    operator_explicitness,
    quote_in_span,
    route_rule,
)
from nemotron.steps.byob.runtime.benchmark_families.safety.config import ResolvedConfig
from nemotron.steps.byob.runtime.benchmark_families.safety.conflicts import detect_conflicts
from nemotron.steps.byob.runtime.benchmark_families.safety.coverage import COVERAGE_OUTPUTS
from nemotron.steps.byob.runtime.benchmark_families.safety.dataset import DATASET_OUTPUTS
from nemotron.steps.byob.runtime.benchmark_families.safety.extract import (
    ExtractedCandidate,
    ExtractedOperator,
    ExtractedScopeItem,
    ExtractionError,
    PassOutcome,
)
from nemotron.steps.byob.runtime.benchmark_families.safety.ingest import GLOSSARY_FILE, INDEX_FILE
from nemotron.steps.byob.runtime.benchmark_families.safety.ingest import RULES_FILE as PREAUTHORED_FILE
from nemotron.steps.byob.runtime.benchmark_families.safety.ir import (
    CONDITION_CLASSES,
    DEFAULT_TARGETS,
    PolicyRule,
    RuleValidationError,
    detect_operator,
    normalize_candidate,
    validate_rule,
)
from nemotron.steps.byob.runtime.benchmark_families.safety.paraphrase import PARAPHRASE_OUTPUTS
from nemotron.steps.byob.runtime.benchmark_families.safety.scenarios import SCENARIO_OUTPUTS
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
_VIOLATION_ACTIONS = frozenset({"block", "caution", "rewrite", "redact"})
_HASHED_CONDITION = re.compile(r"^c_[0-9a-f]{8}$")


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
        operators = _operator_map(self)
        return (self.expected_action, tuple(mapped), tuple(mapped.values()), tuple(operators.values()))

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
    for name in (*RULE_OUTPUTS, *COVERAGE_OUTPUTS, *SCENARIO_OUTPUTS, *PARAPHRASE_OUTPUTS, *DATASET_OUTPUTS):
        (stage_cache / name).unlink(missing_ok=True)
    try:
        windows = build_windows(documents, spans, terms)
    except (TypeError, ValueError) as exc:
        raise abort(
            RunStatus.INVALID_RUN_ARTIFACT, run_dir, f"{INDEX_FILE} is damaged: {exc} Run stage=prepare."
        ) from exc
    uncertainties: list[dict[str, Any]] = []
    cited = _cited_window_ids(preauthored)
    pending = [window for window in windows if window.window_id not in cited]
    extracted, log = _extract(pending, config, run_dir, uncertainties)
    log["cited_windows"] = len(windows) - len(pending)
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


def _cited_window_ids(preauthored: Sequence[Mapping[str, Any]]) -> set[str]:
    """Windows whose clause is already the source quote of a pre-authored rule.

    Only ``quote_found`` citations skip extraction. A mis-cited or self-authored
    rule leaves the window pending, so leftover policy text still needs a model.
    """
    cited: set[str] = set()
    for row in preauthored:
        if not isinstance(row, Mapping):
            continue
        evidence = row.get("evidence") if isinstance(row.get("evidence"), Mapping) else {}
        if evidence.get("status") != "quote_found":
            continue
        for item in evidence.get("span_ids") or []:
            if isinstance(item, str) and item.strip():
                cited.add(item)
        rule = row.get("rule") if isinstance(row.get("rule"), Mapping) else {}
        provenance = rule.get("provenance") if isinstance(rule.get("provenance"), Mapping) else {}
        span_id = provenance.get("source_span_id")
        if isinstance(span_id, str) and span_id.strip():
            cited.add(span_id)
    return cited


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
        if match is not None and _operator_map(draft) != _operator_map(match):
            draft.decision_reasons.append("the second extraction pass disagrees on condition operators")
            uncertainties.append(_uncertainty(window_id, "operator_disagreement", draft.rule.provenance.source_quote))
    for other in remaining:
        uncertainties.append(_uncertainty(window_id, "second_pass_only", other.rule.provenance.source_quote))


def _condition_map(draft: Draft) -> dict[str, frozenset[str]]:
    names = set(draft.rule.rule.conditions()) | set(draft.rule.unresolved_conditions)
    return {name: frozenset(draft._legs(name)) for name in sorted(names)}


def _operator_map(draft: Draft) -> dict[str, str | None]:
    names = set(draft.rule.rule.conditions()) | set(draft.rule.unresolved_conditions)
    return {
        name: (group.operator if (group := draft.rule.rule.conditions().get(name)) is not None else None)
        for name in sorted(names)
    }


def _agreement(first: Draft, second: Draft) -> float:
    """Share of action, condition classes, condition names, and operators across passes."""
    left, right = _condition_map(first), _condition_map(second)
    checks = (
        first.expected_action == second.expected_action,
        set(left) == set(right),
        left == right,
        _operator_map(first) == _operator_map(second),
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
    # The clause, its neighbors, and structural definitions. A verb glossary is context for
    # the model, not evidence that can clear the grounding check.
    sources = [clause, *window.grounding_texts]
    flags: list[dict[str, Any]] = []
    unsupported: list[str] = []
    body: dict[str, Any] = {name: _text(getattr(candidate, name)) for name in ("actor", "action", "object")}
    unresolved: dict[str, list[str]] = {}
    condition_names: set[str] = set()
    for name in CONDITION_CLASSES:
        legs, _, _ = _legs(getattr(candidate.conditions, name), sources, f"rule.{name}", unsupported, flags)
        if not legs:
            continue
        condition_names.add(name)
        group = _extracted_operator(name, legs, candidate.operators_by_class.get(name), quote, flags)
        if group is None:
            unresolved[name] = legs
        else:
            body[name] = group
    for name in sorted(set(candidate.operators_by_class) - condition_names):
        flags.append(_flag("operator_without_conditions", f"the model supplied an operator for empty {name}", name))
        unsupported.append(f"rule.{name}.operator")
    exceptions, _, _ = _legs(candidate.exceptions, sources, "exceptions", unsupported, flags)
    if not condition_names:
        uncertainties.append(_uncertainty(window.window_id, "no_conditions", quote))
        return None
    expected = candidate.expected_action
    produced = {
        "block_if": expected if expected in _VIOLATION_ACTIONS else "block",
        "escalate_if": "escalate",
        "require_confirmation_if": "require_confirmation",
        "allow_if": "allow",
    }
    actions = {produced[name] for name in condition_names}
    decision_reasons = []
    if expected not in actions:
        decision_reasons.append(f"expected_action {expected} is not produced by any of its conditions")
    disabled = sorted((actions | {expected}) - set(config.proposals["decision_actions"]))
    if disabled:
        decision_reasons.append(f"{', '.join(disabled)} is not enabled in proposals.decision_actions")
    scoped = {
        name: values
        for name in ("domains", "jurisdictions", "languages", "audiences", "targets")
        if (
            values := _scope_values(
                name,
                getattr(candidate.scope, name),
                quote,
                unsupported,
                flags,
            )
        )
    }
    scope = {**_proposal_scope(config.proposals), **scoped}
    raw = {
        "policy_id": "PENDING",
        "title": candidate.title.strip() or None,
        "scope": scope,
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
    if unresolved:
        raw["unresolved_conditions"] = unresolved
    stored_signals = _grounded_signals(candidate, quote, flags)
    if stored_signals:
        raw["signals"] = stored_signals
    normalized = normalize_candidate(raw)
    if normalized.rule is None:
        uncertainties.append(_uncertainty(window.window_id, "invalid_rule", "; ".join(normalized.errors)))
        return None
    rule = normalized.rule
    flags.extend(dataclasses.asdict(flag) for flag in normalized.flags)
    if candidate.operator_hint is not None:
        flags.append(
            _flag(
                "legacy_operator_hint_ignored",
                "operator_hint is not an authority; use condition_operators with exact evidence",
            )
        )
    severity, basis = assign_severity(
        rule,
        quote,
        expected,
        candidate.risk_category,
        candidate.critical_risk_basis,
    )
    rule = _replace(rule, severity=severity, severity_basis=basis)
    signals = ConfidenceSignals(
        quote_match=1.0 if found else 0.0,
        schema_completeness=_completeness(rule, expected in actions),
        operator_explicitness=operator_explicitness(rule),
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


def _extracted_operator(
    name: str,
    legs: Sequence[str],
    proposed: ExtractedOperator | None,
    quote: str,
    flags: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Store a model operator only with exact evidence; keywords may confirm or reject it."""
    if len(legs) == 1 and (proposed is None or proposed.operator != "none_of"):
        if proposed is not None:
            flags.append(_flag("operator_not_needed", f"{name} has one positive condition", f"rule.{name}"))
        return {"any_of": list(legs), "operator_source": "single_condition"}
    if proposed is None:
        flags.append(
            _flag(
                "operator_missing",
                f"{name} needs a per-condition operator with exact source evidence",
                f"rule.{name}",
            )
        )
        return None
    evidence = proposed.evidence.strip()
    if not evidence or not quote_in_span(evidence, quote):
        flags.append(
            _flag(
                "operator_evidence_not_grounded",
                f"{name} operator evidence is not an exact part of source_quote",
                f"rule.{name}",
            )
        )
        return None
    detected = detect_operator(evidence)
    if detected == "mixed":
        flags.append(
            _flag(
                "operator_keyword_ambiguous",
                f"{name} evidence contains more than one operator frame",
                f"rule.{name}",
            )
        )
        return None
    if detected is not None and detected != proposed.operator:
        flags.append(
            _flag(
                "operator_evidence_conflict",
                f"{name} proposes {proposed.operator}, while deterministic wording says {detected}",
                f"rule.{name}",
            )
        )
        return None
    flags.append(
        _flag(
            "operator_from_model_evidence",
            f"{name} stores {proposed.operator} from grounded model evidence",
            f"rule.{name}",
        )
    )
    if detected == proposed.operator:
        flags.append(
            _flag(
                "operator_keyword_confirmed",
                f"{name} operator agrees with deterministic wording",
                f"rule.{name}",
            )
        )
    return {proposed.operator: list(legs), "operator_source": "model_evidence"}


def _legs(
    items: Sequence[Any], sources: Sequence[str], where: str, unsupported: list[str], flags: list[dict[str, Any]]
) -> tuple[list[str], int, int]:
    """Condition names, plus how many are stated in the clause, a neighbor, or a structural definition."""
    names: list[str] = []
    grounded = 0
    for item in items:
        name = condition_name(item.name)
        if name is None:
            flags.append(_flag("condition_name_unusable", f"{item.name!r} cannot be a condition name", where))
            if f"{where}.unusable_name" not in unsupported:
                unsupported.append(f"{where}.unusable_name")
            continue
        if name in names:
            flags.append(
                _flag(
                    "condition_name_collision",
                    f"{item.name!r} normalizes to the already used condition name {name!r}",
                    where,
                )
            )
            collision = f"{where}.{name}"
            if collision not in unsupported:
                unsupported.append(collision)
            continue
        names.append(name)
        found = bool(item.evidence.strip()) and any(_evidence_in_source(item.evidence, source) for source in sources)
        problems = []
        if _hashed_condition_name(name):
            problems.append(
                _flag("condition_name_hashed", f"{item.name!r} has no identifier form and was stored as {name}", where)
            )
        if not found:
            problems.append(
                _flag(
                    "condition_not_grounded",
                    f"{name} is not in the clause, a neighboring clause, or a structural definition",
                    where,
                )
            )
        if problems:
            flags.extend(problems)
            unsupported.append(f"{where}.{name}")
        else:
            grounded += 1
    return names, grounded, len(names)


def _evidence_in_source(evidence: str, source: str) -> bool:
    """Exact grounding, including a substring match for scripts that do not separate words."""
    return quote_in_span(evidence, source)


def _scope_values(
    name: str,
    items: Sequence[ExtractedScopeItem],
    quote: str,
    unsupported: list[str],
    flags: list[dict[str, Any]],
) -> list[str]:
    values: list[str] = []
    for item in items:
        value = item.value.strip()
        evidence = item.evidence.strip()
        where = f"scope.{name}.{value or 'empty'}"
        if not value or not evidence or not quote_in_span(evidence, quote):
            flags.append(_flag("scope_evidence_not_grounded", f"{where} is not grounded in source_quote", where))
            unsupported.append(where)
            continue
        if value not in values:
            values.append(value)
    return values


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
    rule = _with_cited_span(rule, evidence, spans, flags)
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
        operator_explicitness=operator_explicitness(rule),
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
        expected_action=_strongest_action(actions),
        flags=flags,
        decision_reasons=decision_reasons,
        quarantine_reasons=provenance_reasons,
        glossary_terms=tuple(sorted({term for span in spans for term in span.get("glossary_terms", [])})),
        window_id=rule.provenance.source_span_id,
    )


def _with_cited_span(
    rule: PolicyRule,
    evidence: Mapping[str, Any],
    spans: Sequence[Mapping[str, Any]],
    flags: list[dict[str, Any]],
) -> PolicyRule:
    """Bind a unique matching span onto provenance so accepted cases can cite it.

    Ingest does not rewrite the authored IR. Generate fills the span only when the
    quote matched exactly one source span and the author did not already name one.
    """
    if rule.provenance.source_span_id or len(spans) != 1:
        return rule
    span = spans[0]
    span_id = span.get("source_span_id")
    if not isinstance(span_id, str) or not span_id.strip():
        return rule
    provenance = dict(rule.provenance.model_dump())
    provenance["source_span_id"] = span_id
    section = span.get("section")
    if isinstance(section, str) and section.strip():
        provenance["section"] = section
    sha = evidence.get("document_sha256")
    if isinstance(sha, str) and sha:
        provenance["document_sha256"] = sha
    document_id = span.get("document_id")
    if isinstance(document_id, str) and document_id:
        provenance["document_id"] = document_id
    flags.append(
        _flag(
            "span_bound_from_evidence",
            "pre-authored quote matched exactly one source span",
            "provenance.source_span_id",
        )
    )
    return _replace(rule, provenance=provenance)


# IDs, pairing, severity, and signals


def stable_policy_id(
    document_id: str,
    section: str,
    quote: str,
    action: str,
    shape: Sequence[Any] | None = None,
) -> str:
    """Same source and canonical rule shape give the same ID on every run."""
    parts = [document_id, section, canonical_text(quote).casefold(), action]
    if shape is not None:
        parts.append(json.dumps(shape, ensure_ascii=False, separators=(",", ":")))
    material = "\x1f".join(parts)
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:8].upper()
    return f"{_id_part(document_id)}-{_id_part(section)}-{digest}"


def _id_shape(draft: Draft) -> tuple[Any, ...]:
    return tuple(
        (
            name,
            group.operator if (group := draft.rule.rule.conditions().get(name)) is not None else None,
            tuple(sorted(draft._legs(name))),
        )
        for name in sorted(set(draft.rule.rule.conditions()) | set(draft.rule.unresolved_conditions))
    )


def _assign_ids(drafts: Sequence[Draft], uncertainties: list[dict[str, Any]]) -> list[Draft]:
    """Stable IDs, then paired allow rules named after the rule they relax."""
    kept: list[Draft] = []
    by_id: dict[str, Draft] = {}
    for draft in drafts:
        provenance = draft.rule.provenance
        policy_id = stable_policy_id(
            provenance.document_id,
            provenance.section or "",
            provenance.source_quote,
            draft.expected_action,
            _id_shape(draft),
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
            if len(acting) != 1:
                draft.decision_reasons.append(f"allow rule has {len(acting)} possible acting partners in its section")
                continue
            partner = acting[0].rule.policy_id
            suffixes[partner] = suffixes.get(partner, 0) + 1
            count = suffixes[partner]
            policy_id = f"{partner}{ALLOW_SUFFIX}" + (f"-{count}" if count > 1 else "")
            draft.rule = _replace(draft.rule, policy_id=policy_id, paired_with=partner)
    return kept


def _is_allow_only(draft: Draft) -> bool:
    names = set(draft.rule.rule.conditions()) | set(draft.rule.unresolved_conditions)
    return draft.expected_action == "allow" and names == {"allow_if"}


def assign_severity(
    rule: PolicyRule,
    quote: str,
    expected_action: str,
    risk_category: str,
    critical_risk_basis: str | None = None,
) -> tuple[str, str]:
    """The rubric; the first matching level wins. A model severity is never an input.

    Binding and conditional wording count only when extraction stored exact evidence from
    ``quote``. The quote itself is not scanned for a language's keyword list.
    """
    del quote
    if expected_action == "allow":
        return "low", "rubric:explicit_allow"
    if critical_risk_basis is not None and risk_category not in ("", "none") and rule.signals.binding_evidence:
        return "critical", f"rubric:binding_evidence+{critical_risk_basis}"
    if expected_action == "escalate":
        return "high", "rubric:escalation"
    acting = [group for name, group in rule.rule.conditions().items() if name != "allow_if"]
    conditional = (
        bool(rule.exceptions)
        or any(group.operator == "none_of" for group in acting)
        or bool(rule.signals.conditional_evidence)
    )
    if expected_action == "block" and not conditional:
        return "high", "rubric:explicit_prohibition"
    return "medium", "rubric:conditional_restriction"


def condition_name(text: str) -> str | None:
    """A stored condition name.

    All-Latin text is folded to ASCII snake_case (``Bỏ qua`` becomes ``bo_qua``).
    A name containing another script keeps all of its letters and numbers, including
    mixed-script suffixes. Punctuation-only or digit-leading text becomes ``c_`` plus
    a stable hash and is sent to review. None is returned only for empty text.
    """
    folded = unicodedata.normalize("NFC", text).strip()
    if not folded:
        return None
    letters = [char for char in folded if char.isalpha()]
    if letters and all("LATIN" in unicodedata.name(char, "") for char in letters):
        readable = _ascii_snake(folded)
        if readable and readable[0].isalpha():
            return readable
    if any(char.isalpha() and "LATIN" not in unicodedata.name(char, "") for char in folded):
        script = re.sub(r"[^\w]+", "_", folded.casefold(), flags=re.UNICODE).strip("_")[:128].strip("_")
        if script and script[0].isalpha() and all(char.isalnum() or char == "_" for char in script):
            return script
    return "c_" + hashlib.sha256(folded.casefold().encode("utf-8")).hexdigest()[:8]


def _hashed_condition_name(name: str) -> bool:
    return bool(_HASHED_CONDITION.fullmatch(name))


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


def _rule_actions(rule: PolicyRule) -> set[str]:
    produced = {
        "block_if": rule.decision.violation_action,
        "escalate_if": rule.decision.critical_action or "escalate",
        "require_confirmation_if": "require_confirmation",
        "allow_if": "allow",
    }
    return {produced[name] for name in rule.rule.conditions()}


def _strongest_action(actions: set[str]) -> str:
    return next((action for action in ACTION_ORDER if action in actions), "block")


def _proposal_languages(proposals: Mapping[str, Any]) -> list[str]:
    """The clause's fallback language is the source language. Targets are translation work."""
    value = proposals.get("source_language")
    if isinstance(value, str) and value.strip() and value.strip().casefold() != "unspecified":
        return [value.strip()]
    return []


def _proposal_scope(proposals: Mapping[str, Any]) -> dict[str, list[str]]:
    """Run-level fallbacks; empty domain and jurisdiction become IR's wildcard ``unspecified``."""

    def stated(value: Any) -> list[str]:
        if isinstance(value, str) and value.strip() and value.strip().casefold() != "unspecified":
            return [value.strip()]
        return []

    targets = [
        item.strip()
        for item in proposals.get("moderation_targets", [])
        if isinstance(item, str) and item.strip() and item.strip().casefold() != "unspecified"
    ]
    return {
        "domains": stated(proposals.get("domain")),
        "jurisdictions": stated(proposals.get("jurisdiction")),
        "languages": _proposal_languages(proposals),
        "targets": targets or list(DEFAULT_TARGETS),
    }


def _grounded_signals(
    candidate: extraction.ExtractedCandidate, quote: str, flags: list[dict[str, Any]]
) -> dict[str, Any]:
    """Keep modality and discourse hints only when their evidence is inside the quote."""
    stored: dict[str, Any] = {}
    binding = _optional_evidence(candidate.binding_evidence, quote, "binding_evidence", flags)
    conditional = _optional_evidence(candidate.conditional_evidence, quote, "conditional_evidence", flags)
    multi_turn = _optional_evidence(candidate.multi_turn_evidence, quote, "multi_turn_evidence", flags)
    code_switch = _optional_evidence(candidate.code_switch_evidence, quote, "code_switch_evidence", flags)
    if binding:
        stored["binding_evidence"] = binding
    if conditional:
        stored["conditional_evidence"] = conditional
    if multi_turn:
        stored["multi_turn_turns"] = _turn_count(candidate.multi_turn_count, flags)
    if code_switch:
        stored["code_switch"] = True
    return stored


def _optional_evidence(evidence: str, quote: str, field_name: str, flags: list[dict[str, Any]]) -> str | None:
    text = evidence.strip()
    if not text:
        return None
    if quote_in_span(text, quote):
        return text
    flags.append(_flag("signal_not_grounded", f"{field_name} is not an exact part of source_quote", field_name))
    return None


def _turn_count(count: int | None, flags: list[dict[str, Any]]) -> int:
    if isinstance(count, int) and not isinstance(count, bool) and 2 <= count <= 99:
        return count
    if count is not None:
        flags.append(
            _flag("signal_count_ignored", "multi_turn_count must be an integer from 2 to 99", "multi_turn_count")
        )
    return 2


def _id_part(text: str) -> str:
    folded = unicodedata.normalize("NFKD", text.replace("đ", "d").replace("Đ", "D"))
    ascii_text = "".join(char for char in folded if not unicodedata.combining(char)).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^A-Z0-9]+", "-", ascii_text.upper()).strip("-")[:40].strip("-")
    if slug:
        return slug
    digest = hashlib.sha256(unicodedata.normalize("NFC", text).encode("utf-8")).hexdigest()[:4].upper()
    return f"U{digest}"


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
