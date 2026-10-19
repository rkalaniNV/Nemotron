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

"""Budgeted coverage plan for eligible policy rules.

The plan names the cases a later stage must generate. It does not write case text and
it does not call a model. Required cells are scheduled before optional cells. A required
cell that does not fit is listed in the gap file, and the plan is Silver. Per-rule and
global maxima are hard caps. Mix-derived multi-turn is the first optional kind considered
when room remains. If later validation or exact-dedup leaves room under the case cap,
quality-gate backfill may promote unscheduled distinct cells. It never copies a cell that
shares the quote-copy identity of a scheduled or surviving row.
"""

import hashlib
import json
from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nemotron.steps.byob.runtime.benchmark_families.safety.artifacts import write_json, write_jsonl
from nemotron.steps.byob.runtime.benchmark_families.safety.case_facts import (
    acting_groups,
    case_context,
    primary_action,
    violation_facts,
)
from nemotron.steps.byob.runtime.benchmark_families.safety.ir import (
    ConditionGroup,
    PolicyRule,
    RuleValidationError,
    validate_rule,
)

PLAN_FILE = "coverage_plan.jsonl"
GAPS_FILE = "coverage_gaps.json"
COVERAGE_OUTPUTS = (PLAN_FILE, GAPS_FILE)
MAX_BACKFILL_ROUNDS = 8
_UNGENERATABLE_GAP_REASONS = frozenset({"translation_required", "span_missing", "max_turns"})
# A case rejected for one of these fails because of its rule, so every other cell of that rule fails too.
RULE_LEVEL_REJECTIONS = frozenset({"quote_not_in_span", "span_missing", "evidence_mismatch", "evidence_missing"})

# Required cells are filled in this order across every rule, so a short budget keeps a
# clear violation for each rule before any rule receives a later cell. Optional cells are
# a separate pass after every required cell.
_REQUIRED_KIND_ORDER = (
    "clear_violation",
    "all_legs",
    "condition_path",
    "clear_allow",
    "boundary",
    "exception",
    "missing_leg",
    "language",
    "code_switch",
    "multi_turn",
)
# Mix-derived multi-turn reuses kind "multi_turn" so turn_mix can rank it.
_OPTIONAL_KIND_ORDER = ("adversarial", "actor_swap", "persona")
_KIND_RANK = {kind: rank for rank, kind in enumerate(_REQUIRED_KIND_ORDER + _OPTIONAL_KIND_ORDER)}
_RECORD_FIELDS = (
    "certification",
    "reason",
    "coverage_of",
    "case_cap",
    "required_cases",
    "gold_cases_needed",
    "scheduled_cases",
    "optional_scheduled",
    "missing_required",
    "excluded_rules",
    "modules",
    "capacity",
    "quality_gate_preflight",
)


class CoverageInputError(ValueError):
    """The eligible-rule file or the source index cannot be planned."""


@dataclass(frozen=True)
class CoveragePlan:
    """Scheduled rows, the cells that did not fit, and the certification those imply."""

    rows: tuple[dict[str, Any], ...]
    gaps: tuple[dict[str, Any], ...]
    contrast_families: tuple[dict[str, Any], ...]
    certification: str
    reason: str
    case_cap: int
    required_cases: int
    excluded_rules: tuple[dict[str, str], ...]
    modules: dict[str, str]
    capacity: dict[str, Any]
    quality_gate_preflight: dict[str, Any]

    @property
    def scheduled_cases(self) -> int:
        return len(self.rows)

    @property
    def optional_scheduled(self) -> int:
        return sum(1 for row in self.rows if row["priority"] == "optional")

    def summary(self) -> dict[str, Any]:
        missing = [gap["slot_id"] for gap in self.gaps if gap["blocks_gold"]]
        return {
            "certification": self.certification,
            "reason": self.reason,
            "coverage_of": "eligible_rules",
            "case_cap": self.case_cap,
            "required_cases": self.required_cases,
            "gold_cases_needed": self.required_cases,
            "scheduled_cases": self.scheduled_cases,
            "optional_scheduled": self.optional_scheduled,
            "missing_required": missing,
            "excluded_rules": [dict(row) for row in self.excluded_rules],
            "modules": self.modules,
            "capacity": dict(self.capacity),
            "quality_gate_preflight": dict(self.quality_gate_preflight),
            "gaps": [dict(gap) for gap in self.gaps],
            "contrast_families": [dict(family) for family in self.contrast_families],
        }

    def record(self) -> dict[str, Any]:
        """The part of the summary stored on the run record. Gap rows stay in the gap file."""
        return {key: self.summary()[key] for key in _RECORD_FIELDS}


def case_cap_for(mode: str, budget: Mapping[str, Any]) -> int:
    """Preview has its own cap. Assisted and expert both use ``assisted_max_cases``."""
    key = "preview_max_cases" if mode == "preview" else "assisted_max_cases"
    return int(budget[key])


def load_eligible_rules(data: Any) -> list[PolicyRule]:
    """Read the eligible-rule document. A rule that is not eligible is refused."""
    if not isinstance(data, dict) or not isinstance(data.get("rules"), list):
        raise CoverageInputError("policy_rules_eligible.yaml must contain a rules list.")
    rules: list[PolicyRule] = []
    seen: set[str] = set()
    for index, row in enumerate(data["rules"]):
        try:
            rule = validate_rule(row)
        except RuleValidationError as exc:
            raise CoverageInputError(f"policy_rules_eligible.yaml rule {index} is invalid: {exc}") from exc
        if rule.status != "eligible_for_generation":
            raise CoverageInputError(f"{rule.policy_id} is listed for generation but its status is {rule.status}.")
        if rule.policy_id in seen:
            raise CoverageInputError(f"policy_rules_eligible.yaml repeats {rule.policy_id}.")
        seen.add(rule.policy_id)
        rules.append(rule)
    return rules


def span_text(index: Any) -> dict[str, str]:
    """Map each source span id to its text. A damaged or repeated span stops the plan."""
    if not isinstance(index, dict) or not isinstance(index.get("spans"), list):
        raise CoverageInputError("source_index.json must contain a spans list.")
    text: dict[str, str] = {}
    for number, span in enumerate(index["spans"], start=1):
        where = f"source_index.json span {number}"
        if not isinstance(span, dict):
            raise CoverageInputError(f"{where} must be an object.")
        span_id = span.get("source_span_id")
        body = span.get("text", "")
        if not isinstance(span_id, str) or not span_id.strip():
            raise CoverageInputError(f"{where} needs a source_span_id.")
        if not isinstance(body, str):
            raise CoverageInputError(f"{where} text must be a string.")
        if span_id in text:
            raise CoverageInputError(f"{where} repeats {span_id}.")
        text[span_id] = body
    return text


def load_routed_statuses(text: str) -> list[dict[str, Any]]:
    """Read ``policy_rules.jsonl``. Each non-empty line must be a rule object."""
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CoverageInputError(f"policy_rules.jsonl line {number} is not JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise CoverageInputError(f"policy_rules.jsonl line {number} must be an object.")
        rows.append(row)
    return rows


def build_coverage_plan(
    rules: Sequence[PolicyRule],
    spans: Mapping[str, str],
    *,
    case_cap: int,
    budget: Mapping[str, Any],
    modules: Mapping[str, Any],
    generation: Mapping[str, Any] | None = None,
    routed: Sequence[Mapping[str, Any]] | None = None,
    target_languages: Sequence[str] = (),
    quality_gates: Mapping[str, Any] | None = None,
) -> CoveragePlan:
    """Plan eligible rules. Other statuses are listed under ``excluded_rules`` and add no cell.

    ``routed`` is the ``policy_rules.jsonl`` rows. When it is present, an eligible rule that
    the file does not list as eligible stops the plan.
    """
    eligible = sorted(
        (rule for rule in rules if rule.status == "eligible_for_generation"),
        key=lambda rule: rule.policy_id,
    )
    excluded = _excluded_rules(rules, routed, {rule.policy_id for rule in eligible})
    by_id = {rule.policy_id: rule for rule in eligible}
    required: list[dict[str, Any]] = []
    optional: list[dict[str, Any]] = []
    families: list[dict[str, Any]] = []
    planned: dict[str, dict[str, Any]] = {}
    max_turns = int(budget["max_turns"])
    for rule in eligible:
        _require_source(rule, spans)
        languages = _declared_languages(rule)
        primary = languages[0] if languages else None
        translation = _translation_languages(languages, target_languages)
        rule_required = _required_cells(rule, by_id, translation, primary, max_turns)
        missing_span = (
            not isinstance(rule.provenance.source_span_id, str) or not rule.provenance.source_span_id.strip()
        )
        if missing_span:
            for cell in rule_required:
                cell["hold"] = "span_missing"
            rule_optional = []
        else:
            rule_optional = _optional_cells(
                rule,
                by_id,
                primary,
                modules,
                max_turns=max_turns,
                mix_multi_turn=_wants_mix_multi_turn(generation) and rule.signals.multi_turn_turns is None,
            )
        required.extend(rule_required)
        optional.extend(rule_optional)
        for cell in (*rule_required, *rule_optional):
            planned[cell["slot_id"]] = cell
        families.extend(_contrast_families(rule, _paired_allow(rule, by_id), planned))
    required.sort(key=_sort_key)
    optional.sort(key=_sort_key)
    rows, gaps = _schedule(required, optional, case_cap, budget, generation)
    scheduled = {row["slot_id"] for row in rows}
    marked = tuple(_with_scheduled(family, scheduled) for family in families)
    blocking = [gap for gap in gaps if gap["blocks_gold"]]
    certification, reason = _certification(len(required), blocking)
    capacity = _capacity(rows, gaps, case_cap, budget)
    return CoveragePlan(
        rows=tuple(rows),
        gaps=tuple(gaps),
        contrast_families=marked,
        certification=certification,
        reason=reason,
        case_cap=case_cap,
        required_cases=len(required),
        excluded_rules=excluded,
        modules=_module_status(modules),
        capacity=capacity,
        quality_gate_preflight=_quality_gate_preflight(capacity, quality_gates),
    )


def write_coverage_plan(stage_cache: Path, plan: CoveragePlan) -> dict[str, Any]:
    """Write the scheduled rows and the gap report. Returns the run-record summary."""
    write_jsonl(stage_cache / PLAN_FILE, plan.rows)
    summary = plan.summary()
    write_json(stage_cache / GAPS_FILE, summary)
    return plan.record()


def cell_distinctness_key(cell: Mapping[str, Any]) -> tuple[Any, ...]:
    """Identity used to refuse quote-copy backfill. Turn text is implied by the rule quote."""
    language = cell.get("language")
    if not isinstance(language, str) or not language.strip():
        listed = cell.get("languages")
        if isinstance(listed, list) and listed and all(isinstance(item, str) for item in listed):
            language = ",".join(item.strip() for item in listed if item.strip())
        else:
            language = cell.get("source_language") if isinstance(cell.get("source_language"), str) else ""
    present = cell.get("present_legs") if isinstance(cell.get("present_legs"), list) else []
    absent = cell.get("absent_legs") if isinstance(cell.get("absent_legs"), list) else []
    action = cell.get("expected_action")
    return (
        str(cell.get("policy_id") or ""),
        str(language or ""),
        "" if action is None else str(action),
        str(cell.get("kind") or ""),
        tuple(str(leg) for leg in present),
        tuple(str(leg) for leg in absent),
    )


def select_backfill_cells(
    gaps: Sequence[Mapping[str, Any]],
    scheduled: Sequence[Mapping[str, Any]],
    surviving: Sequence[Mapping[str, Any]],
    *,
    case_cap: int,
    budget: Mapping[str, Any],
    gate_result: Mapping[str, Any],
    blocked_policies: Collection[str] = (),
) -> list[dict[str, Any]]:
    """Choose unscheduled distinct cells that can fill quality-gate shortfalls.

    Held translation/span/turn gaps cannot be generated. A cell whose distinctness
    key already exists in the surviving dataset or the current plan is a quote-copy
    and is skipped. A cell of a policy in ``blocked_policies`` would be rejected
    like that policy's earlier cases and is skipped. The case cap is applied to
    surviving rows, so rejected plan rows can be replaced.
    """
    blocked = {policy_id for policy_id in blocked_policies if isinstance(policy_id, str)}
    failures = gate_result.get("failures") if isinstance(gate_result.get("failures"), list) else []
    if not failures:
        return []
    occupied = {cell_distinctness_key(row) for row in (*scheduled, *surviving)}
    scheduled_ids = {
        str(row.get("slot_id"))
        for row in scheduled
        if isinstance(row.get("slot_id"), str) and row.get("slot_id")
    }
    policy_counts: dict[str, int] = {}
    for row in surviving:
        policy_id = row.get("policy_id")
        if isinstance(policy_id, str) and policy_id:
            policy_counts[policy_id] = policy_counts.get(policy_id, 0) + 1
    room = int(case_cap) - len(surviving)
    if room <= 0:
        return []
    ranked: list[tuple[int, int, str, dict[str, Any]]] = []
    for gap in gaps:
        if not isinstance(gap, Mapping):
            continue
        slot_id = gap.get("slot_id")
        if not isinstance(slot_id, str) or not slot_id or slot_id in scheduled_ids:
            continue
        reason = str(gap.get("reason") or "")
        if reason in _UNGENERATABLE_GAP_REASONS:
            continue
        if str(gap.get("policy_id") or "") in blocked:
            continue
        key = cell_distinctness_key(gap)
        if key in occupied:
            continue
        score = _backfill_score(gap, failures)
        if score <= 0:
            continue
        required = 0 if gap.get("priority") == "optional" else 1
        ranked.append((required, score, slot_id, dict(gap)))
    ranked.sort(
        key=lambda item: (
            -item[0],
            -item[1],
            hashlib.sha256(item[2].encode("utf-8")).hexdigest(),
            item[2],
        )
    )
    chosen: list[dict[str, Any]] = []
    for _required, _score, slot_id, cell in ranked:
        if len(chosen) >= room:
            break
        policy_id = str(cell.get("policy_id") or "")
        if policy_id:
            if policy_counts.get(policy_id, 0) >= _rule_limit(cell, budget):
                continue
        key = cell_distinctness_key(cell)
        if key in occupied:
            continue
        chosen.append(cell)
        occupied.add(key)
        scheduled_ids.add(slot_id)
        if policy_id:
            policy_counts[policy_id] = policy_counts.get(policy_id, 0) + 1
    return chosen


def apply_backfill(
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    added: Sequence[Mapping[str, Any]],
    *,
    case_cap: int,
    budget: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Move chosen gap cells onto the plan and refresh gap/family bookkeeping."""
    added_ids = {str(cell["slot_id"]) for cell in added if isinstance(cell.get("slot_id"), str)}
    new_rows = [dict(row) for row in rows]
    for cell in added:
        stored = {key: value for key, value in cell.items() if key not in {"reason", "blocks_gold"}}
        stored["order"] = len(new_rows)
        stored["backfill"] = True
        new_rows.append(stored)
    remaining = [
        dict(gap)
        for gap in summary.get("gaps") or []
        if isinstance(gap, Mapping) and str(gap.get("slot_id")) not in added_ids
    ]
    scheduled_ids = {str(row.get("slot_id")) for row in new_rows if isinstance(row.get("slot_id"), str)}
    families = []
    for family in summary.get("contrast_families") or []:
        if not isinstance(family, Mapping):
            continue
        members = []
        for member in family.get("members") or []:
            if not isinstance(member, Mapping):
                continue
            members.append({**member, "scheduled": member.get("slot_id") in scheduled_ids})
        families.append({**family, "members": members})
    blocking = [gap for gap in remaining if gap.get("blocks_gold") is True]
    required_cases = summary.get("required_cases")
    required_count = required_cases if isinstance(required_cases, int) and not isinstance(required_cases, bool) else 0
    certification, reason = _certification(required_count, blocking)
    added_slots = [str(cell["slot_id"]) for cell in added if isinstance(cell.get("slot_id"), str)]
    previous = summary.get("backfill") if isinstance(summary.get("backfill"), Mapping) else {}
    previous_added = [str(slot) for slot in previous.get("added_slot_ids") or [] if isinstance(slot, str)]
    capacity = _capacity(new_rows, remaining, case_cap, budget)
    requirements = _preflight_requirements(summary)
    updated = dict(summary)
    updated.update(
        {
            "certification": certification,
            "reason": reason,
            "scheduled_cases": len(new_rows),
            "optional_scheduled": sum(1 for row in new_rows if row.get("priority") == "optional"),
            "missing_required": [str(gap["slot_id"]) for gap in blocking if isinstance(gap.get("slot_id"), str)],
            "gaps": remaining,
            "contrast_families": families,
            "capacity": capacity,
            "quality_gate_preflight": _quality_gate_preflight(capacity, requirements),
            "backfill": {
                "rounds": (
                    previous.get("rounds") + 1
                    if isinstance(previous.get("rounds"), int) and not isinstance(previous.get("rounds"), bool)
                    else 1
                ),
                "added_slot_ids": previous_added + added_slots,
            },
        }
    )
    return new_rows, updated


def rewrite_coverage_plan(
    stage_cache: Path, rows: Sequence[Mapping[str, Any]], summary: Mapping[str, Any]
) -> dict[str, Any]:
    """Replace the plan files after backfill. Gap rows stay in the gap file."""
    stored_rows = [dict(row) for row in rows]
    stored_summary = dict(summary)
    write_jsonl(stage_cache / PLAN_FILE, stored_rows)
    write_json(stage_cache / GAPS_FILE, stored_summary)
    record = {key: stored_summary[key] for key in _RECORD_FIELDS}
    if isinstance(stored_summary.get("backfill"), Mapping):
        record["backfill"] = dict(stored_summary["backfill"])
    return record


def rule_level_rejected_policies(rejected: Sequence[Mapping[str, Any]]) -> set[str]:
    """Policies with a case rejected because of the rule itself, not because of that case."""
    blocked: set[str] = set()
    for row in rejected:
        policy_id = row.get("policy_id")
        reasons = row.get("reasons")
        if not isinstance(policy_id, str) or not isinstance(reasons, list):
            continue
        if any(reason in RULE_LEVEL_REJECTIONS for reason in reasons):
            blocked.add(policy_id)
    return blocked


def quality_gate_shortfall(gate_result: Mapping[str, Any]) -> int:
    """Total missing cases across unmet gates. Backfill must lower it to be kept."""
    failures = gate_result.get("failures") if isinstance(gate_result.get("failures"), list) else []
    total = 0
    for failure in failures:
        missing = failure.get("missing") if isinstance(failure, Mapping) else None
        if isinstance(missing, int) and not isinstance(missing, bool) and missing > 0:
            total += missing
    return total


def _backfill_score(cell: Mapping[str, Any], failures: Sequence[Mapping[str, Any]]) -> int:
    policy_id = str(cell.get("policy_id") or "")
    action = cell.get("expected_action")
    tags = set()
    language = cell.get("language")
    if isinstance(language, str) and language.strip():
        tags.add(language.casefold())
    source = cell.get("source_language")
    if isinstance(source, str) and source.strip():
        tags.add(source.casefold())
    listed = cell.get("languages")
    if isinstance(listed, list):
        tags.update(item.casefold() for item in listed if isinstance(item, str) and item.strip())
    score = 0
    for failure in failures:
        if not isinstance(failure, Mapping):
            continue
        gate = failure.get("gate")
        slice_name = failure.get("slice")
        if gate in {"min_total_cases", "min_development_cases"}:
            score += 1
        elif gate == "min_cases_per_policy" and isinstance(slice_name, str) and slice_name == policy_id:
            score += 3
        elif gate == "min_cases_per_action" and isinstance(slice_name, str) and slice_name == action:
            score += 3
        elif (
            gate == "min_cases_per_language"
            and isinstance(slice_name, str)
            and slice_name.casefold() in tags
        ):
            score += 3
    return score


def _required_cells(
    rule: PolicyRule,
    by_id: Mapping[str, PolicyRule],
    translation_languages: Sequence[str],
    primary: str | None,
    max_turns: int,
) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    rules = tuple(by_id[policy_id] for policy_id in sorted(by_id))
    _partner_action, partner, allow_flags = _allow_case(rule, by_id)
    if primary_action(rule) is not None:
        cells.append(_planned(rule, rules, "clear_violation", primary=primary))
    cells.extend(_all_leg_cells(rule, rules, primary))
    cells.extend(_condition_path_cells(rule, rules, primary))
    cells.append(_planned(rule, rules, "clear_allow", primary=primary, flags=allow_flags, paired_policy_id=partner))
    if rule.decision.ambiguous_action is not None:
        cells.append(
            _planned(
                rule,
                rules,
                "boundary",
                primary=primary,
            )
        )
    if rule.exceptions:
        cells.append(_planned(rule, rules, "exception", primary=primary, exceptions=list(rule.exceptions)))
    cells.extend(_missing_legs(rule, rules, primary, allow_flags, partner))
    declared = _declared_languages(rule)
    for language in translation_languages:
        cells.append(
            _planned(
                rule,
                rules,
                "language",
                primary=primary,
                language=language,
                slot_id=_slot(rule.policy_id, "language", language),
                hold="translation_required",
            )
        )
    switched = _code_switch_languages(rule, declared)
    if switched is not None:
        cells.append(
            _planned(
                rule,
                rules,
                "code_switch",
                primary=primary,
                languages=switched,
                hold="translation_required",
            )
        )
    multi_turn = _multi_turn(rule, rules, primary, max_turns)
    if multi_turn is not None:
        cells.append(multi_turn)
    return cells


def _all_leg_cells(rule: PolicyRule, rules: Sequence[PolicyRule], primary: str | None) -> list[dict[str, Any]]:
    """One block case per ``all_of`` group that is not already the clear violation.

    The clear violation keeps the strongest action. A second group needs its own case,
    so its contrast pair can remove exactly one of its own legs.
    """
    cells: list[dict[str, Any]] = []
    for name, group_action, group in acting_groups(rule):
        if group.operator != "all_of" or len(group.legs) < 2:
            continue
        if _block_slot(rule, name, group, group_action) == _slot(rule.policy_id, "clear_violation"):
            continue
        cells.append(
            _planned(
                rule,
                rules,
                "all_legs",
                primary=primary,
                slot_id=_slot(rule.policy_id, "all_legs", name),
                condition_class=name,
            )
        )
    return cells


def _condition_path_cells(
    rule: PolicyRule, rules: Sequence[PolicyRule], primary: str | None
) -> list[dict[str, Any]]:
    """One minimal case for every any-of leg and every none-of condition class."""
    cells: list[dict[str, Any]] = []
    clear = case_context(rules, rule, "clear_violation")
    for name, _group_action, group in acting_groups(rule):
        if group.operator == "all_of":
            continue
        triggers: Sequence[str | None] = group.legs if group.operator == "any_of" else (None,)
        for trigger in triggers:
            extra: dict[str, Any] = {"condition_class": name}
            parts = [name]
            if trigger is not None:
                extra["trigger_leg"] = trigger
                parts.append(trigger)
            context = case_context(rules, rule, "condition_path", trigger_leg=trigger, condition_class=name)
            if context == clear:
                continue
            cells.append(
                _planned(
                    rule,
                    rules,
                    "condition_path",
                    primary=primary,
                    slot_id=_slot(rule.policy_id, "condition_path", *parts),
                    **extra,
                )
            )
    return cells


def _missing_legs(
    rule: PolicyRule,
    rules: Sequence[PolicyRule],
    primary: str | None,
    allow_flags: Sequence[str],
    partner: str | None,
) -> list[dict[str, Any]]:
    """One case per leg of an ``all_of`` group. ``any_of`` adds none, including the default."""
    cells: list[dict[str, Any]] = []
    for name, _group_action, group in acting_groups(rule):
        if group.operator != "all_of" or len(group.legs) < 2:
            continue
        for absent in group.legs:
            cells.append(
                _planned(
                    rule,
                    rules,
                    "missing_leg",
                    primary=primary,
                    flags=allow_flags,
                    slot_id=_slot(rule.policy_id, "missing_leg", name, absent),
                    condition_class=name,
                    absent_leg=absent,
                    paired_policy_id=partner,
                )
            )
    return cells


def _multi_turn(
    rule: PolicyRule, rules: Sequence[PolicyRule], primary: str | None, max_turns: int
) -> dict[str, Any] | None:
    source_turns = rule.signals.multi_turn_turns
    if source_turns is None:
        return None
    cell = _planned(rule, rules, "multi_turn", primary=primary, source_turns=source_turns)
    # A multi-turn case needs two turns. A lower cap cannot be filled by a one-turn case.
    if max_turns < 2:
        cell["hold"] = "max_turns"
        return cell
    cell["turns"] = min(source_turns, max_turns)
    if source_turns > max_turns:
        cell.setdefault("flags", []).append("turns_capped")
    return cell


def _optional_cells(
    rule: PolicyRule,
    by_id: Mapping[str, PolicyRule],
    primary: str | None,
    modules: Mapping[str, Any],
    *,
    max_turns: int,
    mix_multi_turn: bool,
) -> list[dict[str, Any]]:
    rules = tuple(by_id[policy_id] for policy_id in sorted(by_id))
    cells: list[dict[str, Any]] = []
    preserved = ("preserves_label",)
    if mix_multi_turn and primary_action(rule) is not None and max_turns >= 2:
        cells.append(
            _planned(
                rule,
                rules,
                "multi_turn",
                primary=primary,
                priority="optional",
                flags=("mix_turns", *preserved),
                turns=min(2, max_turns),
            )
        )
    if modules.get("adversarial") is True and primary_action(rule) is not None:
        cells.append(_planned(rule, rules, "adversarial", primary=primary, priority="optional", flags=preserved))
    if modules.get("political_symmetry") is True:
        cells.append(_planned(rule, rules, "actor_swap", primary=primary, priority="optional", flags=preserved))
    if modules.get("persona") is True:
        cells.append(_planned(rule, rules, "persona", primary=primary, priority="optional", flags=preserved))
    return cells


def _wants_mix_multi_turn(generation: Mapping[str, Any] | None) -> bool:
    """Optional multi-turn cells exist only when the configured mix asks for some."""
    if not isinstance(generation, Mapping):
        return False
    mix = generation.get("turn_mix")
    if not isinstance(mix, Mapping):
        return False
    share = mix.get("multi_turn")
    return isinstance(share, (int, float)) and not isinstance(share, bool) and share > 0


def _contrast_families(
    rule: PolicyRule, paired_allow: bool, planned: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    if primary_action(rule) is None:
        return []
    violation = _slot(rule.policy_id, "clear_violation")
    families = []
    for name, group_action, group in acting_groups(rule):
        if group.operator != "all_of" or len(group.legs) < 2:
            continue
        block_slot = _block_slot(rule, name, group, group_action)
        for absent in group.legs:
            allow_slot = _slot(rule.policy_id, "missing_leg", name, absent)
            families.append(
                {
                    "family_id": _family_id(rule.policy_id, name, absent),
                    "policy_id": rule.policy_id,
                    "changed_dimension": absent,
                    "members": [
                        {
                            "slot_id": block_slot,
                            "role": "all_legs",
                            "expected_action": planned[block_slot]["expected_action"],
                        },
                        {
                            "slot_id": allow_slot,
                            "role": "missing_leg",
                            "expected_action": planned[allow_slot]["expected_action"],
                        },
                    ],
                }
            )
    if families:
        return families
    if paired_allow:
        dimension = "paired_allow"
    elif rule.paired_with:
        dimension = "paired_rule"
    else:
        dimension = "benign_neighbor"
    return [
        {
            "family_id": _family_id(rule.policy_id),
            "policy_id": rule.policy_id,
            "changed_dimension": dimension,
            "members": [
                {
                    "slot_id": violation,
                    "role": "violation",
                    "expected_action": planned[violation]["expected_action"],
                },
                {
                    "slot_id": _slot(rule.policy_id, "clear_allow"),
                    "role": "allow",
                    "expected_action": planned[_slot(rule.policy_id, "clear_allow")]["expected_action"],
                },
            ],
        }
    ]


def _schedule(
    required: Sequence[dict[str, Any]],
    optional: Sequence[dict[str, Any]],
    case_cap: int,
    budget: Mapping[str, Any],
    generation: Mapping[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    gaps: list[dict[str, Any]] = []
    used: dict[str, int] = {}

    def place(cell: dict[str, Any], *, blocks_gold: bool) -> None:
        stored = {key: value for key, value in cell.items() if key != "hold"}
        hold = cell.get("hold")
        if hold:
            gaps.append({**stored, "reason": hold, "blocks_gold": True})
            return
        policy_id = str(cell["policy_id"])
        if used.get(policy_id, 0) >= _rule_limit(cell, budget):
            gaps.append({**stored, "reason": "rule_cap", "blocks_gold": blocks_gold})
            return
        if len(rows) >= case_cap:
            gaps.append({**stored, "reason": "budget_cap", "blocks_gold": blocks_gold})
            return
        stored["order"] = len(rows)
        rows.append(stored)
        used[policy_id] = used.get(policy_id, 0) + 1

    for cell in _fair_order(required, generation):
        place(cell, blocks_gold=True)
    for cell in _fair_order(optional, generation):
        place(cell, blocks_gold=False)
    return rows, gaps


def _fair_order(
    cells: Sequence[dict[str, Any]], generation: Mapping[str, Any] | None
) -> list[dict[str, Any]]:
    """Give every policy an anchor, then approach configured mixes without lexical bias."""
    queues: dict[str, list[dict[str, Any]]] = {}
    for cell in cells:
        queues.setdefault(str(cell["policy_id"]), []).append(cell)
    for queue in queues.values():
        queue.sort(key=_sort_key)
    policies = sorted(
        queues,
        key=lambda policy_id: (hashlib.sha256(policy_id.encode("utf-8")).hexdigest(), policy_id),
    )
    ordered = [queues[policy_id].pop(0) for policy_id in policies if queues[policy_id]]
    counts = Counter(str(cell["policy_id"]) for cell in ordered)
    remaining = [cell for policy_id in policies for cell in queues[policy_id]]
    while remaining:
        lowest = min(counts[str(cell["policy_id"])] for cell in remaining)
        candidates = [cell for cell in remaining if counts[str(cell["policy_id"])] == lowest]
        chosen = min(
            candidates,
            key=lambda cell: (
                _mix_error([*ordered, cell], generation),
                hashlib.sha256(str(cell["slot_id"]).encode("utf-8")).hexdigest(),
                str(cell["slot_id"]),
            ),
        )
        remaining.remove(chosen)
        ordered.append(chosen)
        counts[str(chosen["policy_id"])] += 1
    return ordered


def _mix_error(cells: Sequence[Mapping[str, Any]], generation: Mapping[str, Any] | None) -> float:
    if generation is None:
        return 0.0
    total_error = 0.0
    for mix_name in ("decision_mix", "case_family_mix", "turn_mix"):
        target = generation.get(mix_name)
        if not isinstance(target, Mapping):
            continue
        observed = Counter(
            bucket
            for cell in cells
            if (bucket := _mix_bucket(cell, mix_name)) is not None and bucket in target
        )
        total = sum(observed.values())
        if not total:
            continue
        total_error += sum(abs(observed.get(str(name), 0) - float(share) * total) for name, share in target.items())
    return total_error


def _mix_bucket(cell: Mapping[str, Any], mix_name: str) -> str | None:
    if mix_name == "decision_mix":
        action = cell.get("expected_action")
        return action if isinstance(action, str) else None
    if mix_name == "turn_mix":
        return "multi_turn" if cell.get("kind") == "multi_turn" else "single_turn"
    kind = cell.get("kind")
    if kind in {"clear_violation", "clear_allow"}:
        return "clear"
    if kind == "boundary":
        return "boundary"
    if kind == "exception":
        return "exception"
    if kind in {"all_legs", "condition_path", "missing_leg"}:
        return "contrast"
    return None


def _certification(required_cases: int, blocking: Sequence[dict[str, Any]]) -> tuple[str, str]:
    if required_cases == 0:
        return "silver", "no_eligible_rules"
    if not blocking:
        return "gold", "complete"
    reasons = {str(gap["reason"]) for gap in blocking}
    if len(reasons) == 1:
        return "silver", reasons.pop()
    return "silver", "required_omitted"


def _with_scheduled(family: dict[str, Any], scheduled: set[str]) -> dict[str, Any]:
    members = [{**member, "scheduled": member["slot_id"] in scheduled} for member in family["members"]]
    return {**family, "members": members}


def _block_slot(rule: PolicyRule, name: str, group: ConditionGroup, group_action: str) -> str:
    """The case that contains exactly this group's legs.

    The clear violation is that case only when its own legs are this group and nothing else.
    Legs added so a different ``none_of`` group stays quiet do not count.
    """
    primary = primary_action(rule)
    present, absent = violation_facts(rule, primary) if primary is not None else ([], [])
    if primary == group_action and not absent and present == list(group.legs):
        return _slot(rule.policy_id, "clear_violation")
    return _slot(rule.policy_id, "all_legs", name)


def _allow_case(rule: PolicyRule, by_id: Mapping[str, PolicyRule]) -> tuple[str, str | None, tuple[str, ...]]:
    partner = by_id.get(rule.paired_with) if rule.paired_with else None
    if partner is not None and partner.rule.allow_if is not None:
        return "allow", partner.policy_id, ()
    if rule.rule.allow_if is not None:
        return "allow", None, ()
    if rule.paired_with and partner is None:
        return "allow", rule.paired_with, ("paired_rule_not_eligible",)
    if partner is not None:
        return "allow", partner.policy_id, ("paired_rule_not_allow",)
    return "allow", None, ("allow_unpaired",)


def _paired_allow(rule: PolicyRule, by_id: Mapping[str, PolicyRule]) -> bool:
    partner = by_id.get(rule.paired_with) if rule.paired_with else None
    return partner is not None and partner.rule.allow_if is not None


def _declared_languages(rule: PolicyRule) -> list[str]:
    """Languages the rule itself lists. Text is not scanned for a language."""
    declared: list[str] = []
    for language in rule.scope.languages:
        if language.casefold() != "unspecified" and language not in declared:
            declared.append(language)
    return declared


def _translation_languages(declared: Sequence[str], targets: Sequence[str]) -> list[str]:
    """Languages that still need a translation, without treating them as clause languages."""
    seen = {declared[0].casefold()} if declared else set()
    chosen: list[str] = []
    for language in (*declared[1:], *targets):
        if not isinstance(language, str):
            continue
        text = language.strip()
        key = text.casefold()
        if not text or key == "unspecified" or key in seen:
            continue
        seen.add(key)
        chosen.append(text)
    return chosen


def _code_switch_languages(rule: PolicyRule, languages: Sequence[str]) -> list[str] | None:
    """Declared languages, when grounded evidence says this rule is code-switched."""
    if not rule.signals.code_switch or len(languages) < 2:
        return None
    return list(languages)


def _require_source(rule: PolicyRule, spans: Mapping[str, str]) -> None:
    span_id = rule.provenance.source_span_id
    if span_id is not None and span_id not in spans:
        raise CoverageInputError(f"{rule.policy_id} cites source span {span_id}, which is not in the source index.")


def _rule_cap(severity: str, budget: Mapping[str, Any]) -> int:
    key = "cases_per_critical_rule" if severity == "critical" else "cases_per_normal_rule"
    return int(budget[key])


def _rule_limit(cell: Mapping[str, Any], budget: Mapping[str, Any]) -> int:
    """Every cell, including a mix-derived cell, obeys the per-rule maximum."""
    return _rule_cap(str(cell.get("severity") or "medium"), budget)


def _capacity(
    rows: Sequence[Mapping[str, Any]],
    gaps: Sequence[Mapping[str, Any]],
    case_cap: int,
    budget: Mapping[str, Any],
) -> dict[str, Any]:
    """Conservative final-count upper bounds from grounded inventory."""
    inventory = [dict(row) for row in rows]
    occupied = {cell_distinctness_key(row) for row in rows}
    distinct_gaps = 0
    for gap in gaps:
        if str(gap.get("reason") or "") in _UNGENERATABLE_GAP_REASONS:
            continue
        key = cell_distinctness_key(gap)
        if key in occupied:
            continue
        occupied.add(key)
        inventory.append(dict(gap))
        distinct_gaps += 1

    by_policy_inventory: dict[str, int] = {}
    policy_caps: dict[str, int] = {}
    by_action_inventory: dict[str, int] = {}
    by_language_inventory: dict[str, int] = {}
    for cell in inventory:
        policy_id = str(cell.get("policy_id") or "")
        if policy_id:
            by_policy_inventory[policy_id] = by_policy_inventory.get(policy_id, 0) + 1
            policy_caps[policy_id] = _rule_limit(cell, budget)
        action = cell.get("expected_action")
        action_name = "unlabeled" if action is None else str(action)
        by_action_inventory[action_name] = by_action_inventory.get(action_name, 0) + 1
        for language in _cell_languages(cell):
            by_language_inventory[language] = by_language_inventory.get(language, 0) + 1

    by_policy = {
        policy_id: min(count, policy_caps[policy_id], case_cap)
        for policy_id, count in sorted(by_policy_inventory.items())
    }
    final_upper_bound = min(case_cap, sum(by_policy.values()))
    return {
        "case_cap": case_cap,
        "planned_candidates": len(rows),
        "additional_distinct_cells": distinct_gaps,
        "final_case_upper_bound": final_upper_bound,
        "by_policy": by_policy,
        "by_action": {
            action: min(count, final_upper_bound)
            for action, count in sorted(by_action_inventory.items())
        },
        "by_language": {
            language: min(count, final_upper_bound)
            for language, count in sorted(by_language_inventory.items())
        },
    }


def _cell_languages(cell: Mapping[str, Any]) -> tuple[str, ...]:
    languages: list[str] = []
    listed = cell.get("languages")
    if isinstance(listed, list):
        languages.extend(item for item in listed if isinstance(item, str) and item.strip())
    for key in ("language", "source_language"):
        language = cell.get(key)
        if isinstance(language, str) and language.strip():
            languages.append(language)
    return tuple(dict.fromkeys(item.casefold() for item in languages))


def _quality_gate_preflight(
    capacity: Mapping[str, Any], requirements: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Prove only inventory-level impossibilities; final gates run after split."""
    copied = dict(requirements) if isinstance(requirements, Mapping) else {}
    failures: list[dict[str, Any]] = []
    upper = int(capacity.get("final_case_upper_bound") or 0)

    def fail(gate: str, required: int, available: int, slice_name: str | None = None) -> None:
        if required <= available:
            return
        row: dict[str, Any] = {
            "gate": gate,
            "required": required,
            "available_upper_bound": available,
            "missing": required - available,
        }
        if slice_name is not None:
            row["slice"] = slice_name
        failures.append(row)

    min_total = _preflight_count(copied, "min_total_cases")
    min_development = _preflight_count(copied, "min_development_cases")
    fail("min_total_cases", max(min_total, min_development), upper)

    per_policy = _preflight_count(copied, "min_cases_per_policy")
    by_policy = capacity.get("by_policy") if isinstance(capacity.get("by_policy"), Mapping) else {}
    for policy_id, available in sorted(by_policy.items()):
        if isinstance(available, int) and not isinstance(available, bool):
            fail("min_cases_per_policy", per_policy, available, str(policy_id))

    for gate, capacity_key in (
        ("min_cases_per_action", "by_action"),
        ("min_cases_per_language", "by_language"),
    ):
        minima = copied.get(gate)
        available_by_slice = (
            capacity.get(capacity_key) if isinstance(capacity.get(capacity_key), Mapping) else {}
        )
        if not isinstance(minima, Mapping):
            continue
        for name, required in minima.items():
            if isinstance(required, bool) or not isinstance(required, int):
                continue
            key = str(name).casefold()
            available = available_by_slice.get(key, 0)
            if isinstance(available, int) and not isinstance(available, bool):
                fail(gate, required, available, str(name))

    failures.sort(key=lambda row: (str(row["gate"]), str(row.get("slice") or "")))
    return {
        "enabled": _preflight_enabled(copied),
        "possible": not failures,
        "requirements": copied,
        "failures": failures,
    }


def _preflight_count(requirements: Mapping[str, Any], key: str) -> int:
    value = requirements.get(key, 0)
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _preflight_enabled(requirements: Mapping[str, Any]) -> bool:
    if any(
        _preflight_count(requirements, key) > 0
        for key in (
            "min_total_cases",
            "min_development_cases",
            "min_cases_per_policy",
        )
    ):
        return True
    for key in ("min_cases_per_action", "min_cases_per_language"):
        minima = requirements.get(key)
        if isinstance(minima, Mapping) and any(
            isinstance(value, int) and not isinstance(value, bool) and value > 0
            for value in minima.values()
        ):
            return True
    return False


def _preflight_requirements(summary: Mapping[str, Any]) -> Mapping[str, Any] | None:
    preflight = summary.get("quality_gate_preflight")
    if not isinstance(preflight, Mapping):
        return None
    requirements = preflight.get("requirements")
    return requirements if isinstance(requirements, Mapping) else None


def _module_status(modules: Mapping[str, Any]) -> dict[str, str]:
    symmetry = modules.get("political_symmetry")
    if symmetry is True:
        symmetry_status = "enabled"
    elif symmetry is False:
        symmetry_status = "disabled"
    elif symmetry == "confirm_if_detected":
        symmetry_status = "confirm_if_detected"
    else:
        symmetry_status = "disabled"
    return {
        "adversarial": "enabled" if modules.get("adversarial") is True else "disabled",
        "political_symmetry": symmetry_status,
        "persona": "enabled" if modules.get("persona") is True else "disabled",
        "paraphrase": "enabled" if modules.get("paraphrase") is True else "disabled",
        "production_replay": "not_configured",
    }


def _planned(
    rule: PolicyRule,
    rules: Sequence[PolicyRule],
    kind: str,
    *,
    primary: str | None,
    priority: str = "required",
    flags: Sequence[str] = (),
    slot_id: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """One cell whose legs and action come from the aggregated facts."""
    context = case_context(
        rules,
        rule,
        kind,
        condition_class=extra.get("condition_class") if isinstance(extra.get("condition_class"), str) else None,
        absent_leg=extra.get("absent_leg") if isinstance(extra.get("absent_leg"), str) else None,
        trigger_leg=extra.get("trigger_leg") if isinstance(extra.get("trigger_leg"), str) else None,
    )
    if context is None:
        raise CoverageInputError(f"{rule.policy_id} {kind} does not match its rule.")
    present, absent, action = context
    if kind != "boundary" and action is None:
        raise CoverageInputError(f"{rule.policy_id} {kind} has no action its facts support.")
    return _cell(
        rule,
        kind,
        action,
        primary=primary,
        priority=priority,
        flags=flags,
        slot_id=slot_id,
        present_legs=present,
        absent_legs=absent,
        **extra,
    )


def _cell(
    rule: PolicyRule,
    kind: str,
    expected: str | None,
    *,
    primary: str | None,
    priority: str = "required",
    flags: Sequence[str] = (),
    slot_id: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "slot_id": slot_id or _slot(rule.policy_id, kind),
        "policy_id": rule.policy_id,
        "priority": priority,
        "kind": kind,
        "expected_action": expected,
        "severity": rule.severity,
        "source_span_id": rule.provenance.source_span_id,
        "source_language": primary,
    }
    if flags:
        row["flags"] = list(flags)
    row.update(extra)
    return {key: value for key, value in row.items() if value is not None or key == "expected_action"}


def _family_id(policy_id: str, *parts: str) -> str:
    return "contrast/" + "/".join(_slot_part(part) for part in (policy_id, *parts))


def _slot(policy_id: str, kind: str, *parts: str) -> str:
    """Join the parts with ``/``. A ``/`` or ``\\`` inside a part is escaped, so the id is not split on ``:``."""
    return "/".join(_slot_part(part) for part in (policy_id, kind, *parts))


def _slot_part(value: str) -> str:
    return value.replace("\\", "\\\\").replace("/", "\\/")


def _sort_key(cell: dict[str, Any]) -> tuple[int, str, str]:
    return (_KIND_RANK[str(cell["kind"])], str(cell["policy_id"]), str(cell["slot_id"]))


def _excluded_rules(
    rules: Sequence[PolicyRule], routed: Sequence[Mapping[str, Any]] | None, eligible_ids: set[str]
) -> tuple[dict[str, str], ...]:
    if routed is not None:
        statuses = _routed_statuses(routed)
        for policy_id in sorted(eligible_ids):
            if statuses.get(policy_id) != "eligible_for_generation":
                raise CoverageInputError(
                    f"{policy_id} is in the eligible rules but policy_rules.jsonl does not list it as eligible."
                )
        excluded = {policy_id: status for policy_id, status in statuses.items() if status != "eligible_for_generation"}
    else:
        excluded = {rule.policy_id: rule.status for rule in rules if rule.status != "eligible_for_generation"}
    return tuple({"policy_id": policy_id, "status": excluded[policy_id]} for policy_id in sorted(excluded))


def _routed_statuses(rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for number, row in enumerate(rows, start=1):
        policy_id = row.get("policy_id")
        status = row.get("status")
        if not isinstance(policy_id, str) or not policy_id or not isinstance(status, str) or not status:
            raise CoverageInputError(f"policy_rules.jsonl row {number} needs a policy_id and a status.")
        if policy_id in statuses:
            raise CoverageInputError(f"policy_rules.jsonl repeats {policy_id}.")
        statuses[policy_id] = status
    return statuses
