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
cell that does not fit is listed in the gap file, and the plan is Silver.
"""

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nemotron.steps.byob.runtime.benchmark_families.safety.actions import ACTION_ORDER
from nemotron.steps.byob.runtime.benchmark_families.safety.artifacts import write_json, write_jsonl
from nemotron.steps.byob.runtime.benchmark_families.safety.ir import (
    ConditionGroup,
    PolicyRule,
    RuleValidationError,
    validate_rule,
)

PLAN_FILE = "coverage_plan.jsonl"
GAPS_FILE = "coverage_gaps.json"
COVERAGE_OUTPUTS = (PLAN_FILE, GAPS_FILE)

# Required cells are filled in this order across every rule, so a short budget keeps a
# clear violation for each rule before any rule receives a later cell. Optional cells are
# a separate pass after every required cell.
_REQUIRED_KIND_ORDER = (
    "clear_violation",
    "clear_allow",
    "boundary",
    "exception",
    "missing_leg",
    "language",
    "code_switch",
    "multi_turn",
)
_OPTIONAL_KIND_ORDER = ("adversarial", "actor_swap", "persona")
_KIND_RANK = {kind: rank for rank, kind in enumerate(_REQUIRED_KIND_ORDER + _OPTIONAL_KIND_ORDER)}
_ACTING = (
    ("escalate_if", "escalate"),
    ("block_if", "block"),
    ("require_confirmation_if", "require_confirmation"),
)
_STEP = re.compile(
    r"(?m)^[ \t]*(?:\d{1,2}[.)]|bước[ \t]+\d+|step[ \t]+\d+)[ \t]+\S",
    re.IGNORECASE,
)
_SEQUENCE_PHRASES = (
    "multi-turn",
    "multi turn",
    "nhiều lượt",
    "nhiều bước",
    "numbered steps",
    "trong hội thoại",
    "in a dialogue",
    "in the dialogue",
    "hidden instruction",
    "instruction hidden",
)
_CODE_SWITCH_PHRASES = (
    "code-switch",
    "code switch",
    "codeswitch",
    "chuyển mã",
    "trộn ngôn ngữ",
    "mixed-language",
    "mixed language",
)
# A phrase after one of these, in the same clause, does not describe a required case.
_NEGATION = re.compile(r"(?<!\w)(?:không|chẳng|chả|chưa|đừng|not|never|without|no)(?!\w)", re.IGNORECASE)
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
    routed: Sequence[Mapping[str, Any]] | None = None,
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
    max_turns = int(budget["max_turns"])
    for rule in eligible:
        text = _source_text(rule, spans)
        languages = _declared_languages(rule)
        primary = languages[0] if languages else None
        required.extend(_required_cells(rule, by_id, text, languages, primary, max_turns))
        optional.extend(_optional_cells(rule, primary, modules))
        families.extend(_contrast_families(rule, _paired_allow(rule, by_id)))
    required.sort(key=_sort_key)
    optional.sort(key=_sort_key)
    rows, gaps = _schedule(required, optional, case_cap, budget)
    scheduled = {row["slot_id"] for row in rows}
    marked = tuple(_with_scheduled(family, scheduled) for family in families)
    blocking = [gap for gap in gaps if gap["blocks_gold"]]
    certification, reason = _certification(len(required), blocking)
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
    )


def write_coverage_plan(stage_cache: Path, plan: CoveragePlan) -> dict[str, Any]:
    """Write the scheduled rows and the gap report. Returns the run-record summary."""
    write_jsonl(stage_cache / PLAN_FILE, plan.rows)
    summary = plan.summary()
    write_json(stage_cache / GAPS_FILE, summary)
    return plan.record()


def _required_cells(
    rule: PolicyRule,
    by_id: Mapping[str, PolicyRule],
    text: str,
    languages: Sequence[str],
    primary: str | None,
    max_turns: int,
) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    action = _primary_action(rule)
    allow_action, partner, allow_flags = _allow_case(rule, by_id)
    if action is not None:
        cells.append(_cell(rule, "clear_violation", action, primary=primary))
    cells.append(
        _cell(rule, "clear_allow", allow_action, primary=primary, flags=allow_flags, paired_policy_id=partner)
    )
    ambiguous = rule.decision.ambiguous_action
    cells.append(
        _cell(
            rule,
            "boundary",
            ambiguous,
            primary=primary,
            flags=() if ambiguous is not None else ("boundary_action_unspecified",),
        )
    )
    if rule.exceptions:
        cells.append(_cell(rule, "exception", "allow", primary=primary, exceptions=list(rule.exceptions)))
    cells.extend(_missing_legs(rule, action, primary, allow_flags, partner))
    for language in languages[1:]:
        cells.append(
            _cell(
                rule,
                "language",
                action or "allow",
                primary=primary,
                language=language,
                slot_id=_slot(rule.policy_id, "language", language),
            )
        )
    switched = _code_switch_languages(text, languages)
    if switched is not None:
        cells.append(
            _cell(
                rule,
                "code_switch",
                action or "allow",
                primary=primary,
                languages=switched,
            )
        )
    multi_turn = _multi_turn(rule, text, action, primary, max_turns)
    if multi_turn is not None:
        cells.append(multi_turn)
    return cells


def _missing_legs(
    rule: PolicyRule,
    action: str | None,
    primary: str | None,
    allow_flags: Sequence[str],
    partner: str | None,
) -> list[dict[str, Any]]:
    """One allow case per leg of an ``all_of`` group. ``any_of`` adds none, including the default."""
    del action
    cells: list[dict[str, Any]] = []
    for name, _group_action, group in _acting_groups(rule):
        if group.operator != "all_of" or len(group.legs) < 2:
            continue
        for absent in group.legs:
            present = [leg for leg in group.legs if leg != absent]
            cells.append(
                _cell(
                    rule,
                    "missing_leg",
                    "allow",
                    primary=primary,
                    flags=allow_flags,
                    slot_id=_slot(rule.policy_id, "missing_leg", name, absent),
                    condition_class=name,
                    absent_leg=absent,
                    present_legs=present,
                    paired_policy_id=partner,
                )
            )
    return cells


def _multi_turn(
    rule: PolicyRule, text: str, action: str | None, primary: str | None, max_turns: int
) -> dict[str, Any] | None:
    steps = len(_STEP.findall(text))
    if steps < 2 and not any(_affirmed(text, phrase) for phrase in _SEQUENCE_PHRASES):
        return None
    source_turns = steps if steps >= 2 else 2
    cell = _cell(
        rule,
        "multi_turn",
        action or "allow",
        primary=primary,
        source_turns=source_turns,
    )
    # A multi-turn case needs two turns. A lower cap cannot be filled by a one-turn case.
    if max_turns < 2:
        cell["hold"] = "max_turns"
        return cell
    cell["turns"] = min(source_turns, max_turns)
    if source_turns > max_turns:
        cell.setdefault("flags", []).append("turns_capped")
    return cell


def _optional_cells(rule: PolicyRule, primary: str | None, modules: Mapping[str, Any]) -> list[dict[str, Any]]:
    action = _primary_action(rule) or "allow"
    cells: list[dict[str, Any]] = []
    if modules.get("adversarial") is True and _primary_action(rule) is not None:
        cells.append(
            _cell(rule, "adversarial", action, primary=primary, priority="optional", flags=("preserves_label",))
        )
    if modules.get("political_symmetry") is True:
        cells.append(
            _cell(rule, "actor_swap", action, primary=primary, priority="optional", flags=("preserves_label",))
        )
    cells.append(_cell(rule, "persona", action, primary=primary, priority="optional", flags=("preserves_label",)))
    return cells


def _contrast_families(rule: PolicyRule, paired_allow: bool) -> list[dict[str, Any]]:
    action = _primary_action(rule)
    if action is None:
        return []
    violation = _slot(rule.policy_id, "clear_violation")
    families = []
    for name, _group_action, group in _acting_groups(rule):
        if group.operator != "all_of" or len(group.legs) < 2:
            continue
        for absent in group.legs:
            families.append(
                {
                    "family_id": _family_id(rule.policy_id, name, absent),
                    "policy_id": rule.policy_id,
                    "changed_dimension": absent,
                    "members": [
                        {"slot_id": violation, "role": "all_legs", "expected_action": action},
                        {
                            "slot_id": _slot(rule.policy_id, "missing_leg", name, absent),
                            "role": "missing_leg",
                            "expected_action": "allow",
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
                {"slot_id": violation, "role": "violation", "expected_action": action},
                {"slot_id": _slot(rule.policy_id, "clear_allow"), "role": "allow", "expected_action": "allow"},
            ],
        }
    ]


def _schedule(
    required: Sequence[dict[str, Any]],
    optional: Sequence[dict[str, Any]],
    case_cap: int,
    budget: Mapping[str, Any],
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
        if used.get(policy_id, 0) >= _rule_cap(str(cell["severity"]), budget):
            gaps.append({**stored, "reason": "rule_cap", "blocks_gold": blocks_gold})
            return
        if len(rows) >= case_cap:
            gaps.append({**stored, "reason": "budget_cap", "blocks_gold": blocks_gold})
            return
        stored["order"] = len(rows)
        rows.append(stored)
        used[policy_id] = used.get(policy_id, 0) + 1

    for cell in required:
        place(cell, blocks_gold=True)
    for cell in optional:
        place(cell, blocks_gold=False)
    return rows, gaps


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


def _acting_groups(rule: PolicyRule) -> list[tuple[str, str, ConditionGroup]]:
    found = []
    for name, fallback in _ACTING:
        group = getattr(rule.rule, name)
        if group is None:
            continue
        if name == "block_if":
            action = rule.decision.violation_action
        elif name == "escalate_if":
            action = rule.decision.critical_action or fallback
        else:
            action = fallback
        found.append((name, action, group))
    return found


def _primary_action(rule: PolicyRule) -> str | None:
    found = _acting_groups(rule)
    if not found:
        return None
    return min(found, key=lambda item: ACTION_ORDER.index(item[1]))[1]


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


def _code_switch_languages(text: str, languages: Sequence[str]) -> list[str] | None:
    """The declared languages, when the source affirms that this passage is code-switched.

    One declared language cannot name a pair, so incidental foreign words do not add a cell.
    """
    if len(languages) < 2:
        return None
    if not any(_affirmed(text, phrase) for phrase in _CODE_SWITCH_PHRASES):
        return None
    return list(languages)


def _source_text(rule: PolicyRule, spans: Mapping[str, str]) -> str:
    parts = [rule.provenance.source_quote]
    span_id = rule.provenance.source_span_id
    if span_id is not None and span_id in spans:
        parts.append(spans[span_id])
    return "\n".join(parts)


def _affirmed(text: str, phrase: str) -> bool:
    """True when ``phrase`` occurs and the same clause does not negate it."""
    folded = text.casefold()
    pattern = re.compile(rf"(?<!\w){re.escape(phrase.casefold())}(?!\w)")
    for match in pattern.finditer(folded):
        window = folded[max(0, match.start() - 40) : match.start()]
        clause = re.split(r"[.!?;\n]", window)[-1]
        if _NEGATION.search(clause) is None:
            return True
    return False


def _rule_cap(severity: str, budget: Mapping[str, Any]) -> int:
    key = "cases_per_critical_rule" if severity == "critical" else "cases_per_normal_rule"
    return int(budget[key])


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
        "production_replay": "not_configured",
    }


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
