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

"""Scenario seeds and contrast pairs for a coverage plan.

The plan decides the meaning of each case. This stage copies that meaning into a
conversation. The seed itself does not call a model and does not add a policy
requirement. A later paraphrase pass may rewrite the user turns when that module
is enabled. A proposal that names an unknown rule, changes the planned meaning, or
removes the wrong condition is rejected. Its legs must equal the seed's legs, in
the seed's order, and its turn text must be that seed's sentence. A rejected slot
is left empty. It is not replaced with a seed. A slot with no proposal receives
one seed. A plan row that does not match an eligible rule stops the build.

A missing-leg case is written once. Its case id is the plan slot id, and the contrast
family points at that same id. Its action is allow only when those facts match no rule.
An allow case is also listed in the over-refusal index. That index does not repeat the
conversation. A contrast member whose slot is neither scheduled nor listed as unscheduled
stops the build.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nemotron.steps.byob.runtime.benchmark_families.safety.actions import ACTION_ORDER
from nemotron.steps.byob.runtime.benchmark_families.safety.artifacts import write_jsonl
from nemotron.steps.byob.runtime.benchmark_families.safety.case_facts import case_context
from nemotron.steps.byob.runtime.benchmark_families.safety.confidence import canonical_text
from nemotron.steps.byob.runtime.benchmark_families.safety.ir import PolicyRule

CASES_FILE = "scenario_candidates.jsonl"
REJECTIONS_FILE = "scenario_rejections.jsonl"
CONTRAST_FILE = "contrast_families.jsonl"
OVERREFUSAL_FILE = "overrefusal_cases.jsonl"
SCENARIO_OUTPUTS = (CASES_FILE, REJECTIONS_FILE, CONTRAST_FILE, OVERREFUSAL_FILE)

class ScenarioInputError(ValueError):
    """The plan, a contrast family, or a proposal list cannot be read."""


@dataclass(frozen=True)
class ScenarioBuild:
    """Accepted cases, rejected attempts, contrast families, and over-refusal pointers."""

    cases: tuple[dict[str, Any], ...]
    rejections: tuple[dict[str, Any], ...]
    families: tuple[dict[str, Any], ...]
    overrefusal: tuple[dict[str, Any], ...]

    def summary(self) -> dict[str, Any]:
        missing = [case["case_id"] for case in self.cases if case["kind"] == "missing_leg"]
        reused = sum(
            1
            for family in self.families
            for member in family["members"]
            if member.get("reused") and member.get("case_id") in missing
        )
        return {
            "accepted": len(self.cases),
            "rejected": len(self.rejections),
            "contrast_families": len(self.families),
            "missing_leg_cases": len(missing),
            "reused_missing_leg": reused,
            "over_refusal_probes": len(self.overrefusal),
        }


def load_plan_rows(text: str) -> list[dict[str, Any]]:
    """Read a coverage plan. A damaged line stops the stage."""
    rows = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ScenarioInputError(f"coverage plan line {line_number} is not JSON") from exc
        if not isinstance(row, dict):
            raise ScenarioInputError(f"coverage plan line {line_number} must be an object")
        rows.append(row)
    return rows


def load_contrast_families(payload: Any) -> list[Any]:
    """Read contrast families from the gap report."""
    if not isinstance(payload, dict) or not isinstance(payload.get("contrast_families"), list):
        raise ScenarioInputError("coverage_gaps.json has no contrast_families list")
    return payload["contrast_families"]


def load_unscheduled_slots(payload: Any) -> list[str]:
    """Slot ids the planner listed as gaps. A gap without a slot id stops the stage."""
    if not isinstance(payload, dict) or not isinstance(payload.get("gaps"), list):
        raise ScenarioInputError("coverage_gaps.json has no gaps list")
    slots = []
    for index, row in enumerate(payload["gaps"], start=1):
        slot_id = row.get("slot_id") if isinstance(row, dict) else None
        if not isinstance(slot_id, str) or not slot_id:
            raise ScenarioInputError(f"coverage_gaps.json gap {index} needs a slot_id")
        slots.append(slot_id)
    return slots


def build_scenarios(
    rows: Sequence[Mapping[str, Any]],
    families: Sequence[Mapping[str, Any]],
    rules: Sequence[PolicyRule],
    proposals: Sequence[Mapping[str, Any]] | None = None,
    *,
    unscheduled: Sequence[str] | None = None,
) -> ScenarioBuild:
    """Build one case per scheduled slot, then point each contrast member at that case."""
    cells = _cells(rows)
    checked_families = _check_families(families)
    _known_members(checked_families, set(cells) | _slot_ids(unscheduled))
    by_id = _index_rules(rules)
    _require_plan(cells, by_id)
    accepted, rejected_slots, rejections = _proposals(proposals, cells, by_id)
    cases: list[dict[str, Any]] = []
    for cell in cells.values():
        slot_id = str(cell["slot_id"])
        if slot_id in accepted:
            cases.append(accepted[slot_id])
            continue
        if slot_id in rejected_slots:
            continue
        case, reasons = _seed(cell, by_id[str(cell["policy_id"])], by_id)
        if reasons:
            raise ScenarioInputError(f"plan row {slot_id} cannot be seeded: {', '.join(reasons)}")
        cases.append(case)
    by_slot = {case["slot_id"]: case for case in cases}
    overrefusal = tuple(
        {
            "case_id": case["case_id"],
            "slot_id": case["slot_id"],
            "policy_id": case["policy_id"],
            "kind": case["kind"],
            "expected_action": "allow",
            "probe": "over_refusal",
        }
        for case in cases
        if case["expected_action"] == "allow"
    )
    return ScenarioBuild(
        cases=tuple(cases),
        rejections=tuple(rejections),
        families=_attach(checked_families, by_slot, rejected_slots, by_id),
        overrefusal=overrefusal,
    )


def write_scenarios(stage_cache: Path, build: ScenarioBuild) -> dict[str, Any]:
    """Write the four scenario files and return the counts stored on the run."""
    write_jsonl(stage_cache / CASES_FILE, build.cases)
    write_jsonl(stage_cache / REJECTIONS_FILE, build.rejections)
    write_jsonl(stage_cache / CONTRAST_FILE, build.families)
    write_jsonl(stage_cache / OVERREFUSAL_FILE, build.overrefusal)
    return build.summary()


def _cells(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise ScenarioInputError("plan rows must be a list")
    cells: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ScenarioInputError("a plan row must be an object")
        slot_id = row.get("slot_id")
        if not isinstance(slot_id, str) or not slot_id:
            raise ScenarioInputError("a plan row needs a slot_id")
        if slot_id in cells:
            raise ScenarioInputError(f"plan repeats slot {slot_id}")
        if not isinstance(row.get("policy_id"), str) or not isinstance(row.get("kind"), str):
            raise ScenarioInputError(f"plan row {slot_id} needs a policy_id and a kind")
        if "expected_action" not in row:
            raise ScenarioInputError(f"plan row {slot_id} needs an expected_action")
        action = row.get("expected_action")
        if action is None:
            if row.get("kind") != "boundary":
                raise ScenarioInputError(f"plan row {slot_id} has an empty expected_action")
        elif action not in ACTION_ORDER:
            raise ScenarioInputError(f"plan row {slot_id} has an unknown expected_action")
        cells[slot_id] = row
    return cells


def _check_families(families: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    if isinstance(families, (str, bytes)) or not isinstance(families, Sequence):
        raise ScenarioInputError("contrast families must be a list")
    checked = []
    seen: set[str] = set()
    for family in families:
        if not isinstance(family, Mapping):
            raise ScenarioInputError("a contrast family must be an object")
        family_id = family.get("family_id")
        if not isinstance(family_id, str) or not family_id:
            raise ScenarioInputError("a contrast family needs a family_id")
        if family_id in seen:
            raise ScenarioInputError(f"contrast families repeat {family_id}")
        seen.add(family_id)
        members = family.get("members")
        if isinstance(members, (str, bytes)) or not isinstance(members, Sequence) or not members:
            raise ScenarioInputError(f"contrast family {family_id} needs members")
        copied = []
        for member in members:
            if not isinstance(member, Mapping) or not isinstance(member.get("slot_id"), str):
                raise ScenarioInputError(f"contrast family {family_id} has a member without a slot_id")
            copied.append(dict(member))
        stored = dict(family)
        stored["members"] = copied
        checked.append(stored)
    return tuple(checked)


def _slot_ids(slots: Sequence[str] | None) -> set[str]:
    if slots is None:
        return set()
    if isinstance(slots, (str, bytes)) or not isinstance(slots, Sequence):
        raise ScenarioInputError("unscheduled slots must be a list")
    found = set()
    for slot in slots:
        if not isinstance(slot, str) or not slot:
            raise ScenarioInputError("an unscheduled slot must be a slot id")
        found.add(slot)
    return found


def _known_members(families: Sequence[Mapping[str, Any]], known: set[str]) -> None:
    """A contrast slot is scheduled or listed in the gaps. Anything else stops the build."""
    for family in families:
        for member in family["members"]:
            slot_id = member["slot_id"]
            if slot_id not in known:
                raise ScenarioInputError(f"contrast family {family['family_id']} names unknown slot {slot_id}")


def _index_rules(rules: Sequence[PolicyRule]) -> dict[str, PolicyRule]:
    if isinstance(rules, (str, bytes)) or not isinstance(rules, Sequence):
        raise ScenarioInputError("rules must be a list")
    indexed: dict[str, PolicyRule] = {}
    for rule in rules:
        if not isinstance(rule, PolicyRule):
            raise ScenarioInputError("rules must be PolicyRule values")
        if rule.policy_id in indexed:
            raise ScenarioInputError(f"rules repeat {rule.policy_id}")
        if rule.status == "eligible_for_generation":
            indexed[rule.policy_id] = rule
    return indexed


def _require_plan(cells: Mapping[str, Mapping[str, Any]], by_id: Mapping[str, PolicyRule]) -> None:
    """A scheduled row that disagrees with its rule stops the build."""
    for cell in cells.values():
        slot_id = str(cell["slot_id"])
        rule = by_id.get(str(cell["policy_id"]))
        if rule is None:
            raise ScenarioInputError(f"plan row {slot_id} names unknown policy {cell['policy_id']}")
        if cell.get("kind") == "exception" and list(cell.get("exceptions") or []) != list(rule.exceptions):
            raise ScenarioInputError(f"plan row {slot_id} does not name the rule's exceptions")
        context = _context(cell, rule, by_id)
        legs_match = (
            context is not None
            and _stated(cell, "present_legs", context[0])
            and _stated(cell, "absent_legs", context[1])
        )
        if not legs_match:
            raise ScenarioInputError(f"plan row {slot_id} does not match its rule")
        if cell.get("expected_action") != context[2]:
            raise ScenarioInputError(f"plan row {slot_id} has an expected_action that does not match its rule")


def _proposals(
    proposals: Sequence[Mapping[str, Any]] | None,
    cells: Mapping[str, Mapping[str, Any]],
    by_id: Mapping[str, PolicyRule],
) -> tuple[dict[str, dict[str, Any]], set[str], list[dict[str, Any]]]:
    if proposals is None:
        return {}, set(), []
    if isinstance(proposals, (str, bytes)) or not isinstance(proposals, Sequence):
        raise ScenarioInputError("proposals must be a list")
    accepted: dict[str, dict[str, Any]] = {}
    rejected: set[str] = set()
    rejections: list[dict[str, Any]] = []
    seen: set[str] = set()
    for proposal in proposals:
        if not isinstance(proposal, Mapping):
            raise ScenarioInputError("a proposal must be an object")
        slot_id = proposal.get("slot_id")
        policy_id = proposal.get("policy_id") if isinstance(proposal.get("policy_id"), str) else None
        if not isinstance(slot_id, str) or slot_id not in cells:
            rejections.append(_rejection(slot_id if isinstance(slot_id, str) else None, policy_id, ["unknown_slot"]))
            continue
        if slot_id in seen:
            rejections.append(_rejection(slot_id, policy_id or cells[slot_id].get("policy_id"), ["duplicate_case"]))
            continue
        seen.add(slot_id)
        cell = cells[slot_id]
        reasons = _validate(proposal, cell, by_id)
        if reasons:
            rejections.append(_rejection(slot_id, policy_id or cell.get("policy_id"), reasons))
            rejected.add(slot_id)
            continue
        rule = by_id[str(cell["policy_id"])]
        present, absent = _canonical_legs(proposal, cell, rule, by_id)
        accepted[slot_id] = _assemble(cell, rule, _canonical_turns(proposal, cell), present, absent, "proposal")
    return accepted, rejected, rejections


def _seed(
    cell: Mapping[str, Any], rule: PolicyRule, by_id: Mapping[str, PolicyRule]
) -> tuple[dict[str, Any] | None, list[str]]:
    count = _turn_count(cell)
    if count is None:
        return None, ["invalid_multi_turn"]
    legs = _seed_legs(cell, rule, by_id)
    if legs is None:
        return None, ["invalid_missing_leg"]
    present, absent = legs
    turns = [
        _turn(number, _sentence(cell, rule, present, absent, number, count), cell.get("expected_action"))
        for number in range(1, count + 1)
    ]
    case = _assemble(cell, rule, turns, present, absent, "seed")
    reasons = _validate(case, cell, by_id)
    return (None, reasons) if reasons else (case, [])


def _validate(
    case: Mapping[str, Any], cell: Mapping[str, Any], by_id: Mapping[str, PolicyRule]
) -> list[str]:
    reasons: list[str] = []
    policy_id = case.get("policy_id")
    cell_id = cell.get("policy_id")
    if not isinstance(policy_id, str) or policy_id not in by_id:
        _add(reasons, "unknown_policy_id")
    if policy_id != cell_id:
        _add(reasons, "changed_meaning")
    rule = by_id.get(cell_id) if isinstance(cell_id, str) else None
    if rule is None or policy_id != rule.policy_id:
        return reasons
    if "case_id" in case and case.get("case_id") != cell.get("slot_id"):
        _add(reasons, "duplicate_case")
    if case.get("kind") != cell.get("kind"):
        _add(reasons, "changed_meaning")
    if "expected_action" not in case or case.get("expected_action") != cell.get("expected_action"):
        _add(reasons, "changed_meaning")
    present = _as_legs(case.get("present_legs"))
    absent = _as_legs(case.get("absent_legs"))
    if present is None or absent is None:
        _add(reasons, "invalid_missing_leg" if cell.get("kind") == "missing_leg" else "changed_meaning")
    else:
        _extend(reasons, _legs_ok(present, absent, cell, by_id))
    _extend(reasons, _evidence_ok(case.get("evidence"), rule))
    _extend(reasons, _turns_ok(case.get("turns"), cell, rule, by_id, present or [], absent or []))
    return reasons


def _legs_ok(
    present: list[str], absent: list[str], cell: Mapping[str, Any], by_id: Mapping[str, PolicyRule]
) -> list[str]:
    known = _known_legs(by_id)
    if any(leg not in known for leg in (*present, *absent)):
        return ["changed_meaning"]
    rule = by_id.get(str(cell.get("policy_id")))
    if rule is None:
        return ["changed_meaning"]
    expected = _seed_legs(cell, rule, by_id)
    if expected is None:
        return ["invalid_missing_leg"] if cell.get("kind") == "missing_leg" else ["changed_meaning"]
    expected_present, expected_absent = expected
    if present != expected_present or absent != expected_absent:
        return ["invalid_missing_leg"] if cell.get("kind") == "missing_leg" else ["changed_meaning"]
    return []


def _evidence_ok(evidence: Any, rule: PolicyRule) -> list[str]:
    if not isinstance(evidence, Mapping):
        return ["changed_meaning"]
    reasons: list[str] = []
    quote = evidence.get("source_quote")
    provenance = rule.provenance
    if not isinstance(quote, str) or canonical_text(quote) != canonical_text(provenance.source_quote):
        _add(reasons, "changed_meaning")
    if evidence.get("document_id") != provenance.document_id:
        _add(reasons, "changed_meaning")
    if provenance.source_span_id is not None and evidence.get("source_span_id") != provenance.source_span_id:
        _add(reasons, "changed_meaning")
    if provenance.source_span_id is None and evidence.get("source_span_id") is not None:
        _add(reasons, "changed_meaning")
    if provenance.document_sha256 is not None and evidence.get("document_sha256") != provenance.document_sha256:
        _add(reasons, "changed_meaning")
    return reasons


def _turns_ok(
    turns: Any,
    cell: Mapping[str, Any],
    rule: PolicyRule,
    by_id: Mapping[str, PolicyRule],
    present: list[str],
    absent: list[str],
) -> list[str]:
    count = _turn_count(cell)
    kind = cell.get("kind")
    if count is None:
        return ["invalid_multi_turn"]
    if not isinstance(turns, list) or len(turns) != count:
        return ["invalid_multi_turn" if kind == "multi_turn" else "changed_meaning"]
    reasons: list[str] = []
    action = cell.get("expected_action")
    expected = _seed_legs(cell, rule, by_id)
    check_sentence = expected is not None and (present, absent) == expected
    for number, turn in enumerate(turns, start=1):
        if not isinstance(turn, Mapping):
            _add(reasons, "invalid_multi_turn" if kind == "multi_turn" else "changed_meaning")
            continue
        if turn.get("turn") != number or turn.get("role") != "user" or turn.get("scored") is not True:
            _add(reasons, "changed_meaning")
        if "expected_action" not in turn or turn.get("expected_action") in ("",):
            _add(reasons, "missing_turn_decision")
        elif turn.get("expected_action") != action:
            _add(reasons, "changed_meaning" if turn.get("expected_action") else "missing_turn_decision")
        text = turn.get("text")
        if not isinstance(text, str) or not text.strip():
            _add(reasons, "changed_meaning")
            continue
        if check_sentence and expected is not None:
            _extend(reasons, _text_ok(text, cell, rule, expected[0], expected[1], number, count))
    return reasons


def _text_ok(
    text: str,
    cell: Mapping[str, Any],
    rule: PolicyRule,
    present: list[str],
    absent: list[str],
    number: int,
    count: int,
) -> list[str]:
    """Accept only the seed sentence, so a proposal cannot add a requirement in prose."""
    expected = _sentence(cell, rule, present, absent, number, count)
    if canonical_text(text) != canonical_text(expected):
        return ["changed_meaning"]
    return []


def _seed_legs(
    cell: Mapping[str, Any], rule: PolicyRule, by_id: Mapping[str, PolicyRule]
) -> tuple[list[str], list[str]] | None:
    context = _context(cell, rule, by_id)
    if context is None or cell.get("expected_action") != context[2]:
        return None
    present, absent, _action = context
    if not _stated(cell, "present_legs", present) or not _stated(cell, "absent_legs", absent):
        return None
    return present, absent


def _context(
    cell: Mapping[str, Any], rule: PolicyRule, by_id: Mapping[str, PolicyRule]
) -> tuple[list[str], list[str], str | None] | None:
    kind = cell.get("kind")
    if not isinstance(kind, str):
        return None
    condition = cell.get("condition_class")
    absent = cell.get("absent_leg")
    trigger = cell.get("trigger_leg")
    return case_context(
        tuple(by_id.values()),
        rule,
        kind,
        condition_class=condition if isinstance(condition, str) else None,
        absent_leg=absent if isinstance(absent, str) else None,
        trigger_leg=trigger if isinstance(trigger, str) else None,
    )


def _stated(cell: Mapping[str, Any], key: str, expected: list[str]) -> bool:
    """A plan row that omits the leg list takes the derived legs. A stated list must equal them."""
    if key not in cell:
        return True
    stated = cell.get(key)
    if not isinstance(stated, (list, tuple)) or any(not isinstance(leg, str) for leg in stated):
        return False
    return list(stated) == expected


def _canonical_legs(
    proposal: Mapping[str, Any], cell: Mapping[str, Any], rule: PolicyRule, by_id: Mapping[str, PolicyRule]
) -> tuple[list[str], list[str]]:
    expected = _seed_legs(cell, rule, by_id)
    if expected is None:
        return _as_legs(proposal.get("present_legs")) or [], _as_legs(proposal.get("absent_legs")) or []
    return expected


def _canonical_turns(proposal: Mapping[str, Any], cell: Mapping[str, Any]) -> list[dict[str, Any]]:
    action = cell.get("expected_action")
    turns = proposal.get("turns")
    if not isinstance(turns, list):
        return []
    return [
        _turn(number, turn.get("text") if isinstance(turn, Mapping) else "", action)
        for number, turn in enumerate(turns, start=1)
    ]


def _assemble(
    cell: Mapping[str, Any],
    rule: PolicyRule,
    turns: list[dict[str, Any]],
    present: list[str],
    absent: list[str],
    origin: str,
) -> dict[str, Any]:
    evidence: dict[str, Any] = {
        "document_id": rule.provenance.document_id,
        "source_quote": rule.provenance.source_quote,
    }
    if rule.provenance.document_sha256 is not None:
        evidence["document_sha256"] = rule.provenance.document_sha256
    if rule.provenance.source_span_id is not None:
        evidence["source_span_id"] = rule.provenance.source_span_id
    case: dict[str, Any] = {
        "case_id": cell["slot_id"],
        "slot_id": cell["slot_id"],
        "policy_id": rule.policy_id,
        "kind": cell["kind"],
        "priority": cell.get("priority"),
        "expected_action": cell.get("expected_action"),
        "origin": origin,
        "present_legs": list(present),
        "absent_legs": list(absent),
        "over_refusal_probe": cell.get("expected_action") == "allow",
        "evidence": evidence,
        "turns": turns,
    }
    flags = cell.get("flags")
    if isinstance(flags, (list, tuple)) and flags:
        case["flags"] = [flag for flag in flags if isinstance(flag, str)]
    if isinstance(cell.get("language"), str):
        case["language"] = cell["language"]
    languages = cell.get("languages")
    if isinstance(languages, (list, tuple)) and languages:
        case["languages"] = list(languages)
    if cell.get("kind") == "missing_leg" and isinstance(cell.get("condition_class"), str):
        case["condition_class"] = cell["condition_class"]
        case["absent_leg"] = cell.get("absent_leg")
    if cell.get("kind") == "all_legs" and isinstance(cell.get("condition_class"), str):
        case["condition_class"] = cell["condition_class"]
    if cell.get("kind") == "condition_path" and isinstance(cell.get("condition_class"), str):
        case["condition_class"] = cell["condition_class"]
        if isinstance(cell.get("trigger_leg"), str):
            case["trigger_leg"] = cell["trigger_leg"]
    if cell.get("kind") == "exception":
        case["exceptions"] = list(rule.exceptions)
    return case


def _attach(
    families: Sequence[Mapping[str, Any]],
    cases_by_slot: Mapping[str, Mapping[str, Any]],
    rejected: set[str],
    by_id: Mapping[str, PolicyRule],
) -> tuple[dict[str, Any], ...]:
    attached = []
    for family in families:
        members = []
        for member in family["members"]:
            slot_id = member["slot_id"]
            case = cases_by_slot.get(slot_id)
            row = dict(member)
            if case is None:
                row["case_id"] = None
                row["reused"] = False
                row["generation"] = "rejected" if slot_id in rejected else "unscheduled"
            else:
                row["case_id"] = case["case_id"]
                row["reused"] = case["kind"] == "missing_leg"
                row["generation"] = "accepted"
            members.append(row)
        if all(member["generation"] == "accepted" for member in members):
            _require_pair(family, cases_by_slot, by_id)
        stored = dict(family)
        stored["members"] = members
        attached.append(stored)
    return tuple(attached)


def _require_pair(
    family: Mapping[str, Any], cases_by_slot: Mapping[str, Mapping[str, Any]], by_id: Mapping[str, PolicyRule]
) -> None:
    """A complete contrast pair may change only the named leg, and its label must match its facts."""
    found: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    family_id = family.get("family_id")
    for member in family["members"]:
        case = cases_by_slot[member["slot_id"]]
        if case["policy_id"] != family.get("policy_id"):
            raise ScenarioInputError(f"contrast family {family_id} points at {case['policy_id']}")
        if member.get("expected_action") != case["expected_action"]:
            raise ScenarioInputError(f"contrast family {family_id} disagrees with {case['slot_id']}")
        if not _facts_match(case, by_id):
            raise ScenarioInputError(f"contrast family {family_id} disagrees with {case['slot_id']}")
        found.append((member, case))
    allows = [case for member, case in found if member.get("role") == "missing_leg"]
    if not allows:
        by_role = {member.get("role"): case for member, case in found}
        benign = by_role.get("allow")
        violation = by_role.get("violation")
        if benign is not None and violation is not None and _same_facts(benign, violation):
            raise ScenarioInputError(f"contrast family {family_id} does not change the facts")
        return
    blocks = [case for member, case in found if member.get("role") != "missing_leg"]
    if len(allows) != 1 or len(blocks) != 1:
        raise ScenarioInputError(f"contrast family {family_id} needs one block case and one allow case")
    block, allow = blocks[0], allows[0]
    absent = family.get("changed_dimension")
    one_leg = (
        isinstance(absent, str)
        and allow["absent_legs"] == [absent]
        and absent not in allow["present_legs"]
        and not block["absent_legs"]
        and set(block["present_legs"]) == set(allow["present_legs"]) | {absent}
    )
    if not one_leg:
        raise ScenarioInputError(f"contrast family {family_id} does not remove exactly {absent}")
    if block.get("condition_class") and block.get("condition_class") != allow.get("condition_class"):
        raise ScenarioInputError(f"contrast family {family_id} changes the condition group")


def _facts_match(case: Mapping[str, Any], by_id: Mapping[str, PolicyRule]) -> bool:
    rule = by_id.get(str(case.get("policy_id")))
    if rule is None:
        return False
    context = _context(case, rule, by_id)
    if context is None:
        return False
    return (list(case["present_legs"]), list(case["absent_legs"]), case["expected_action"]) == context


def _same_facts(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return left["present_legs"] == right["present_legs"] and left["absent_legs"] == right["absent_legs"]


def _sentence(
    cell: Mapping[str, Any],
    rule: PolicyRule,
    present: list[str],
    absent: list[str],
    number: int,
    total: int,
) -> str:
    exceptions = list(rule.exceptions) if cell.get("kind") == "exception" else []
    return seed_sentence(
        rule, present, absent, number, total, kind=str(cell.get("kind")), exceptions=exceptions
    )


def seed_sentence(
    rule: PolicyRule,
    present: Sequence[str],
    absent: Sequence[str],
    number: int,
    total: int,
    *,
    kind: str,
    exceptions: Sequence[str] = (),
) -> str:
    """The scored turn text is the source quote in its original language.

    Legs, absent legs, exceptions, and the turn index stay on the case. They are not
    rewritten into one language's scaffolding.
    """
    del present, absent, number, total, kind, exceptions
    return rule.provenance.source_quote


def _turn(number: int, text: Any, action: Any) -> dict[str, Any]:
    return {"turn": number, "role": "user", "text": text, "scored": True, "expected_action": action}


def _turn_count(cell: Mapping[str, Any]) -> int | None:
    if cell.get("kind") != "multi_turn":
        return 1
    count = cell.get("turns")
    if isinstance(count, bool) or not isinstance(count, int) or count < 2:
        return None
    return count


def _known_legs(by_id: Mapping[str, PolicyRule]) -> set[str]:
    """Legs and exception names of every eligible rule. A fact from another rule is still a fact."""
    legs: set[str] = set()
    for rule in by_id.values():
        legs.update(rule.exceptions)
        for group in rule.rule.conditions().values():
            legs.update(group.legs)
    return legs


def _as_legs(value: Any) -> list[str] | None:
    if value is None:
        return []
    if isinstance(value, (list, tuple)) and all(isinstance(item, str) and item for item in value):
        return list(value)
    return None


def _rejection(slot_id: Any, policy_id: Any, reasons: list[str]) -> dict[str, Any]:
    return {
        "slot_id": slot_id if isinstance(slot_id, str) else None,
        "policy_id": policy_id if isinstance(policy_id, str) else None,
        "reasons": _unique(reasons),
    }


def _add(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def _extend(reasons: list[str], extra: list[str]) -> None:
    for reason in extra:
        _add(reasons, reason)


def _unique(reasons: list[str]) -> list[str]:
    found: list[str] = []
    _extend(found, reasons)
    return found
