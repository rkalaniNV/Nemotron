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

"""Preliminary labels, deterministic checks, exact deduplication, and the split.

The label is recomputed from the case facts. A stored action that disagrees is
rejected and is not overwritten. A case with no eligible rule, or a quote that is
not inside its source span, is rejected. When one member of a complete contrast
family fails, the other members are rejected with it.

An exact duplicate has the same policy, action, language, kind, legs, and
canonical turn text. The lowest case id is kept. Members of one contrast family
stay together, and a different action stays as its own row.

Each connected group goes to development or private_test. A group holds a
contrast family, the variants of its violation, and any case that still carries
that violation's conditions, including an exception and a labeled boundary.
Both splits are filled when two or more groups exist. One group stays in
development and the manifest records an incomplete split. This stage does not
call a model. Semantic deduplication is not configured, and a public test split
is not assigned.
"""

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nemotron.steps.byob.runtime.benchmark_families.safety.actions import ACTION_ORDER
from nemotron.steps.byob.runtime.benchmark_families.safety.artifacts import write_json, write_jsonl
from nemotron.steps.byob.runtime.benchmark_families.safety.case_facts import case_context
from nemotron.steps.byob.runtime.benchmark_families.safety.confidence import canonical_text, quote_in_span
from nemotron.steps.byob.runtime.benchmark_families.safety.ir import PolicyRule
from nemotron.steps.byob.runtime.benchmark_families.safety.scenarios import seed_sentence

LABEL_FILE = "label_checks.jsonl"
VALIDATED_FILE = "validated_cases.jsonl"
REJECTED_FILE = "rejected_cases.jsonl"
REPORT_FILE = "validation_report.json"
DEDUPED_FILE = "deduplicated_cases.jsonl"
MANIFEST_FILE = "split_manifest.json"
DATASET_OUTPUTS = (LABEL_FILE, VALIDATED_FILE, REJECTED_FILE, REPORT_FILE, DEDUPED_FILE, MANIFEST_FILE)

# Variants that repeat a violation. They stay in that violation's split.
_WITH_VIOLATION = frozenset(
    {
        "language",
        "code_switch",
        "multi_turn",
        "adversarial",
        "actor_swap",
        "persona",
        "all_legs",
        "condition_path",
        "missing_leg",
    }
)
_MISSING = object()
_ROLES = frozenset({"user", "assistant", "tool"})
_CONTRAST_REASONS = frozenset({"contrast_pair_broken", "attack_removed"})


class DatasetInputError(ValueError):
    """The candidates, contrast families, rules, or spans cannot be read."""


@dataclass(frozen=True)
class DatasetBuild:
    """Label checks, accepted and rejected rows, the deduplicated rows, and the split."""

    labels: tuple[dict[str, Any], ...]
    validated: tuple[dict[str, Any], ...]
    rejected: tuple[dict[str, Any], ...]
    deduplicated: tuple[dict[str, Any], ...]
    report: dict[str, Any]
    manifest: dict[str, Any]

    def summary(self) -> dict[str, Any]:
        splits = self.manifest["splits"]
        return {
            "checked": self.report["checked"],
            "validated": self.report["validated"],
            "rejected": self.report["rejected"],
            "duplicates_removed": self.report["duplicates_removed"],
            "development": len(splits["development"]),
            "private_test": len(splits["private_test"]),
            "split_complete": self.manifest["complete"],
            "split_reason": self.manifest["reason"],
        }


def load_jsonl_objects(text: str, name: str) -> list[dict[str, Any]]:
    """Read a JSONL file of objects. A damaged line stops the stage."""
    rows = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetInputError(f"{name} line {line_number} is not JSON") from exc
        if not isinstance(row, dict):
            raise DatasetInputError(f"{name} line {line_number} must be an object")
        rows.append(row)
    return rows


def select_dataset(
    cases: Sequence[Mapping[str, Any]],
    families: Sequence[Mapping[str, Any]],
    rules: Sequence[PolicyRule],
    spans: Mapping[str, str],
) -> DatasetBuild:
    """Check every case, drop exact duplicates, and assign the two splits.

    A damaged list stops the build. A case that fails a check is rejected and the
    other cases continue. A stored label that disagrees with the aggregated action
    is rejected. The case keeps the label it arrived with.
    """
    stored = _cases(cases)
    parsed_families = _families(families)
    eligible = _rules(rules)
    span_text = _spans(spans)
    by_id = {rule.policy_id: rule for rule in eligible}
    reasons: dict[str, list[str]] = {}
    aggregated: dict[str, Any] = {}
    for case_id in sorted(stored):
        found, action = _judge(stored[case_id], by_id, eligible, span_text)
        reasons[case_id] = found
        aggregated[case_id] = action
    pairs = _complete_pairs(parsed_families, stored)
    _close_related(stored, reasons, pairs, eligible, by_id)
    labels = tuple(
        _label_row(stored[case_id], aggregated[case_id], reasons[case_id]) for case_id in sorted(stored)
    )
    validated_rows = []
    for case_id in sorted(stored):
        if reasons[case_id]:
            continue
        annotated = _annotate(stored[case_id], by_id)
        stored[case_id] = annotated
        validated_rows.append(annotated)
    validated = tuple(validated_rows)
    rejected = []
    for case_id in sorted(stored):
        if not reasons[case_id]:
            continue
        stage = "contrast" if set(reasons[case_id]) <= _CONTRAST_REASONS else "validation"
        rejected.append(_rejection(stored[case_id], reasons[case_id], stage))
    deduplicated, drops = _deduplicate(validated, pairs, by_id)
    rejected.extend(drops)
    rejected.sort(key=lambda row: str(row["case_id"]))
    manifest = _manifest(deduplicated, drops, pairs, by_id)
    report = _report(reasons, validated, rejected, deduplicated, pairs, drops)
    return DatasetBuild(
        labels=labels,
        validated=validated,
        rejected=tuple(rejected),
        deduplicated=tuple(deduplicated),
        report=report,
        manifest=manifest,
    )


def write_dataset(stage_cache: Path, build: DatasetBuild) -> dict[str, Any]:
    """Write the check, rejection, deduplicated rows, and ``split_manifest.json``."""
    write_jsonl(stage_cache / LABEL_FILE, build.labels)
    write_jsonl(stage_cache / VALIDATED_FILE, build.validated)
    write_jsonl(stage_cache / REJECTED_FILE, build.rejected)
    write_json(stage_cache / REPORT_FILE, build.report)
    write_jsonl(stage_cache / DEDUPED_FILE, build.deduplicated)
    write_json(stage_cache / MANIFEST_FILE, build.manifest)
    return build.summary()


def _cases(cases: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    if isinstance(cases, (str, bytes)) or not isinstance(cases, Sequence):
        raise DatasetInputError("scenario candidates must be a list")
    stored: dict[str, dict[str, Any]] = {}
    for case in cases:
        if not isinstance(case, Mapping):
            raise DatasetInputError("a scenario candidate must be an object")
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise DatasetInputError("a scenario candidate needs a case_id")
        if case_id in stored:
            raise DatasetInputError(f"scenario candidates repeat {case_id}")
        stored[case_id] = dict(case)
    return stored


def _families(families: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    if isinstance(families, (str, bytes)) or not isinstance(families, Sequence):
        raise DatasetInputError("contrast families must be a list")
    checked: list[dict[str, Any]] = []
    seen: set[str] = set()
    for family in families:
        if not isinstance(family, Mapping):
            raise DatasetInputError("a contrast family must be an object")
        family_id = family.get("family_id")
        if not isinstance(family_id, str) or not family_id:
            raise DatasetInputError("a contrast family needs a family_id")
        if family_id in seen:
            raise DatasetInputError(f"contrast families repeat {family_id}")
        seen.add(family_id)
        members = family.get("members")
        if isinstance(members, (str, bytes)) or not isinstance(members, Sequence) or not members:
            raise DatasetInputError(f"contrast family {family_id} needs members")
        copied = []
        for member in members:
            if not isinstance(member, Mapping) or not isinstance(member.get("slot_id"), str):
                raise DatasetInputError(f"contrast family {family_id} has a member without a slot_id")
            copied.append(dict(member))
        stored = dict(family)
        stored["members"] = copied
        checked.append(stored)
    return tuple(checked)


def _rules(rules: Sequence[PolicyRule]) -> tuple[PolicyRule, ...]:
    if isinstance(rules, (str, bytes)) or not isinstance(rules, Sequence):
        raise DatasetInputError("eligible rules must be a list")
    eligible: list[PolicyRule] = []
    seen: set[str] = set()
    for rule in rules:
        if not isinstance(rule, PolicyRule):
            raise DatasetInputError("eligible rules must be policy rules")
        if rule.policy_id in seen:
            raise DatasetInputError(f"eligible rules repeat {rule.policy_id}")
        seen.add(rule.policy_id)
        if rule.status != "eligible_for_generation":
            raise DatasetInputError(f"{rule.policy_id} is not eligible")
        eligible.append(rule)
    return tuple(eligible)


def _spans(spans: Mapping[str, str]) -> dict[str, str]:
    if isinstance(spans, (str, bytes)) or not isinstance(spans, Mapping):
        raise DatasetInputError("spans must map a span id to text")
    checked: dict[str, str] = {}
    for key, value in spans.items():
        if not isinstance(key, str) or not key or not isinstance(value, str):
            raise DatasetInputError("spans must map a span id to text")
        checked[key] = value
    return checked


def _complete_pairs(
    families: Sequence[Mapping[str, Any]], cases: Mapping[str, Mapping[str, Any]]
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Families whose every member was accepted and whose facts actually change.

    An unscheduled member is not a pair. Two copies of the same facts are not a
    pair either, so deduplication can still reduce them to one row.
    """
    pairs = []
    case_ids = set(cases)
    for family in families:
        family_id = str(family["family_id"])
        states = [_member_state(family_id, member, case_ids) for member in family["members"]]
        if any(state != "accepted" for state in states):
            continue
        ids = tuple(str(member["case_id"]) for member in family["members"])
        if len(set(ids)) != len(ids):
            raise DatasetInputError(f"contrast family {family_id} repeats a case")
        if len(ids) < 2:
            continue
        if _is_contrast(family, [cases[case_id] for case_id in ids]):
            pairs.append((family_id, ids))
    return tuple(pairs)


def _member_state(family_id: str, member: Mapping[str, Any], case_ids: set[str]) -> str:
    generation = member.get("generation")
    case_id = member.get("case_id")
    if generation is None:
        if isinstance(case_id, str) and case_id in case_ids:
            return "accepted"
        if case_id is None:
            return "unscheduled"
        raise DatasetInputError(_missing_case(family_id, case_id))
    if generation == "accepted":
        if not isinstance(case_id, str) or case_id not in case_ids:
            raise DatasetInputError(_missing_case(family_id, case_id))
        return "accepted"
    if generation in {"rejected", "unscheduled"}:
        return str(generation)
    raise DatasetInputError(f"contrast family {family_id} has an unknown generation")


def _is_contrast(family: Mapping[str, Any], members: Sequence[Mapping[str, Any]]) -> bool:
    """True when the accepted members are a real block-versus-allow change."""
    family_id = str(family.get("family_id"))
    policies = {case.get("policy_id") for case in members}
    stated = family.get("policy_id")
    if len(policies) != 1 or (stated is not None and policies != {stated}):
        raise DatasetInputError(f"contrast family {family_id} names more than one policy")
    missing = [
        case
        for member, case in zip(family["members"], members, strict=True)
        if member.get("role") == "missing_leg" or case.get("kind") == "missing_leg"
    ]
    if missing:
        return _one_leg_removed(family, members, missing)
    facts = {_fact_key(case) for case in members}
    return len(facts) >= 2


def _one_leg_removed(
    family: Mapping[str, Any], members: Sequence[Mapping[str, Any]], missing: Sequence[Mapping[str, Any]]
) -> bool:
    if len(missing) != 1 or len(members) != 2:
        return False
    allow = missing[0]
    block = next(case for case in members if case is not allow)
    absent = family.get("changed_dimension")
    allow_absent = allow.get("absent_legs")
    if not isinstance(absent, str):
        if isinstance(allow_absent, list) and len(allow_absent) == 1 and isinstance(allow_absent[0], str):
            absent = allow_absent[0]
        else:
            return False
    block_present = block.get("present_legs") or []
    allow_present = allow.get("present_legs") or []
    return (
        allow_absent == [absent]
        and absent not in allow_present
        and not block.get("absent_legs")
        and set(block_present) == set(allow_present) | {absent}
    )


def _fact_key(case: Mapping[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    present = _leg_list(case.get("present_legs")) or []
    absent = _leg_list(case.get("absent_legs")) or []
    return tuple(present), tuple(absent)


def _close_related(
    cases: Mapping[str, Mapping[str, Any]],
    reasons: dict[str, list[str]],
    pairs: Sequence[tuple[str, Sequence[str]]],
    rules: Sequence[PolicyRule],
    by_id: Mapping[str, PolicyRule],
) -> None:
    """Keep a real contrast family whole, and drop copies of a violation that was removed."""
    changed = True
    while changed:
        changed = _close_pairs(reasons, pairs)
        changed = _close_copies(cases, reasons, rules, by_id) or changed


def _close_pairs(reasons: dict[str, list[str]], pairs: Sequence[tuple[str, Sequence[str]]]) -> bool:
    """A complete family is kept whole. One failed member rejects the members that passed."""
    changed = False
    for _family_id, ids in pairs:
        failed = [case_id for case_id in ids if reasons[case_id]]
        if not failed or len(failed) == len(ids):
            continue
        for case_id in ids:
            if not reasons[case_id]:
                reasons[case_id] = ["contrast_pair_broken"]
                changed = True
    return changed


def _close_copies(
    cases: Mapping[str, Mapping[str, Any]],
    reasons: dict[str, list[str]],
    rules: Sequence[PolicyRule],
    by_id: Mapping[str, PolicyRule],
) -> bool:
    """A persona, exception, or labeled boundary does not survive after its violation is removed."""
    changed = False
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for case in cases.values():
        policy_id = case.get("policy_id")
        if isinstance(policy_id, str):
            grouped.setdefault(policy_id, []).append(case)
    for policy_id, grouped_cases in grouped.items():
        rule = by_id.get(policy_id)
        if rule is None or not _violation_removed(grouped_cases, reasons):
            continue
        attack = _attack_facts(rules, rule)
        if attack is None:
            continue
        for case in grouped_cases:
            case_id = str(case["case_id"])
            if reasons.get(case_id) or case.get("kind") == "clear_violation":
                continue
            if _copies_attack(case, attack):
                reasons[case_id] = ["attack_removed"]
                changed = True
    return changed


def _violation_removed(cases: Sequence[Mapping[str, Any]], reasons: Mapping[str, Sequence[str]]) -> bool:
    return any(case.get("kind") == "clear_violation" and reasons.get(str(case["case_id"])) for case in cases)


def _attack_facts(rules: Sequence[PolicyRule], rule: PolicyRule) -> tuple[list[str], list[str]] | None:
    context = case_context(rules, rule, "clear_violation")
    if context is None:
        return None
    present, absent, _action = context
    if not present and not absent:
        return None
    return present, absent


def _copies_attack(case: Mapping[str, Any], attack: tuple[Sequence[str], Sequence[str]]) -> bool:
    present = _leg_list(case.get("present_legs")) or []
    absent = _leg_list(case.get("absent_legs")) or []
    anchor_present, anchor_absent = attack
    if not anchor_present and not anchor_absent:
        return False
    return set(anchor_present) <= set(present) and set(anchor_absent) <= set(absent)


def _judge(
    case: Mapping[str, Any],
    by_id: Mapping[str, PolicyRule],
    rules: Sequence[PolicyRule],
    spans: Mapping[str, str],
) -> tuple[list[str], Any]:
    policy_id = case.get("policy_id")
    rule = by_id.get(policy_id) if isinstance(policy_id, str) else None
    if rule is None:
        return ["unknown_policy"], None
    reasons, action, legs = _label_reasons(case, rule, rules)
    reasons.extend(_evidence_reasons(case, rule, spans))
    reasons.extend(_turn_reasons(case, rule, legs))
    return _unique(reasons), action


def _label_reasons(
    case: Mapping[str, Any], rule: PolicyRule, rules: Sequence[PolicyRule]
) -> tuple[list[str], Any, tuple[list[str], list[str]] | None]:
    kind = case.get("kind")
    present = _leg_list(case.get("present_legs"))
    absent = _leg_list(case.get("absent_legs"))
    if not isinstance(kind, str) or present is None or absent is None:
        return ["invalid_case"], None, None
    condition = case.get("condition_class")
    absent_leg = case.get("absent_leg")
    trigger_leg = case.get("trigger_leg")
    context = case_context(
        rules,
        rule,
        kind,
        condition_class=condition if isinstance(condition, str) else None,
        absent_leg=absent_leg if isinstance(absent_leg, str) else None,
        trigger_leg=trigger_leg if isinstance(trigger_leg, str) else None,
    )
    if context is None:
        return ["invalid_case"], None, None
    expected_present, expected_absent, action = context
    reasons = []
    if present != expected_present or absent != expected_absent:
        reasons.append("facts_mismatch")
    if kind == "exception" and list(case.get("exceptions") or []) != list(rule.exceptions):
        reasons.append("facts_mismatch")
    proposed = case.get("expected_action", _MISSING)
    if proposed is _MISSING or proposed != action:
        reasons.append("label_disagreement")
    elif proposed is not None and proposed not in ACTION_ORDER:
        reasons.append("invalid_case")
    return reasons, action, (expected_present, expected_absent)


def _evidence_reasons(case: Mapping[str, Any], rule: PolicyRule, spans: Mapping[str, str]) -> list[str]:
    evidence = case.get("evidence")
    if not isinstance(evidence, Mapping):
        return ["evidence_missing"]
    quote = evidence.get("source_quote")
    document_id = evidence.get("document_id")
    if not isinstance(quote, str) or not quote.strip() or not isinstance(document_id, str) or not document_id:
        return ["evidence_missing"]
    provenance = rule.provenance
    span_id = evidence.get("source_span_id")
    rule_span = provenance.source_span_id
    reasons: list[str] = []
    if not isinstance(rule_span, str) or not rule_span:
        reasons.append("span_missing")
    elif not isinstance(span_id, str) or not span_id or span_id not in spans:
        reasons.append("span_missing")
    if "document_sha256" not in evidence:
        sha_differs = provenance.document_sha256 is not None
    else:
        sha_differs = evidence.get("document_sha256") != provenance.document_sha256
    mismatched = (
        canonical_text(quote) != canonical_text(provenance.source_quote)
        or document_id != provenance.document_id
        or (isinstance(rule_span, str) and rule_span and span_id != rule_span)
        or sha_differs
    )
    if mismatched:
        reasons.append("evidence_mismatch")
    if isinstance(span_id, str) and span_id in spans and not quote_in_span(quote, spans[span_id]):
        reasons.append("quote_not_in_span")
    return reasons


def _turn_reasons(
    case: Mapping[str, Any], rule: PolicyRule, legs: tuple[list[str], list[str]] | None
) -> list[str]:
    turns = case.get("turns")
    if not isinstance(turns, list) or not turns:
        return ["invalid_turns"]
    kind = case.get("kind")
    scored = 0
    for index, turn in enumerate(turns, start=1):
        if not _turn_shape(turn, index):
            return ["invalid_turns"]
        if turn["scored"]:
            scored += 1
            if turn["role"] != "user" or "expected_action" not in turn:
                return ["invalid_turns"]
            if turn["expected_action"] != case.get("expected_action", _MISSING):
                return ["turn_label_disagreement"]
        if legs is not None and isinstance(kind, str) and _text_differs(turn, rule, legs, kind, index, len(turns)):
            return ["turn_text_mismatch"]
    if kind == "multi_turn":
        return [] if scored >= 2 else ["invalid_turns"]
    return [] if scored == 1 else ["invalid_turns"]


def _text_differs(
    turn: Mapping[str, Any],
    rule: PolicyRule,
    legs: tuple[list[str], list[str]],
    kind: str,
    number: int,
    total: int,
) -> bool:
    present, absent = legs
    exceptions = list(rule.exceptions) if kind == "exception" else []
    expected = seed_sentence(rule, present, absent, number, total, kind=kind, exceptions=exceptions)
    text = turn.get("text")
    return not isinstance(text, str) or canonical_text(text) != canonical_text(expected)


def _turn_shape(turn: Any, index: int) -> bool:
    if not isinstance(turn, Mapping):
        return False
    number = turn.get("turn")
    text = turn.get("text")
    return (
        number == index
        and not isinstance(number, bool)
        and turn.get("role") in _ROLES
        and isinstance(text, str)
        and bool(text.strip())
        and isinstance(turn.get("scored"), bool)
    )


def _leg_list(value: Any) -> list[str] | None:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return None
    if any(not isinstance(leg, str) for leg in value):
        return None
    return list(value)


def _deduplicate(
    cases: Sequence[Mapping[str, Any]],
    pairs: Sequence[tuple[str, Sequence[str]]],
    by_id: Mapping[str, PolicyRule],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    surviving = {str(case["case_id"]) for case in cases}
    protected: set[str] = set()
    for _family_id, ids in pairs:
        if all(case_id in surviving for case_id in ids):
            protected.update(ids)
    buckets: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for case in sorted(cases, key=lambda item: str(item["case_id"])):
        buckets.setdefault(_signature(case, by_id), []).append(case)
    kept: list[dict[str, Any]] = []
    drops: list[dict[str, Any]] = []
    for rows in buckets.values():
        if len(rows) == 1:
            kept.append(dict(rows[0]))
            continue
        guarded = [row for row in rows if str(row["case_id"]) in protected]
        free = [row for row in rows if str(row["case_id"]) not in protected]
        if guarded:
            kept.extend(dict(row) for row in guarded)
            keeper = str(guarded[0]["case_id"])
            drops.extend(_duplicate(row, keeper) for row in free)
        else:
            kept.append(dict(free[0]))
            keeper = str(free[0]["case_id"])
            drops.extend(_duplicate(row, keeper) for row in free[1:])
    kept.sort(key=lambda item: str(item["case_id"]))
    return kept, drops


def _signature(case: Mapping[str, Any], by_id: Mapping[str, PolicyRule]) -> tuple[Any, ...]:
    action = case.get("expected_action")
    turns = case.get("turns")
    texts = [turn.get("text", "") for turn in turns] if isinstance(turns, list) else []
    text = canonical_text("\n".join(text for text in texts if isinstance(text, str)))
    present = _leg_list(case.get("present_legs")) or []
    absent = _leg_list(case.get("absent_legs")) or []
    return (
        case.get("policy_id"),
        _case_language(case, _rule_for(case, by_id)),
        "" if action is None else action,
        case.get("kind"),
        tuple(present),
        tuple(absent),
        text,
    )


def _case_language(case: Mapping[str, Any], rule: PolicyRule | None) -> str:
    language = case.get("language")
    if isinstance(language, str) and language.strip():
        return language
    languages = case.get("languages")
    if isinstance(languages, list) and languages and all(isinstance(item, str) for item in languages):
        return ",".join(languages)
    if rule is None:
        return ""
    for name in rule.scope.languages:
        if isinstance(name, str) and name.casefold() != "unspecified":
            return name
    return ""


def _rule_for(case: Mapping[str, Any], by_id: Mapping[str, PolicyRule]) -> PolicyRule | None:
    policy_id = case.get("policy_id")
    if not isinstance(policy_id, str):
        return None
    return by_id.get(policy_id)


def _annotate(case: Mapping[str, Any], by_id: Mapping[str, PolicyRule]) -> dict[str, Any]:
    """Copy the case and record the source language when the row does not name one."""
    stored = dict(case)
    rule = _rule_for(stored, by_id)
    language = _case_language(stored, rule)
    listed = stored.get("languages")
    has_list = isinstance(listed, list) and bool(listed)
    if language and not has_list and not (isinstance(stored.get("language"), str) and stored["language"].strip()):
        stored["language"] = language
    return stored


def _duplicate(case: Mapping[str, Any], keeper: str) -> dict[str, Any]:
    return {
        "case_id": case.get("case_id"),
        "policy_id": case.get("policy_id") if isinstance(case.get("policy_id"), str) else None,
        "reasons": ["exact_duplicate"],
        "stage": "dedup",
        "kept_case_id": keeper,
    }


def _manifest(
    cases: Sequence[Mapping[str, Any]],
    drops: Sequence[Mapping[str, Any]],
    pairs: Sequence[tuple[str, Sequence[str]]],
    by_id: Mapping[str, PolicyRule],
) -> dict[str, Any]:
    groups = _groups(cases, pairs)
    ordered = sorted(
        ((min(members), sorted(members)) for members in groups.values()),
        key=lambda item: (-len(item[1]), item[0]),
    )
    splits: dict[str, list[str]] = {"development": [], "private_test": []}
    group_rows: list[dict[str, Any]] = []
    if not ordered:
        reason = "no_cases"
    elif len(ordered) == 1:
        reason = "one_group"
        group_id, members = ordered[0]
        splits["development"] = members
        group_rows.append({"group_id": group_id, "split": "development", "case_ids": members})
    else:
        reason = "complete"
        assignments = _balanced_assignments(ordered, cases, by_id)
        for (group_id, members), name in zip(ordered, assignments, strict=True):
            splits[name].extend(members)
            group_rows.append({"group_id": group_id, "split": name, "case_ids": members})
        for name in splits:
            splits[name] = sorted(splits[name])
    case_by_id = {str(case["case_id"]): case for case in cases}
    split_cases = {
        name: [case_by_id[case_id] for case_id in case_ids]
        for name, case_ids in splits.items()
    }
    return {
        "complete": reason == "complete",
        "reason": reason,
        "splits": splits,
        "groups": group_rows,
        "dropped_duplicates": [
            {"case_id": row["case_id"], "kept_case_id": row["kept_case_id"]}
            for row in sorted(drops, key=lambda item: str(item["case_id"]))
        ],
        "balance": {
            "by_policy": _tally(cases, lambda case: str(case.get("policy_id"))),
            "by_action": _tally(cases, _action_name),
            "by_kind": _tally(cases, lambda case: str(case.get("kind"))),
            "by_language": _tally(cases, lambda case: _case_language(case, _rule_for(case, by_id))),
            "by_split": {
                name: {
                    "cases": len(rows),
                    "by_policy": _tally(rows, lambda case: str(case.get("policy_id"))),
                    "by_action": _tally(rows, _action_name),
                    "by_kind": _tally(rows, lambda case: str(case.get("kind"))),
                    "by_language": _tally(
                        rows, lambda case: _case_language(case, _rule_for(case, by_id))
                    ),
                }
                for name, rows in split_cases.items()
            },
        },
        "semantic_dedup": "not_configured",
        "public_test": "not_assigned",
    }


def _balanced_assignments(
    groups: Sequence[tuple[str, Sequence[str]]],
    cases: Sequence[Mapping[str, Any]],
    by_id: Mapping[str, PolicyRule],
) -> list[str]:
    """Greedily balance case count, policy, action, kind, and language without splitting groups."""
    case_by_id = {str(case["case_id"]): case for case in cases}
    features = [_group_features(ids, case_by_id, by_id) for _group_id, ids in groups]
    total = Counter[str]()
    for counts in features:
        total.update(counts)
    current = {"development": Counter[str](), "private_test": Counter[str]()}
    assignments: list[str] = []
    for index, counts in enumerate(features):
        if index == 0:
            name = "development"
        elif index == 1:
            name = "private_test"
        else:
            scores = {
                split: sum(abs(2 * (current[split][key] + counts[key]) - total[key]) for key in total)
                for split in current
            }
            name = min(
                current,
                key=lambda split: (scores[split], current[split]["cases"], split),
            )
        assignments.append(name)
        current[name].update(counts)
    return assignments


def _group_features(
    ids: Sequence[str],
    case_by_id: Mapping[str, Mapping[str, Any]],
    by_id: Mapping[str, PolicyRule],
) -> Counter[str]:
    counts = Counter({"cases": len(ids)})
    for case_id in ids:
        case = case_by_id[case_id]
        counts[f"policy:{case.get('policy_id')}"] += 1
        counts[f"action:{_action_name(case)}"] += 1
        counts[f"kind:{case.get('kind')}"] += 1
        counts[f"language:{_case_language(case, _rule_for(case, by_id))}"] += 1
    return counts


def _groups(
    cases: Sequence[Mapping[str, Any]], pairs: Sequence[tuple[str, Sequence[str]]]
) -> dict[str, list[str]]:
    """Union a contrast pair with the variants that copy the same violation."""
    parent = {str(case["case_id"]): str(case["case_id"]) for case in cases}

    def find(case_id: str) -> str:
        while parent[case_id] != case_id:
            parent[case_id] = parent[parent[case_id]]
            case_id = parent[case_id]
        return case_id

    def union(left: str, right: str) -> None:
        root_left, root_right = find(left), find(right)
        if root_left == root_right:
            return
        if root_left < root_right:
            parent[root_right] = root_left
        else:
            parent[root_left] = root_right

    anchors: dict[str, Mapping[str, Any]] = {}
    for case in cases:
        policy_id = case.get("policy_id")
        if not isinstance(policy_id, str) or case.get("kind") != "clear_violation":
            continue
        current = anchors.get(policy_id)
        if current is None or str(case["case_id"]) < str(current["case_id"]):
            anchors[policy_id] = case
    for case in cases:
        policy_id = case.get("policy_id")
        if not isinstance(policy_id, str):
            continue
        anchor = anchors.get(policy_id)
        case_id = str(case["case_id"])
        if anchor is not None and case_id != str(anchor["case_id"]):
            attack = (_leg_list(anchor.get("present_legs")) or [], _leg_list(anchor.get("absent_legs")) or [])
            if case.get("kind") in _WITH_VIOLATION or _copies_attack(case, attack):
                union(str(anchor["case_id"]), case_id)
            continue
        if anchor is None and case.get("kind") in _WITH_VIOLATION:
            attached = [
                str(other["case_id"])
                for other in cases
                if other.get("policy_id") == policy_id and other.get("kind") in _WITH_VIOLATION
            ]
            if attached:
                head = min(attached)
                if case_id != head:
                    union(head, case_id)
    surviving = set(parent)
    for _family_id, ids in pairs:
        present = [case_id for case_id in ids if case_id in surviving]
        if len(present) == len(tuple(ids)) and len(present) >= 2:
            for case_id in present[1:]:
                union(present[0], case_id)
    groups: dict[str, list[str]] = {}
    for case_id in parent:
        groups.setdefault(find(case_id), []).append(case_id)
    return groups


def _report(
    reasons: Mapping[str, Sequence[str]],
    validated: Sequence[Mapping[str, Any]],
    rejected: Sequence[Mapping[str, Any]],
    deduplicated: Sequence[Mapping[str, Any]],
    pairs: Sequence[tuple[str, Sequence[str]]],
    drops: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    deduped_ids = {str(case["case_id"]) for case in deduplicated}
    kept_pairs = sum(1 for _family_id, ids in pairs if all(case_id in deduped_ids for case_id in ids))
    counts = Counter(reason for row in rejected for reason in row["reasons"])
    disagreements = sum(1 for found in reasons.values() if "label_disagreement" in found)
    validation_rejected = sum(1 for found in reasons.values() if found)
    return {
        "checked": len(reasons),
        "validated": len(validated),
        "rejected": validation_rejected,
        "duplicates_removed": len(drops),
        "reasons": {key: counts[key] for key in sorted(counts)},
        "judge": {"method": "deterministic", "model": None, "disagreements": disagreements},
        "contrast_pairs_kept": kept_pairs,
        "contrast_pairs_rejected": len(pairs) - kept_pairs,
        "semantic_dedup": "not_configured",
    }


def _action_name(case: Mapping[str, Any]) -> str:
    action = case.get("expected_action")
    return "unlabeled" if action is None else str(action)


def _missing_case(family_id: str, case_id: Any) -> str:
    return f"contrast family {family_id} names case {case_id}, which is not in the candidates"


def _tally(cases: Sequence[Mapping[str, Any]], key: Any) -> dict[str, int]:
    counts = Counter(key(case) for case in cases)
    return {name: counts[name] for name in sorted(counts)}


def _label_row(case: Mapping[str, Any], action: Any, reasons: Sequence[str]) -> dict[str, Any]:
    proposed = case.get("expected_action") if "expected_action" in case else None
    return {
        "case_id": case.get("case_id"),
        "policy_id": case.get("policy_id") if isinstance(case.get("policy_id"), str) else None,
        "proposed_action": proposed,
        "aggregated_action": action,
        "status": "rejected" if reasons else "confirmed",
        "reasons": list(reasons),
    }


def _rejection(case: Mapping[str, Any], reasons: Sequence[str], stage: str) -> dict[str, Any]:
    return {
        "case_id": case.get("case_id"),
        "policy_id": case.get("policy_id") if isinstance(case.get("policy_id"), str) else None,
        "reasons": list(reasons),
        "stage": stage,
    }


def _unique(reasons: Sequence[str]) -> list[str]:
    found: list[str] = []
    for reason in reasons:
        if reason not in found:
            found.append(reason)
    return found
