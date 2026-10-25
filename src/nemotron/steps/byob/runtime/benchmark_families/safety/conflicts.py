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

"""Mechanical conflict checks between policy rules, run before the review packet.

Two rules that demand different actions for the same facts are normally settled by the fixed
action order, and that settlement is recorded with ``resolution: action_order``. A conflict
is unresolved when the order would silently override what a source says:

- ``contradictory_rules``: one rule allows exactly the situation another rule acts on.
- ``exception_contradicts_condition``: a rule lists the same situation as a condition and
  as an exception, so it can never be read one way.
- ``definition_conflict``: the rule's evidence uses a term defined differently in two places.

An unresolved conflict sends each rule involved to review. On a critical rule it quarantines
the rule, and only the rules involved are affected.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from nemotron.steps.byob.runtime.benchmark_families.safety.actions import (
    ACTION_ORDER,
    aggregate,
    rule_matches,
    rules_overlap,
)
from nemotron.steps.byob.runtime.benchmark_families.safety.confidence import quote_in_span
from nemotron.steps.byob.runtime.benchmark_families.safety.ir import PolicyRule

RESOLVED_BY_ACTION_ORDER = "action_order"
UNRESOLVED = "unresolved"


@dataclass
class ConflictResult:
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    graph: list[dict[str, Any]] = field(default_factory=list)

    def reasons(self, rules: Mapping[str, PolicyRule]) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
        """Decision and quarantine reasons per policy id."""
        decision: dict[str, list[str]] = {}
        quarantine: dict[str, list[str]] = {}
        for conflict in self.conflicts:
            for policy_id in conflict["policy_ids"]:
                rule = rules.get(policy_id)
                if rule is None:
                    continue
                reason = f"unresolved conflict ({conflict['type']}): {conflict['detail']}"
                target = quarantine if rule.severity == "critical" else decision
                target.setdefault(policy_id, []).append(reason)
        return decision, quarantine


def detect_conflicts(
    rules: Sequence[PolicyRule],
    *,
    term_usage: Mapping[str, Iterable[str]] | None = None,
    glossary_conflicts: Sequence[Mapping[str, Any]] = (),
) -> ConflictResult:
    """Evaluate each rule's own triggers, and the combined triggers of related rule pairs.

    Two rules are related when they share a condition name, a source section, or an object
    that both quotes actually use. A missing section does not count as the same section, and
    an object the quote never states does not relate rules. ``term_usage`` maps a policy id
    to the glossary terms its evidence uses.
    """
    result = ConflictResult()
    live = sorted((rule for rule in rules if rule.status != "rejected"), key=lambda rule: rule.policy_id)
    by_id = {rule.policy_id: rule for rule in live}
    seen_nodes: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
    seen_conflicts: set[tuple[str, tuple[str, ...], str]] = set()
    self_conflicts = {rule.policy_id: set(_self_contradictions(rule)) for rule in live}
    for facts in _fact_sets(live):
        matches = [match for rule in live for match in rule_matches(rule, facts)]
        actions = {match.action for match in matches}
        ids = tuple(sorted({match.policy_id for match in matches}))
        if len(actions) < 2 or not _share_scope([by_id[pid] for pid in ids]):
            continue
        key = (tuple(sorted(facts)), ids)
        if key in seen_nodes:
            continue
        seen_nodes.add(key)
        decision = aggregate(matches)
        contradiction = _contradiction(by_id, facts, matches)
        unresolved = contradiction is not None or any(facts in self_conflicts[pid] for pid in ids)
        result.graph.append(
            {
                "case_context": sorted(facts),
                "matched_policy_ids": list(ids),
                "actions": [action for action in ACTION_ORDER if action in actions],
                "expected_action": decision.action,
                "primary_policy_id": decision.primary_policy_id,
                "resolution": UNRESOLVED if unresolved else RESOLVED_BY_ACTION_ORDER,
            }
        )
        if contradiction:
            allowing, acting = contradiction
            _add(
                result,
                seen_conflicts,
                "contradictory_rules",
                sorted({allowing, *acting}),
                f"{allowing} allows exactly the situation {', '.join(acting)} acts on: {', '.join(sorted(facts))}",
                sorted(facts),
            )
    for rule in live:
        for facts in sorted(self_conflicts[rule.policy_id], key=lambda item: sorted(item)):
            _add(
                result,
                seen_conflicts,
                "contradictory_rules",
                [rule.policy_id],
                f"{rule.policy_id} both allows and acts on: {', '.join(sorted(facts))}",
                sorted(facts),
            )
        overlap = sorted(
            set(rule.exceptions) & {leg for group in rule.rule.conditions().values() for leg in group.legs}
        )
        if overlap:
            _add(
                result,
                seen_conflicts,
                "exception_contradicts_condition",
                [rule.policy_id],
                f"{', '.join(overlap)} is both a condition and an exception of {rule.policy_id}",
                overlap,
            )
    conflicting_terms = {entry["key"]: entry for entry in glossary_conflicts}
    for rule in live:
        used = sorted(set((term_usage or {}).get(rule.policy_id, ())) & set(conflicting_terms))
        for term in used:
            places = ", ".join(
                item["source_span_id"]
                for item in conflicting_terms[term].get("definitions", [])
                if isinstance(item, dict) and isinstance(item.get("source_span_id"), str)
            )
            if not places:
                continue
            _add(
                result,
                seen_conflicts,
                "definition_conflict",
                [rule.policy_id],
                f'"{term}" is defined differently in {places}',
                [term],
            )
    return result


def _trigger_sets(rule: PolicyRule, names: Sequence[str] | None = None) -> list[frozenset[str]]:
    """The smallest fact sets that make each stored condition of a rule fire."""
    sets: list[frozenset[str]] = []
    for name, group in rule.rule.conditions().items():
        if names is not None and name not in names:
            continue
        if group.operator == "all_of":
            sets.append(frozenset(group.legs))
        elif group.operator == "any_of":
            sets.extend(frozenset([leg]) for leg in group.legs)
        else:
            sets.append(frozenset())
    return list(dict.fromkeys(sets))


def _self_contradictions(rule: PolicyRule) -> list[frozenset[str]]:
    """Triggers a rule both allows and acts on."""
    acting = ("block_if", "escalate_if", "require_confirmation_if")
    overlap = set(_trigger_sets(rule, ("allow_if",))) & set(_trigger_sets(rule, acting))
    return sorted(overlap, key=lambda item: sorted(item))


def _share_scope(rules: Sequence[PolicyRule]) -> bool:
    """True when every rule can apply to the same scoped event."""
    return all(
        rules_overlap(first, second)
        for position, first in enumerate(rules)
        for second in rules[position + 1 :]
    )


def _fact_sets(rules: Sequence[PolicyRule]) -> list[frozenset[str]]:
    triggers = {rule.policy_id: _trigger_sets(rule) for rule in rules}
    sets = [facts for rule in rules for facts in triggers[rule.policy_id]]
    for position, first in enumerate(rules):
        for second in rules[position + 1 :]:
            if _related(first, second):
                sets.extend(a | b for a in triggers[first.policy_id] for b in triggers[second.policy_id])
    return list(dict.fromkeys(sets))


def _related(first: PolicyRule, second: PolicyRule) -> bool:
    if _condition_names(first) & _condition_names(second):
        return True
    section = first.provenance.section
    same_section = bool(section) and (first.provenance.document_id, section) == (
        second.provenance.document_id,
        second.provenance.section,
    )
    subject = _stated_subject(first)
    return same_section or (subject is not None and subject == _stated_subject(second))


def _condition_names(rule: PolicyRule) -> set[str]:
    return {leg for group in rule.rule.conditions().values() for leg in group.legs}


def _stated_subject(rule: PolicyRule) -> str | None:
    """The rule object, when the quote itself uses that word. A guessed object does not count."""
    subject = " ".join(rule.rule.object.casefold().split())
    if not subject or subject == "unspecified":
        return None
    if quote_in_span(subject, rule.provenance.source_quote.casefold()):
        return subject
    return None


def _contradiction(
    by_id: Mapping[str, PolicyRule], facts: frozenset[str], matches: Sequence[Any]
) -> tuple[str, list[str]] | None:
    """An allow condition and an acting condition with exactly the same trigger, from different rules."""
    allowing = sorted(
        match.policy_id
        for match in matches
        if match.action == "allow" and _has_exact_trigger(by_id[match.policy_id], "allow_if", facts)
    )
    acting = sorted(
        {
            match.policy_id
            for match in matches
            if match.action != "allow" and _exact_acting_trigger(by_id[match.policy_id], facts)
        }
        - set(allowing)
    )
    if not allowing or not acting:
        return None
    allow_group = by_id[allowing[0]].rule.conditions().get("allow_if")
    if allow_group is not None and allow_group.operator == "none_of":
        allow_legs = frozenset(allow_group.legs)
        acting = [policy_id for policy_id in acting if _none_of_acting_legs(by_id[policy_id]) == allow_legs]
        if not acting:
            return None
    return allowing[0], acting


def _none_of_acting_legs(rule: PolicyRule) -> frozenset[str] | None:
    """The none_of legs an acting condition uses, when every such group names the same set."""
    found: frozenset[str] | None = None
    for name in ("block_if", "escalate_if", "require_confirmation_if"):
        group = rule.rule.conditions().get(name)
        if group is None or group.operator != "none_of":
            continue
        legs = frozenset(group.legs)
        if found is not None and found != legs:
            return None
        found = legs
    return found


def _has_exact_trigger(rule: PolicyRule, name: str, facts: frozenset[str]) -> bool:
    """True when ``facts`` is that group's own minimal witness, not merely a firing event.

    ``none_of`` fires on any facts that omit its legs, including another rule's unrelated
    trigger. The exact witness is the empty set ``_trigger_sets`` already records.
    """
    return facts in _trigger_sets(rule, (name,))


def _exact_acting_trigger(rule: PolicyRule, facts: frozenset[str]) -> bool:
    return any(
        _has_exact_trigger(rule, name, facts) for name in ("block_if", "escalate_if", "require_confirmation_if")
    )


def _add(
    result: ConflictResult,
    seen: set[tuple[str, tuple[str, ...], str]],
    kind: str,
    policy_ids: list[str],
    detail: str,
    context: list[str],
) -> None:
    key = (kind, tuple(policy_ids), detail)
    if key in seen:
        return
    seen.add(key)
    result.conflicts.append(
        {"type": kind, "policy_ids": policy_ids, "detail": detail, "case_context": context, "resolution": UNRESOLVED}
    )
