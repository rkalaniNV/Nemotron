"""Mutation operators for the assertion gate.

A0 established that the banking_vn pipeline drops nothing: every expanded task
replays, every assertion passes, publish rate is 100%. That makes the existing
measurement blind to assertion strength — a pack whose assertions are `return None`
would score identically. The gate closes that hole by corrupting a known-good
episode and asking the pack's own assertions whether they notice.

Two mutation modes, because they answer different questions:

  reexecute   the corrupted call sequence is actually run against the backend, so
              state and trace move together. This is the realistic agent-failure
              model: an agent that skips a call also fails to change the world.
  trace       the real episode runs, then the assertion is handed a doctored trace
              through the `run_assertion` trace override. State and trace disagree,
              which isolates what the assertion reads from what actually happened.

`state_reset` is the third mode and the sharpest one: real trace, but state as if
nothing ran. An assertion that only reads `trace` cannot tell the difference.

Every operator is deterministic. A mutation that would not change anything is not
emitted at all, so "not applicable" never masquerades as "not detected".
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

REEXECUTE = "reexecute"
TRACE = "trace"
STATE_RESET = "state_reset"

CALL_LEVEL = "call_level"
ARGUMENT_LEVEL = "argument_level"
STATE_LEVEL = "state_level"

# Money fields the operators perturb. The plan names the first three; the fourth is
# included because without it `assert_card_limit_reported` would have no
# argument-level coverage at all and its row would read as vacuously strong.
NUMERIC_LEAVES = ("amount_vnd", "balance_vnd", "fee_vnd", "remaining_limit_vnd")

# One order of magnitude above the largest fixture balance, so the large-delta
# variant cannot be confused with rounding.
LARGE_DELTA = 100_000_000

# Argument/result fields that name an entity, mapped to the fixture collection and
# key a valid replacement is drawn from.
ID_SOURCES = {
    "account_id": ("accounts", "account_id"),
    "from_account_id": ("accounts", "account_id"),
    "card_id": ("cards", "card_id"),
    "transaction_id": ("transactions", "transaction_id"),
    "dispute_id": ("disputes", "dispute_id"),
    "payment_ref": ("vietqr_payments", "payment_ref"),
}


@dataclass(frozen=True)
class Mutation:
    """One corruption of one episode, plus how to execute it."""

    operator: str
    op_class: str
    mode: str
    detail: str
    calls: tuple[dict[str, Any], ...] | None = None
    trace: tuple[dict[str, Any], ...] | None = None


@dataclass
class PackContext:
    """The pack facts the operators need to build a *valid* corruption.

    Swapping an id for a random string only tests the backend's validator. Swapping
    it for another id that really exists tests whether the assertion checks which
    entity it got, which is the property under measurement.
    """

    tools: list[dict[str, Any]]
    fixtures: dict[str, Any]
    tool_parameters: dict[str, set[str]] = field(default_factory=dict)
    mutating_tools: frozenset[str] = frozenset()
    id_pools: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def from_pack(cls, tools: list[dict[str, Any]], fixtures: dict[str, Any]) -> PackContext:
        parameters: dict[str, set[str]] = {}
        mutating: set[str] = set()
        for tool in tools:
            function = tool.get("function") or {}
            name = str(function.get("name"))
            properties = ((function.get("parameters") or {}).get("properties") or {}).keys()
            parameters[name] = set(properties)
            # x-mutates sits at the top level of the tool dict, not inside "function".
            if tool.get("x-mutates"):
                mutating.add(name)
        pools: dict[str, list[str]] = {}
        for field_name, (collection, key) in ID_SOURCES.items():
            values = sorted(
                {
                    str(row[key])
                    for row in fixtures.get(collection) or []
                    if isinstance(row, dict) and isinstance(row.get(key), str)
                }
            )
            pools[field_name] = values
        return cls(
            tools=tools,
            fixtures=fixtures,
            tool_parameters=parameters,
            mutating_tools=frozenset(mutating),
            id_pools=pools,
        )

    def other_id(self, field_name: str, current: Any) -> str | None:
        """A different but real id of the same entity, chosen deterministically."""
        for candidate in self.id_pools.get(field_name) or []:
            if candidate != current:
                return candidate
        return None

    def nearest_tool(self, name: str, arguments: dict[str, Any]) -> str | None:
        """The wrong tool most likely to be *accepted* by the same arguments.

        Picking a random other tool mostly produces an invalid_argument error, which
        any assertion that looks at `error` catches for free. Picking the tool with
        the most overlapping parameters produces a call that succeeds and returns the
        wrong kind of answer, which is the mutation worth measuring.
        """
        keys = set(arguments)
        best: tuple[int, str] | None = None
        for other in sorted(self.tool_parameters):
            if other == name:
                continue
            score = len(keys & self.tool_parameters[other])
            if best is None or score > best[0]:
                best = (score, other)
        return None if best is None else best[1]

    def probe_call(self, calls: list[dict[str, Any]]) -> dict[str, Any] | None:
        """A read-only call the episode did not already make.

        Used by `inject_extra_call`: the dual of dropping a call, and the only
        operator that reaches a task whose expected trace is empty.
        """
        accounts = self.id_pools.get("account_id") or []
        used = {
            str((call.get("arguments") or {}).get("account_id"))
            for call in calls
            if call["function_name"] == "get_account_balance"
        }
        for account_id in accounts:
            if account_id not in used:
                turn = calls[-1]["turn_index"] if calls else 0
                return {
                    "function_name": "get_account_balance",
                    "arguments": {"account_id": account_id},
                    "turn_index": turn,
                }
        return None


def _numeric_paths(result: dict[str, Any]) -> dict[str, tuple[Any, ...]]:
    """First location of each money field, root before nested rows.

    Only the first row of a nested list is considered: perturbing row 4 instead of
    row 0 exercises the same predicate and would inflate the trial count without
    adding a distinct question.
    """
    paths: dict[str, tuple[Any, ...]] = {}
    for leaf in NUMERIC_LEAVES:
        if isinstance(result.get(leaf), int) and not isinstance(result.get(leaf), bool):
            paths[leaf] = (leaf,)
    for key in sorted(result):
        rows = result[key]
        if not isinstance(rows, list) or not rows or not isinstance(rows[0], dict):
            continue
        for leaf in NUMERIC_LEAVES:
            value = rows[0].get(leaf)
            if leaf not in paths and isinstance(value, int) and not isinstance(value, bool):
                paths[leaf] = (key, 0, leaf)
    return paths


def _set_path(container: Any, path: tuple[Any, ...], value: Any) -> None:
    node = container
    for step in path[:-1]:
        node = node[step]
    node[path[-1]] = value


def _get_path(container: Any, path: tuple[Any, ...]) -> Any:
    node = container
    for step in path:
        node = node[step]
    return node


def _as_steps(calls: list[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "function_name": call["function_name"],
            "arguments": copy.deepcopy(call.get("arguments") or {}),
            "turn_index": int(call.get("turn_index", 0)),
        }
        for call in calls
    )


def build_mutations(
    *,
    calls: list[dict[str, Any]],
    trace: list[dict[str, Any]],
    context: PackContext,
    state_changed: bool,
) -> list[Mutation]:
    """Every applicable single-point corruption of one replayed episode.

    `trace` is the trace the worker built from `calls`, so the two are index-aligned;
    `state_changed` says whether those calls moved the backend at all, which is what
    makes the state-level operator applicable.
    """
    mutations: list[Mutation] = []
    mutations.extend(_call_level(calls, context))
    mutations.extend(_argument_level(calls, trace, context))
    if state_changed:
        mutations.append(
            Mutation(
                operator="state_reverted",
                op_class=STATE_LEVEL,
                mode=STATE_RESET,
                detail="real trace, state rolled back to the post-reset fixtures",
                trace=tuple(copy.deepcopy(trace)),
            )
        )
    return mutations


def _call_level(calls: list[dict[str, Any]], context: PackContext) -> list[Mutation]:
    mutations: list[Mutation] = []
    for index, call in enumerate(calls):
        name = call["function_name"]

        remaining = calls[:index] + calls[index + 1 :]
        mutations.append(
            Mutation(
                operator="drop_call",
                op_class=CALL_LEVEL,
                mode=REEXECUTE,
                detail=f"dropped call[{index}] {name}",
                calls=_as_steps(remaining),
            )
        )

        replacement = context.nearest_tool(name, call.get("arguments") or {})
        if replacement is not None:
            swapped = copy.deepcopy(calls)
            swapped[index]["function_name"] = replacement
            mutations.append(
                Mutation(
                    operator="swap_tool",
                    op_class=CALL_LEVEL,
                    mode=REEXECUTE,
                    detail=f"call[{index}] {name} -> {replacement}",
                    calls=_as_steps(swapped),
                )
            )

        duplicated = calls[: index + 1] + [copy.deepcopy(call)] + calls[index + 1 :]
        # Split by tool kind: repeating an idempotent read is not a defect, so a
        # survivor there is not evidence of a weak assertion. Repeating a mutating
        # call is the classic double-commit bug and a survivor there is.
        mutates = name in context.mutating_tools
        mutations.append(
            Mutation(
                operator="duplicate_call_mutating" if mutates else "duplicate_call_readonly",
                op_class=CALL_LEVEL,
                mode=REEXECUTE,
                detail=f"duplicated call[{index}] {name}",
                calls=_as_steps(duplicated),
            )
        )

    # Order only carries meaning across call groups: two calls sharing a group were
    # declared parallel, so swapping them changes nothing the pack asserted. Swapping
    # across a group boundary violates `call_order: strict` — the one field A1 found no
    # schema can derive — which makes this the operator most likely to expose an
    # assertion that never checked sequence at all.
    for index in range(len(calls) - 1):
        if calls[index].get("call_group") == calls[index + 1].get("call_group"):
            continue
        swapped = copy.deepcopy(calls)
        swapped[index], swapped[index + 1] = swapped[index + 1], swapped[index]
        # Turn indices stay positional: the corruption is the order the calls were
        # issued in, not a claim that the assistant took a different number of turns.
        for position, call_ in enumerate(swapped):
            call_["turn_index"] = int(calls[position].get("turn_index", 0))
        mutations.append(
            Mutation(
                operator="reorder_calls",
                op_class=CALL_LEVEL,
                mode=REEXECUTE,
                detail=(
                    f"swapped call[{index}] {calls[index]['function_name']} with "
                    f"call[{index + 1}] {calls[index + 1]['function_name']}"
                ),
                calls=_as_steps(swapped),
            )
        )

    extra = context.probe_call(calls)
    if extra is not None:
        mutations.append(
            Mutation(
                operator="inject_extra_call",
                op_class=CALL_LEVEL,
                mode=REEXECUTE,
                detail=f"appended unrequested {extra['function_name']}({extra['arguments']})",
                calls=_as_steps(calls + [extra]),
            )
        )
    return mutations


def _argument_level(
    calls: list[dict[str, Any]],
    trace: list[dict[str, Any]],
    context: PackContext,
) -> list[Mutation]:
    mutations: list[Mutation] = []
    for index, entry in enumerate(trace):
        result = entry.get("result")
        if not isinstance(result, dict):
            continue

        for leaf, path in sorted(_numeric_paths(result).items()):
            original = _get_path(result, path)
            for operator, delta in (
                ("perturb_numeric_plus_one", 1),
                ("perturb_numeric_large", LARGE_DELTA),
            ):
                mutated = copy.deepcopy(trace)
                _set_path(mutated[index]["result"], path, original + delta)
                mutations.append(
                    Mutation(
                        operator=operator,
                        op_class=ARGUMENT_LEVEL,
                        mode=TRACE,
                        detail=f"result[{index}].{leaf} {original} -> {original + delta}",
                        trace=tuple(mutated),
                    )
                )

        for id_field in sorted(set(result) & set(ID_SOURCES)):
            replacement = context.other_id(id_field, result[id_field])
            if replacement is None:
                continue
            mutated = copy.deepcopy(trace)
            mutated[index]["result"][id_field] = replacement
            mutations.append(
                Mutation(
                    operator="swap_identity_result",
                    op_class=ARGUMENT_LEVEL,
                    mode=TRACE,
                    detail=f"result[{index}].{id_field} {result[id_field]} -> {replacement}",
                    trace=tuple(mutated),
                )
            )

    for index, call in enumerate(calls):
        arguments = call.get("arguments") or {}
        for id_field in sorted(set(arguments) & set(ID_SOURCES)):
            replacement = context.other_id(id_field, arguments[id_field])
            if replacement is None:
                continue
            swapped = copy.deepcopy(calls)
            swapped[index]["arguments"][id_field] = replacement
            mutations.append(
                Mutation(
                    operator="swap_identity_argument",
                    op_class=ARGUMENT_LEVEL,
                    mode=REEXECUTE,
                    detail=f"call[{index}].{id_field} {arguments[id_field]} -> {replacement}",
                    calls=_as_steps(swapped),
                )
            )
    return mutations


OPERATOR_CLASSES = {
    "drop_call": CALL_LEVEL,
    "swap_tool": CALL_LEVEL,
    "duplicate_call_readonly": CALL_LEVEL,
    "duplicate_call_mutating": CALL_LEVEL,
    "inject_extra_call": CALL_LEVEL,
    "reorder_calls": CALL_LEVEL,
    "perturb_numeric_plus_one": ARGUMENT_LEVEL,
    "perturb_numeric_large": ARGUMENT_LEVEL,
    "swap_identity_result": ARGUMENT_LEVEL,
    "swap_identity_argument": ARGUMENT_LEVEL,
    "state_reverted": STATE_LEVEL,
}

# The strict/advisory split decides the denominator of the headline rate, so the rule
# has to be stated rather than felt: an operator is advisory only when an assertion
# that accepts its output is behaving *correctly*.
#
# Repeating an idempotent read qualifies — the episode is wasteful, not wrong, and an
# assertion that rejected it would be over-fitted to a call count nobody declared.
#
# `inject_extra_call` does NOT qualify, and classing it as advisory was a mistake in
# the first run. Reading a record the user never asked about is a real defect: it is
# the tool-use failure a function-calling benchmark most obviously exists to catch, and
# on a banking pack it is also a data-access one. Reclassifying it moves the human
# suite's call-level false acceptance from 0.137 to 0.374, which is the point — the
# earlier number was an artefact of a lenient denominator, not a property of the
# assertions.
ADVISORY_OPERATORS = frozenset({"duplicate_call_readonly"})
