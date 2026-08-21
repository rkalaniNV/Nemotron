"""Whether one assistant turn earns the tool results the benchmark recorded.

A replayed conversation has one recorded result per gold call, so continuing past
an assistant turn requires deciding *which gold call each predicted call is*. That
decision is the driver's, and it is transport rather than scoring: it selects the
observation the candidate sees next. The scoring component re-derives its verdict
from the recorded episode rather than trusting this gate's continuation decision.

The gate is deliberately an injected :class:`ContinuationGate`. The comparison it
performs must agree with the one the scorer performs — a gate stricter than the
scorer would end an episode the scorer would have credited, turning a correct
model into a failed task — so the pinned publication semantics live here once and
are read from :class:`EvalScoringConfig` rather than hard-coded.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, StrictBool, StrictStr

from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.candidate_contract import (
    CandidateResponse,
    CandidateToolCall,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.conversation_contract import (
    ConversationScript,
    ScriptedCall,
    ScriptedTurn,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.schemas import EvalScoringConfig
from nemotron.steps.byob.runtime.benchmark_families.bfcl.export_contract import (
    NonNegativeInt,
    json_equal,
    thaw_json,
)


class TurnMatch(BaseModel):
    """Whether the turn advances, and if so which gold call answers which call."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    advanced: StrictBool
    detail: StrictStr
    paired_call_indexes: tuple[NonNegativeInt, ...] = ()


@runtime_checkable
class ContinuationGate(Protocol):
    """Decides whether a candidate turn may see the next recorded observation."""

    def evaluate(
        self,
        turn: ScriptedTurn,
        response: CandidateResponse,
        *,
        script: ConversationScript,
    ) -> TurnMatch: ...


def _parameter_schema(script: ConversationScript, function_name: str) -> Mapping[str, Any]:
    for tool in script.tools:
        function = tool.get("function")
        if isinstance(function, Mapping) and function.get("name") == function_name:
            parameters = function.get("parameters")
            return parameters if isinstance(parameters, Mapping) else {}
    return {}


def apply_declared_defaults(arguments: Mapping[str, Any], schema: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively fill parameters the schema gives a default and this side omits.

    Filling the omitting side — rather than both sides, which would be a no-op —
    is what makes spelling out a default neither an advantage nor a penalty.
    Local ``$ref`` and ``allOf`` are followed because oracle packs commonly share
    nested object definitions rather than spelling every parameter inline.
    """
    filled = _defaults_in_value(thaw_json(arguments), schema, root=schema)
    return dict(filled) if isinstance(filled, Mapping) else dict(arguments)


def _resolved_schema(schema: Mapping[str, Any], root: Mapping[str, Any]) -> Mapping[str, Any]:
    reference = schema.get("$ref")
    if not isinstance(reference, str) or not reference.startswith("#/"):
        return schema
    current: Any = root
    for token in reference[2:].split("/"):
        key = token.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, Mapping) or key not in current:
            return schema
        current = current[key]
    return current if isinstance(current, Mapping) else schema


def _defaults_in_value(value: Any, schema: Mapping[str, Any], *, root: Mapping[str, Any]) -> Any:
    schema = _resolved_schema(schema, root)
    for branch in schema.get("allOf", ()):
        if isinstance(branch, Mapping):
            value = _defaults_in_value(value, branch, root=root)

    properties = schema.get("properties")
    if isinstance(value, Mapping) and isinstance(properties, Mapping):
        filled = {str(name): thaw_json(child) for name, child in value.items()}
        for name, child_schema in properties.items():
            if not isinstance(child_schema, Mapping):
                continue
            key = str(name)
            resolved = _resolved_schema(child_schema, root)
            if key not in filled and "default" in resolved:
                filled[key] = thaw_json(resolved["default"])
            if key in filled:
                filled[key] = _defaults_in_value(filled[key], child_schema, root=root)
        return filled

    items = schema.get("items")
    if isinstance(value, list) and isinstance(items, Mapping):
        return [_defaults_in_value(item, items, root=root) for item in value]
    return thaw_json(value)


class CanonicalCallMatchGate:
    """The pinned publication comparison, used only to decide continuation.

    Ordering within one assistant turn follows ``call_order``: a turn's calls are
    issued at once, so ``any`` accepts a permutation of the gold group, ``strict``
    requires every gold position, and ``prefix`` requires only the configured
    global prefix before matching the remainder as a set. Ordering *across* turns
    is not negotiable here — the driver walks the scripted turns in order, so a
    candidate that defers a group to a later turn ends the episode. That is a real
    restriction of trace replay, not a scoring rule: the recorded result for a
    group is only meaningful at the point the trace reached it.
    """

    def __init__(self, scoring: EvalScoringConfig) -> None:
        self._scoring = scoring

    def evaluate(
        self,
        turn: ScriptedTurn,
        response: CandidateResponse,
        *,
        script: ConversationScript,
    ) -> TurnMatch:
        predicted = response.tool_calls
        if not turn.expects_tool_calls:
            if predicted:
                names = ", ".join(sorted({call.function_name or "<unnamed>" for call in predicted}))
                return TurnMatch(
                    advanced=False,
                    detail=f"the trace answers this request in words; the candidate called {names}",
                )
            if response.assistant_content is None:
                return TurnMatch(
                    advanced=False,
                    detail="the candidate returned neither content nor a tool call",
                )
            if not turn.is_terminal and response.assistant_content != turn.expected_assistant_content:
                return TurnMatch(
                    advanced=False,
                    detail="the candidate did not produce the scripted intermediate assistant text",
                )
            return TurnMatch(advanced=True, detail="text turn, as the trace recorded")

        if not predicted:
            expected = ", ".join(call.function_name for call in turn.calls)
            return TurnMatch(
                advanced=False,
                detail=f"the candidate issued no tool call; the trace calls {expected}",
            )
        if len(predicted) != len(turn.calls):
            # This holds even with the grouping gate relaxed: replay has exactly one
            # recorded result per gold call, so a differently sized turn has no
            # faithful reply to hand back, whatever a scorer would award it.
            return TurnMatch(
                advanced=False,
                detail=(
                    f"this turn's call group holds {len(turn.calls)} call(s); "
                    f"the candidate issued {len(predicted)}"
                ),
            )
        for call in predicted:
            if call.type != "function":
                return TurnMatch(
                    advanced=False,
                    detail=f"call {call.index} declares type {call.type!r}; only 'function' can be replayed",
                )
            if call.arguments_status != "valid_object":
                return TurnMatch(
                    advanced=False,
                    detail=(
                        f"call {call.index} to {call.function_name or '<unnamed>'} has "
                        f"{call.arguments_status} arguments"
                    ),
                )
        if self._scoring.respect_call_order and script.call_order == "strict":
            return self._match_positionally(turn, predicted, script=script)
        if self._scoring.respect_call_order and script.call_order == "prefix":
            return self._match_required_tool_prefix(turn, predicted, script=script)
        return self._match_as_a_set(turn, predicted, script=script)

    def _match_required_tool_prefix(
        self,
        turn: ScriptedTurn,
        predicted: Sequence[CandidateToolCall],
        *,
        script: ConversationScript,
    ) -> TurnMatch:
        matched = self._match_as_a_set(turn, predicted, script=script)
        if not matched.advanced:
            return matched

        # The generation contract defines a prefix over first appearances in
        # required_tools, not over raw calls. Prior turns are known to have
        # advanced, so their gold names are sufficient to establish which tools
        # have already appeared; this turn contributes its actual provider order.
        seen: list[str] = []
        required = set(script.required_tools)
        for prior in script.turns[: turn.turn_index]:
            for call in prior.calls:
                if call.function_name in required and call.function_name not in seen:
                    seen.append(call.function_name)
        for call in predicted:
            name = call.function_name
            if name in required and name not in seen:
                seen.append(name)
        prefix = script.call_order_prefix or 0
        expected = list(script.required_tools[:prefix])
        if seen[:prefix] != expected[: len(seen[:prefix])]:
            return TurnMatch(
                advanced=False,
                detail=(
                    f"required-tool prefix is {expected}; "
                    f"the candidate's first appearances are {seen[:prefix]}"
                ),
            )
        return TurnMatch(
            advanced=True,
            detail="calls matched as a set and required-tool first appearances respect the prefix",
            paired_call_indexes=matched.paired_call_indexes,
        )

    def _match_positionally(
        self,
        turn: ScriptedTurn,
        predicted: Sequence[CandidateToolCall],
        *,
        script: ConversationScript,
    ) -> TurnMatch:
        for position, (gold, call) in enumerate(zip(turn.calls, predicted, strict=True)):
            problem = self._compare(gold, call, script=script)
            if problem is not None:
                return TurnMatch(advanced=False, detail=f"call {position}: {problem}")
        return TurnMatch(
            advanced=True,
            detail="every call matched the trace in order",
            paired_call_indexes=tuple(gold.call_index for gold in turn.calls),
        )

    def _match_as_a_set(
        self,
        turn: ScriptedTurn,
        predicted: Sequence[CandidateToolCall],
        *,
        script: ConversationScript,
    ) -> TurnMatch:
        unclaimed = list(turn.calls)
        paired: list[int] = []
        for position, call in enumerate(predicted):
            chosen = next(
                (gold for gold in unclaimed if self._compare(gold, call, script=script) is None),
                None,
            )
            if chosen is None:
                problem = self._compare(unclaimed[0], call, script=script) if unclaimed else "no call left"
                return TurnMatch(advanced=False, detail=f"call {position}: {problem}")
            unclaimed.remove(chosen)
            paired.append(chosen.call_index)
        return TurnMatch(
            advanced=True,
            detail="every call matched the trace, in any order",
            paired_call_indexes=tuple(paired),
        )

    def _compare(
        self,
        gold: ScriptedCall,
        call: CandidateToolCall,
        *,
        script: ConversationScript,
    ) -> str | None:
        if call.function_name != gold.function_name:
            return f"calls {call.function_name or '<unnamed>'}; the trace calls {gold.function_name}"
        predicted_arguments = thaw_json(call.parsed_arguments or {})
        gold_arguments = thaw_json(gold.arguments)
        if self._scoring.argument_matching == "schema_then_canonical" and self._scoring.insert_declared_defaults:
            schema = _parameter_schema(script, gold.function_name)
            predicted_arguments = apply_declared_defaults(predicted_arguments, schema)
            gold_arguments = apply_declared_defaults(gold_arguments, schema)
        if json_equal(predicted_arguments, gold_arguments):
            return None
        missing = sorted(set(gold_arguments) - set(predicted_arguments))
        extra = sorted(set(predicted_arguments) - set(gold_arguments))
        differing = sorted(
            name
            for name in set(gold_arguments) & set(predicted_arguments)
            if not json_equal(predicted_arguments[name], gold_arguments[name])
        )
        parts = [
            f"{label} {', '.join(names)}"
            for label, names in (("missing", missing), ("unexpected", extra), ("differing", differing))
            if names
        ]
        return f"arguments to {gold.function_name} do not match the trace: {'; '.join(parts)}"


__all__ = [
    "CanonicalCallMatchGate",
    "ContinuationGate",
    "TurnMatch",
    "apply_declared_defaults",
]
