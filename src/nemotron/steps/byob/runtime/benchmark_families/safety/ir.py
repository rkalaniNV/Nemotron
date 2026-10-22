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

"""Policy Rule IR: the stored shape of one atomic customer rule.

Every stored condition list declares exactly one operator. Extractor output, where a
condition may still be a bare list, goes through :func:`normalize_candidate` before it
can be stored.
"""

import copy
import datetime
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, get_args

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    WithJsonSchema,
    model_validator,
)

Operator = Literal["all_of", "any_of", "none_of"]
# single_condition: a list with one leg has no operator to state; all_of and any_of mean the same.
OperatorSource = Literal["explicit", "model_evidence", "default_any_of", "single_condition"]
ConditionClass = Literal["allow_if", "block_if", "escalate_if", "require_confirmation_if"]
Action = Literal["escalate", "block", "require_confirmation", "rewrite", "redact", "caution", "allow"]
Severity = Literal["critical", "high", "medium", "low"]
Target = Literal[
    "user_input",
    "assistant_output",
    "retrieved_context",
    "tool_input",
    "tool_output",
    "tool_call",
]
RuleStatus = Literal["eligible_for_generation", "needs_decision", "quarantined", "rejected"]
Testability = Literal["testable", "partially_testable", "needs_clarification", "non_testable", "rejected"]

ACTIONS: tuple[str, ...] = get_args(Action)
TARGETS: tuple[str, ...] = get_args(Target)
DEFAULT_TARGETS: tuple[str, ...] = ("user_input", "assistant_output")
OPERATORS: tuple[str, ...] = ("all_of", "any_of", "none_of")
CONDITION_CLASSES: tuple[str, ...] = ("allow_if", "block_if", "escalate_if", "require_confirmation_if")
UNSPECIFIED = "unspecified"
# Keys an extraction model may return that must never be stored or published.
MODEL_ONLY_KEYS = frozenset({"model_confidence", "self_score", "model_self_score", "severity_hint"})


def _lowercase_hex(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip().lower()
    return value


PolicyId = Annotated[str, StringConstraints(strict=True, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")]


def _condition_leg(value: str) -> str:
    """Require one Unicode identifier token, without imposing a source script."""
    if not value[0].isalpha() or any(not (char.isalnum() or char == "_") for char in value):
        raise ValueError("must start with a letter and contain only Unicode letters, numbers, or underscores")
    return value


# Runtime validation is Unicode-aware. JSON Schema regex has no portable Unicode-property
# syntax, so its pattern is deliberately broader; PolicyRule validation remains authoritative.
Leg = Annotated[
    str,
    StringConstraints(strict=True, min_length=1, max_length=128),
    AfterValidator(_condition_leg),
    WithJsonSchema({"type": "string", "minLength": 1, "maxLength": 128, "pattern": r"^(?![0-9_])\S+$"}),
]
Text = Annotated[str, StringConstraints(strict=True, strip_whitespace=True, min_length=1)]
Sha256 = Annotated[str, BeforeValidator(_lowercase_hex), StringConstraints(strict=True, pattern=r"^[0-9a-f]{64}$")]

_MISSING = object()


class RuleValidationError(ValueError):
    """A mapping is not a valid stored Policy Rule IR entry."""

    def __init__(self, messages: list[str]) -> None:
        self.messages = messages
        super().__init__("\n".join(messages))


class _FrozenMap(dict):
    """Dict equality, with later changes refused."""

    def __setitem__(self, key: Any, value: Any) -> None:
        raise TypeError("this mapping cannot be changed")

    def __delitem__(self, key: Any) -> None:
        raise TypeError("this mapping cannot be changed")

    def clear(self) -> None:
        raise TypeError("this mapping cannot be changed")

    def pop(self, *args: Any) -> Any:
        raise TypeError("this mapping cannot be changed")

    def popitem(self) -> tuple[Any, Any]:
        raise TypeError("this mapping cannot be changed")

    def setdefault(self, *args: Any) -> Any:
        raise TypeError("this mapping cannot be changed")

    def update(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("this mapping cannot be changed")


def _freeze(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        frozen = _FrozenMap()
        dict.update(frozen, {key: _freeze(item) for key, item in value.items()})
        return frozen
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ConditionGroup(_Model):
    """One condition list. Exactly one of all_of, any_of, or none_of is set."""

    all_of: tuple[Leg, ...] | None = None
    any_of: tuple[Leg, ...] | None = None
    none_of: tuple[Leg, ...] | None = None
    operator_source: OperatorSource = "explicit"

    @model_validator(mode="after")
    def _one_operator(self) -> "ConditionGroup":
        present = [name for name in OPERATORS if getattr(self, name) is not None]
        if len(present) != 1:
            raise ValueError(
                f"a condition list must set exactly one of all_of, any_of, or none_of; found {present or 'none'}"
            )
        legs = self.legs
        if not legs:
            raise ValueError(f"{present[0]} must list at least one condition")
        if len(set(legs)) != len(legs):
            raise ValueError(f"{present[0]} repeats a condition")
        if self.operator_source == "default_any_of" and self.operator != "any_of":
            raise ValueError("operator_source default_any_of is only valid with any_of")
        if self.operator_source == "single_condition" and (self.operator == "none_of" or len(legs) != 1):
            raise ValueError("operator_source single_condition needs all_of or any_of with exactly one condition")
        return self

    @property
    def operator(self) -> str:
        return next(name for name in OPERATORS if getattr(self, name) is not None)

    @property
    def legs(self) -> tuple[str, ...]:
        return getattr(self, self.operator)


class RuleBody(_Model):
    actor: Text
    action: Text
    object: Text
    allow_if: ConditionGroup | None = None
    block_if: ConditionGroup | None = None
    escalate_if: ConditionGroup | None = None
    require_confirmation_if: ConditionGroup | None = None

    def conditions(self) -> dict[str, ConditionGroup]:
        return {name: group for name in CONDITION_CLASSES if (group := getattr(self, name)) is not None}


class Scope(_Model):
    domains: tuple[Text, ...] = (UNSPECIFIED,)
    jurisdictions: tuple[Text, ...] = (UNSPECIFIED,)
    languages: tuple[Text, ...] = ()
    audiences: tuple[Text, ...] = ()
    targets: tuple[Target, ...] = DEFAULT_TARGETS

    @model_validator(mode="before")
    @classmethod
    def _empty_means_unspecified(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        for key in ("domains", "jurisdictions"):
            if data.get(key) in ([], ()):
                data[key] = [UNSPECIFIED]
        return data

    @model_validator(mode="after")
    def _no_global(self) -> "Scope":
        if any(value.casefold() == "global" for value in self.jurisdictions):
            raise ValueError("jurisdiction 'global' is not allowed; use 'unspecified' when the source names none")
        return self


class Decision(_Model):
    violation_action: Action = "block"
    ambiguous_action: Action | None = None
    critical_action: Action | None = None


class RuleSignals(_Model):
    """Optional facts recorded only after their evidence was found in the source quote."""

    binding_evidence: Text | None = None
    conditional_evidence: Text | None = None
    multi_turn_turns: Annotated[int, Field(strict=True, ge=2, le=99)] | None = None
    code_switch: bool = False


class Provenance(_Model):
    document_id: Text
    document_sha256: Sha256 | None = None
    section: Text | None = None
    source_span_id: Text | None = None
    source_quote: Text


class PolicyRule(_Model):
    """One stored atomic rule."""

    policy_id: PolicyId
    title: Text | None = None
    source_type: Text | None = None
    version: Text = UNSPECIFIED
    owner: Text = UNSPECIFIED
    effective_date: Text = UNSPECIFIED
    scope: Scope = Field(default_factory=Scope)
    rule: RuleBody
    unresolved_conditions: dict[ConditionClass, tuple[Leg, ...]] = Field(default_factory=dict)
    exceptions: tuple[Leg, ...] = ()
    decision: Decision = Field(default_factory=Decision)
    severity: Severity
    severity_basis: Text | None = None
    paired_with: PolicyId | None = None
    provenance: Provenance
    testability: Testability | None = None
    unsupported_fields: tuple[Text, ...] = ()
    confidence: Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)] | None = None
    status: RuleStatus = "needs_decision"
    signals: RuleSignals = Field(default_factory=RuleSignals)
    extensions: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _consistent(self) -> "PolicyRule":
        conditions = self.rule.conditions()
        if not conditions and not self.unresolved_conditions:
            raise ValueError("a rule needs at least one condition list")
        both = sorted(set(conditions) & set(self.unresolved_conditions))
        if both:
            raise ValueError(f"{', '.join(both)} is both stored and unresolved")
        for name, group in conditions.items():
            if group.operator_source == "default_any_of" and name != "block_if":
                raise ValueError(f"{name} cannot use operator_source default_any_of; only block_if has a default")
        for name, legs in self.unresolved_conditions.items():
            if not legs or len(set(legs)) != len(legs):
                raise ValueError(f"unresolved_conditions.{name} must list distinct conditions")
        if self.status == "eligible_for_generation":
            blockers = []
            if self.unresolved_conditions:
                blockers.append("unresolved condition operators")
            if self.unsupported_fields:
                blockers.append("unsupported inferred fields")
            if self.confidence is None:
                blockers.append("no confidence")
            if self.testability != "testable":
                blockers.append(f"testability {self.testability}")
            if blockers:
                raise ValueError("an eligible_for_generation rule cannot have " + ", ".join(blockers))
        object.__setattr__(self, "unresolved_conditions", _freeze(dict(self.unresolved_conditions)))
        object.__setattr__(self, "extensions", _freeze(dict(self.extensions)))
        return self


def validate_rule(data: Any) -> PolicyRule:
    """Validate one stored rule. A bare condition list is a validation error here."""
    try:
        return PolicyRule.model_validate(data)
    except ValidationError as exc:
        raise RuleValidationError(_messages(exc)) from exc


def policy_rule_json_schema() -> dict[str, Any]:
    """JSON Schema for customers who author Policy Rule IR by hand.

    The one-operator rule lives in a validator, so it is added to the schema here.
    """
    schema = PolicyRule.model_json_schema()
    group = schema["$defs"]["ConditionGroup"]
    group["oneOf"] = [
        {
            "required": [name],
            "properties": {name: {"type": "array", "minItems": 1, "uniqueItems": True}},
            "not": {"anyOf": [{"required": [other]} for other in OPERATORS if other != name]},
        }
        for name in OPERATORS
    ]
    return schema


@dataclass(frozen=True)
class Flag:
    """One normalization event, recorded for the routing log and the review packet."""

    code: str
    detail: str
    field: str | None = None


@dataclass(frozen=True)
class NormalizationResult:
    rule: PolicyRule | None
    flags: list[Flag] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def normalize_candidate(
    raw: dict[str, Any],
    *,
    operator_text: str | Mapping[str, str] | None = None,
) -> NormalizationResult:
    """Turn extractor output into a storable rule.

    - A one-condition ``block_if`` list becomes ``any_of`` with ``operator_source: default_any_of``.
      A longer bare list stays unresolved. Source wording is not scanned for operator frames.
    - ``operator_text`` may be a mapping of condition names; unknown keys are refused. A
      string is ignored. Neither form chooses ``all_of``, ``any_of``, or ``none_of``.
    - A model self-score is dropped. ``jurisdiction: global`` is cleared to unspecified.
    - A reviewer ``rejected`` status is kept. Every other status waits for routing.

    Malformed extractor fields return ``errors`` instead of raising.
    """
    if not isinstance(raw, dict):
        return NormalizationResult(None, errors=["a candidate rule must be a mapping"])
    try:
        return _normalize_candidate(raw, operator_text)
    except (TypeError, ValueError) as exc:
        return NormalizationResult(None, errors=[f"candidate could not be normalized: {exc}"])


def _normalize_candidate(
    raw: dict[str, Any],
    operator_text: str | Mapping[str, str] | None,
) -> NormalizationResult:
    data = copy.deepcopy(raw)
    flags: list[Flag] = []
    _drop_model_keys(data, flags)
    if "confidence" in data:
        data.pop("confidence")
        flags.append(Flag("confidence_dropped", "confidence is computed by routing, not supplied", "confidence"))
    _adopt_policy_id(data, flags)
    _coerce_effective_date(data)

    unresolved = _existing_unresolved(data)
    if isinstance(unresolved, str):
        return NormalizationResult(None, flags, [unresolved])
    if operator_text is not None and not isinstance(operator_text, (str, Mapping)):
        raise TypeError("operator_text must be a string or a mapping of condition names")
    body = data.get("rule")
    if isinstance(body, dict):
        _normalize_bare_lists(body, operator_text, unresolved, flags)
    if unresolved:
        data["unresolved_conditions"] = unresolved

    scope = data.get("scope")
    if isinstance(scope, dict):
        _clear_global_jurisdiction(data, scope, flags)

    if data.get("status") == "rejected":
        flags.append(Flag("status_kept", "a reviewer rejection is kept", "status"))
    else:
        if data.get("status") not in (None, "needs_decision"):
            flags.append(Flag("status_reset", "only routing may mark a rule eligible_for_generation", "status"))
        data["status"] = "needs_decision"
    try:
        return NormalizationResult(PolicyRule.model_validate(data), flags)
    except ValidationError as exc:
        return NormalizationResult(None, flags, _messages(exc))


def _drop_model_keys(data: dict[str, Any], flags: list[Flag]) -> None:
    containers: list[tuple[str, Any]] = [("", data)]
    for name in ("rule", "provenance", "decision", "scope"):
        child = data.get(name)
        if isinstance(child, dict):
            containers.append((name, child))
    for prefix, container in containers:
        for key in sorted(MODEL_ONLY_KEYS & set(container)):
            container.pop(key)
            field_name = f"{prefix}.{key}" if prefix else key
            flags.append(Flag("model_value_dropped", f"{key} is a model signal and is not stored", field_name))


def _adopt_policy_id(data: dict[str, Any], flags: list[Flag]) -> None:
    if "policy_id_candidate" not in data:
        return
    candidate = data.pop("policy_id_candidate")
    if "policy_id" not in data:
        data["policy_id"] = candidate
        return
    if data["policy_id"] != candidate:
        flags.append(
            Flag("policy_id_conflict", "policy_id and policy_id_candidate differ; policy_id is kept", "policy_id")
        )
        _mark_unsupported(data, "policy_id", flags)


def _coerce_effective_date(data: dict[str, Any]) -> None:
    effective = data.get("effective_date")
    if isinstance(effective, datetime.datetime):
        data["effective_date"] = effective.date().isoformat()
    elif isinstance(effective, datetime.date):
        data["effective_date"] = effective.isoformat()


def _existing_unresolved(data: dict[str, Any]) -> dict[str, Any] | str:
    raw = data.get("unresolved_conditions")
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    return "unresolved_conditions must be a mapping"


def _normalize_bare_lists(
    body: dict[str, Any],
    operator_text: str | Mapping[str, str] | None,
    unresolved: dict[str, Any],
    flags: list[Flag],
) -> None:
    """Canonicalize one-condition lists. Multi-leg bare lists stay unresolved.

    ``operator_text`` is validated when it is a mapping. It is never scanned for
    language-specific operator frames.
    """
    if isinstance(operator_text, Mapping):
        unknown = sorted(set(operator_text) - set(CONDITION_CLASSES))
        if unknown:
            raise ValueError(f"operator_text has unknown condition names: {', '.join(unknown)}")
        invalid = sorted(name for name, text in operator_text.items() if not isinstance(text, str))
        if invalid:
            raise ValueError(f"operator_text values must be strings: {', '.join(invalid)}")
    for name in [item for item in CONDITION_CLASSES if isinstance(body.get(item), list)]:
        value = body[name]
        if name != "block_if" and len(value) == 1:
            body[name] = {"any_of": value, "operator_source": "single_condition"}
            flags.append(
                Flag("single_condition", f"{name} has one condition, so it needs no operator", f"rule.{name}")
            )
            continue
        if name == "block_if" and len(value) == 1:
            body[name] = {"any_of": value, "operator_source": "default_any_of"}
            flags.append(
                Flag(
                    "default_any_of",
                    "one block_if condition needs no operator; all_of and any_of mean the same",
                    "rule.block_if",
                )
            )
            continue
        unresolved[name] = body.pop(name)
        flags.append(
            Flag("operator_unresolved", f"{name} has no operator of its own and has no default", f"rule.{name}")
        )


def _clear_global_jurisdiction(data: dict[str, Any], scope: dict[str, Any], flags: list[Flag]) -> None:
    singular = scope.pop("jurisdiction", _MISSING)
    plural = scope.get("jurisdictions", _MISSING)
    if singular is _MISSING and plural is _MISSING:
        return
    groups: list[list[str]] = []
    saw_global = False
    for raw in (singular, plural):
        if raw is _MISSING:
            continue
        parsed = _string_items(raw)
        if parsed is None:
            if singular is not _MISSING:
                scope["jurisdiction"] = singular
            return
        cleaned: list[str] = []
        for value in parsed:
            stripped = value.strip()
            if not stripped:
                continue
            if stripped.casefold() == "global":
                saw_global = True
            else:
                cleaned.append(stripped)
        groups.append(list(dict.fromkeys(cleaned)))
    scope["jurisdictions"] = list(dict.fromkeys(item for group in groups for item in group)) or [UNSPECIFIED]
    if len(groups) == 2 and set(groups[0]) != set(groups[1]):
        _mark_unsupported(data, "scope.jurisdictions", flags)
        flags.append(Flag("jurisdiction_conflict", "jurisdiction and jurisdictions disagree", "scope.jurisdictions"))
    if saw_global:
        _mark_unsupported(data, "scope.jurisdictions", flags)
        flags.append(Flag("global_cleared", "jurisdiction global replaced by unspecified", "scope.jurisdictions"))


def _string_items(value: Any) -> list[str] | None:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return list(value)
    return None


def _mark_unsupported(data: dict[str, Any], field_name: str, flags: list[Flag]) -> None:
    current = data.get("unsupported_fields")
    if not isinstance(current, list):
        if current not in (None, ""):
            flags.append(
                Flag(
                    "unsupported_fields_ignored",
                    "unsupported_fields was not a list, so it was not kept",
                    "unsupported_fields",
                )
            )
        current = []
    if field_name not in current:
        current.append(field_name)
    data["unsupported_fields"] = current


def _messages(exc: ValidationError) -> list[str]:
    messages = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "rule"
        messages.append(f"{location}: {error['msg']}")
    return messages
