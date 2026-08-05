"""Validate expected traces against the pack's tool schemas."""

from __future__ import annotations

import logging
import re
from typing import Any

from nemotron.steps.byob.runtime.benchmark_families.bfcl.config import BfclConfig
from nemotron.steps.byob.runtime.benchmark_families.bfcl.pack_loader import (
    LoadedPack,
    confirmation_protocol,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.stage_tables import (
    SCHEMA_VALIDATED_TRACES,
    schema_validated_row,
    schema_validated_traces_schema,
    write_stage_table,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages import stage_cache_dir
from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.expected_trace import (
    corrected_slot_values,
)

logger = logging.getLogger(__name__)

REJECT_REASON = "expected_trace_schema_mismatch"

_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
}
_SCHEMA_TYPES = {*_JSON_TYPES, "null"}
_UNSUPPORTED_SCHEMA_KEYWORDS = {"$ref", "allOf", "anyOf", "not", "oneOf"}
_SUPPORTED_SCHEMA_KEYWORDS = {
    "type",
    "description",
    "title",
    "default",
    "examples",
    "deprecated",
    "readOnly",
    "writeOnly",
    # JSON Schema treats `format` as an annotation rather than an assertion, so a tool
    # may declare it without this stage having to enforce it.
    "format",
    "enum",
    "const",
    "properties",
    "required",
    "additionalProperties",
    "items",
    "pattern",
    "minLength",
    "maxLength",
    "minimum",
    "maximum",
    "minItems",
    "maxItems",
}


def _same_value(left: Any, right: Any) -> bool:
    """Compare two JSON values without letting ``True`` equal ``1``."""
    if isinstance(left, bool) != isinstance(right, bool):
        return False
    return left == right


def _duplicates(values: list[Any]) -> list[Any]:
    """Return the values that appear more than once, in declaration order."""
    repeated: list[Any] = []
    for index, value in enumerate(values):
        if any(_same_value(value, earlier) for earlier in values[:index]) and not any(
            _same_value(value, seen) for seen in repeated
        ):
            repeated.append(value)
    return repeated


def _tool_index(pack: LoadedPack) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for tool in pack.tools:
        function = tool.get("function") or {}
        name = function.get("name")
        if isinstance(name, str):
            index[name] = function
    return index


def validate_function_schema(function: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate the parameter-schema subset implemented by this pipeline."""
    failures: list[dict[str, Any]] = []
    parameters = function.get("parameters")
    if not isinstance(parameters, dict):
        return [{"reason": "parameters_not_object"}]
    if parameters.get("type", "object") != "object":
        failures.append({"reason": "parameters_type_not_object"})

    def visit(schema: dict[str, Any], path: str) -> None:
        for keyword in sorted(_UNSUPPORTED_SCHEMA_KEYWORDS & schema.keys()):
            failures.append(
                {"reason": "unsupported_schema_keyword", "path": path, "keyword": keyword}
            )
        for keyword in sorted(set(schema) - _SUPPORTED_SCHEMA_KEYWORDS - _UNSUPPORTED_SCHEMA_KEYWORDS):
            # Silently accepting a constraint we do not enforce can certify an invalid
            # expected call. Annotation-only keys are explicitly allowlisted above.
            failures.append(
                {"reason": "unsupported_schema_keyword", "path": path, "keyword": keyword}
            )
        declared = schema.get("type")
        types = [declared] if isinstance(declared, str) else declared
        type_is_valid = declared is None or (
            isinstance(types, list)
            and bool(types)
            and all(isinstance(item, str) and item in _SCHEMA_TYPES for item in types)
        )
        if not type_is_valid:
            failures.append({"reason": "invalid_schema_type", "path": path, "value": declared})
        elif isinstance(types, list) and _duplicates(types):
            # A repeated member of a union says the author meant to allow something else.
            failures.append(
                {"reason": "duplicate_schema_type", "path": path, "values": _duplicates(types)}
            )

        def check_allowed(keyword: str, values: list[Any]) -> None:
            """Refuse fixed values that the declared type can never accept."""
            if declared is None or not type_is_valid or not isinstance(types, list):
                return
            for value in values:
                if not any(_type_matches(value, item) for item in types):
                    failures.append(
                        {
                            "reason": "schema_value_violates_type",
                            "path": path,
                            "keyword": keyword,
                            "value": value,
                        }
                    )

        if "enum" in schema:
            enum = schema["enum"]
            if not isinstance(enum, list):
                failures.append({"reason": "schema_enum_not_array", "path": path})
            elif not enum:
                # Nothing can satisfy an empty enum, so every bound argument would fail.
                failures.append({"reason": "schema_enum_empty", "path": path})
            else:
                repeated = _duplicates(enum)
                if repeated:
                    failures.append(
                        {"reason": "duplicate_schema_enum_value", "path": path, "values": repeated}
                    )
                check_allowed("enum", enum)
        if "const" in schema:
            check_allowed("const", [schema["const"]])
        if "additionalProperties" in schema and not isinstance(
            schema["additionalProperties"], bool
        ):
            failures.append(
                {
                    "reason": "schema_additional_properties_not_boolean",
                    "path": path,
                }
            )
        for keyword in ("minLength", "maxLength", "minItems", "maxItems"):
            value = schema.get(keyword)
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value < 0
            ):
                failures.append(
                    {"reason": "invalid_schema_bound", "path": path, "keyword": keyword}
                )
        for keyword in ("minimum", "maximum"):
            value = schema.get(keyword)
            if value is not None and (
                not isinstance(value, (int, float)) or isinstance(value, bool)
            ):
                failures.append(
                    {"reason": "invalid_schema_bound", "path": path, "keyword": keyword}
                )
        if "pattern" in schema:
            if not isinstance(schema["pattern"], str):
                failures.append({"reason": "schema_pattern_not_string", "path": path})
            else:
                try:
                    re.compile(schema["pattern"])
                except re.error:
                    failures.append({"reason": "invalid_schema_pattern", "path": path})

        comparable_bounds = (
            ("minLength", "maxLength"),
            ("minItems", "maxItems"),
            ("minimum", "maximum"),
        )
        for lower_name, upper_name in comparable_bounds:
            lower = schema.get(lower_name)
            upper = schema.get(upper_name)
            if (
                isinstance(lower, (int, float))
                and not isinstance(lower, bool)
                and isinstance(upper, (int, float))
                and not isinstance(upper, bool)
                and lower > upper
            ):
                failures.append(
                    {
                        "reason": "inconsistent_schema_bounds",
                        "path": path,
                        "lower": lower_name,
                        "upper": upper_name,
                    }
                )

        properties = schema.get("properties")
        if properties is not None and not isinstance(properties, dict):
            failures.append({"reason": "schema_properties_not_object", "path": path})
            properties = {}
        properties = properties or {}
        required = schema.get("required") or []
        if not isinstance(required, list) or any(not isinstance(item, str) for item in required):
            failures.append({"reason": "schema_required_not_string_array", "path": path})
        else:
            repeated_required = _duplicates(required)
            if repeated_required:
                failures.append(
                    {
                        "reason": "duplicate_required_property",
                        "path": path,
                        "values": repeated_required,
                    }
                )
            for name in required:
                if name not in properties:
                    failures.append(
                        {"reason": "required_property_not_declared", "path": path, "property": name}
                    )
        for name, child in properties.items():
            if not isinstance(child, dict):
                failures.append(
                    {"reason": "property_schema_not_object", "path": f"{path}.{name}"}
                )
            else:
                visit(child, f"{path}.{name}")
        if "items" in schema:
            if not isinstance(schema["items"], dict):
                failures.append({"reason": "items_schema_not_object", "path": path})
            else:
                visit(schema["items"], f"{path}[]")

    visit(parameters, "$")
    return failures


def validate_tool_definition(tool: Any) -> list[dict[str, Any]]:
    """Validate the model-facing OpenAI function-tool envelope and its schema."""
    if not isinstance(tool, dict):
        return [{"reason": "tool_not_object"}]
    failures: list[dict[str, Any]] = []
    if tool.get("type", "function") != "function":
        failures.append({"reason": "tool_type_not_function", "value": tool.get("type")})
    function = tool.get("function")
    if not isinstance(function, dict):
        return [*failures, {"reason": "tool_function_not_object"}]
    if not isinstance(function.get("name"), str) or not function["name"].strip():
        failures.append({"reason": "tool_name_not_nonempty_string"})
    if "description" in function and not isinstance(function["description"], str):
        failures.append({"reason": "tool_description_not_string"})
    if "strict" in function and not isinstance(function["strict"], bool):
        failures.append({"reason": "tool_strict_not_boolean"})
    failures.extend(validate_function_schema(function))
    return failures


def _type_matches(value: Any, declared: str) -> bool:
    if declared == "null":
        return value is None
    allowed = _JSON_TYPES.get(declared)
    if allowed is None:
        return False
    if declared in {"integer", "number"} and isinstance(value, bool):
        return False
    return isinstance(value, allowed)


def _check_value(schema: dict[str, Any], value: Any, path: str) -> list[dict[str, Any]]:
    """Validate the JSON-Schema subset used by OpenAI function parameters."""
    failures: list[dict[str, Any]] = []
    declared = schema.get("type")
    declared_types = [declared] if isinstance(declared, str) else declared
    if isinstance(declared_types, list):
        known = [item for item in declared_types if isinstance(item, str)]
        if not known or not any(_type_matches(value, item) for item in known):
            failures.append(
                {
                    "reason": "argument_type_mismatch",
                    "argument": path,
                    "expected_type": declared,
                }
            )
            return failures
    elif declared is not None:
        failures.append({"reason": "invalid_schema_type", "argument": path, "value": declared})
        return failures

    if "enum" in schema and not any(
        _same_value(value, candidate) for candidate in schema["enum"]
    ):
        failures.append({"reason": "argument_not_in_enum", "argument": path})
    if "const" in schema and not _same_value(value, schema["const"]):
        failures.append({"reason": "argument_not_equal_const", "argument": path})

    if isinstance(value, str):
        if isinstance(schema.get("minLength"), int) and len(value) < schema["minLength"]:
            failures.append({"reason": "string_too_short", "argument": path})
        if isinstance(schema.get("maxLength"), int) and len(value) > schema["maxLength"]:
            failures.append({"reason": "string_too_long", "argument": path})
        if isinstance(schema.get("pattern"), str):
            try:
                matches = re.search(schema["pattern"], value) is not None
            except re.error:
                failures.append({"reason": "invalid_schema_pattern", "argument": path})
            else:
                if not matches:
                    failures.append({"reason": "string_pattern_mismatch", "argument": path})

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            failures.append({"reason": "number_below_minimum", "argument": path})
        if "maximum" in schema and value > schema["maximum"]:
            failures.append({"reason": "number_above_maximum", "argument": path})

    if isinstance(value, list):
        if isinstance(schema.get("minItems"), int) and len(value) < schema["minItems"]:
            failures.append({"reason": "array_too_short", "argument": path})
        if isinstance(schema.get("maxItems"), int) and len(value) > schema["maxItems"]:
            failures.append({"reason": "array_too_long", "argument": path})
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                failures.extend(_check_value(item_schema, item, f"{path}[{index}]"))

    if isinstance(value, dict):
        properties = schema.get("properties") or {}
        for name in schema.get("required") or []:
            if name not in value:
                failures.append(
                    {"reason": "missing_required_argument", "argument": f"{path}.{name}"}
                )
        if schema.get("additionalProperties") is False:
            for name in value:
                if name not in properties:
                    failures.append({"reason": "unknown_argument", "argument": f"{path}.{name}"})
        for name, child in value.items():
            child_schema = properties.get(name)
            if isinstance(child_schema, dict):
                failures.extend(_check_value(child_schema, child, f"{path}.{name}"))
    return failures


def _check_arguments(function: dict[str, Any], arguments: dict[str, Any]) -> list[dict[str, Any]]:
    parameters = function.get("parameters") or {}
    properties = parameters.get("properties") or {}
    required = parameters.get("required") or []
    failures: list[dict[str, Any]] = []

    for name in required:
        if name not in arguments:
            failures.append({"reason": "missing_required_argument", "argument": name})
    if parameters.get("additionalProperties") is False:
        for name in arguments:
            if name not in properties:
                failures.append({"reason": "unknown_argument", "argument": name})
    for name, value in arguments.items():
        schema = properties.get(name)
        if not isinstance(schema, dict):
            continue
        failures.extend(_check_value(schema, value, name))
    return failures


def validate_function_arguments(
    function: dict[str, Any], arguments: Any
) -> list[dict[str, Any]]:
    """Validate one declared call payload against a function's parameter schema."""
    if not isinstance(arguments, dict):
        return [{"reason": "arguments_not_object"}]
    return _check_arguments(function, arguments)


def validate_task(
    pack: LoadedPack,
    task: dict[str, Any],
    expected_calls: list[dict[str, Any]],
    tools: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Return schema failures for one task's expected trace."""
    tool_index = tools if tools is not None else _tool_index(pack)
    confirmation_tools = {
        str((tool.get("function") or {}).get("name"))
        for tool in pack.tools
        if tool.get("x-requires-confirmation")
    }
    confirm_parameter = confirmation_protocol(pack.manifest)["parameter"]
    corrected = corrected_slot_values(task)
    confirmed_turns = task.get("confirmed_call_turns")
    failures: list[dict[str, Any]] = []
    group_turns: dict[int, set[int]] = {}
    group_positions: dict[tuple[int, int], list[int]] = {}

    for call in expected_calls:
        for field in ("turn_index", "call_group", "position_in_group"):
            value = call.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                failures.append({"reason": "bad_struct_field", "field": field, "value": value})
        if not isinstance(call.get("arguments"), dict):
            failures.append({"reason": "arguments_not_object"})
            continue
        turn = call.get("turn_index")
        group = call.get("call_group")
        position = call.get("position_in_group")
        if all(isinstance(value, int) and not isinstance(value, bool) for value in (turn, group, position)):
            group_turns.setdefault(group, set()).add(turn)
            group_positions.setdefault((turn, group), []).append(position)
        name = call.get("function_name")
        function = tool_index.get(str(name))
        if function is None:
            failures.append({"reason": "unknown_tool", "tool": name})
            continue
        if str(name) not in (task.get("tools_present") or []):
            failures.append({"reason": "tool_not_exposed", "tool": name})
        # A confirmed mutation is only gold when this very turn is covered by a
        # confirmation the user already gave and no later turn withdrew, whatever the
        # template's policy label says.
        if (
            str(name) in confirmation_tools
            and call["arguments"].get(confirm_parameter) is True
            and call.get("turn_index") not in (confirmed_turns or [])
        ):
            failures.append({"reason": "confirmed_mutation_without_user_confirmation", "tool": name})
        for argument, value in call["arguments"].items():
            if argument in corrected and value != corrected[argument]:
                failures.append(
                    {
                        "reason": "superseded_slot_value_in_trace",
                        "tool": name,
                        "argument": argument,
                        "slot": argument,
                    }
                )
        failures.extend({**failure, "tool": name} for failure in _check_arguments(function, call["arguments"]))

    call_order = task.get("call_order", "strict")
    prefix = task.get("call_order_prefix")
    if task.get("turn_policy") == "dependent_call" and call_order != "strict":
        failures.append(
            {
                "reason": "dependent_call_requires_strict_order",
                "call_order": call_order,
            }
        )
    if call_order not in {"strict", "any", "prefix"}:
        failures.append({"reason": "unknown_call_order", "call_order": call_order})
    # required_tools declares the scoring order. For strict/prefix, the first
    # appearance of each required tool in the trace must follow that sequence.
    required = [str(name) for name in (task.get("required_tools") or [])]
    if call_order == "prefix":
        # The prefix counts required tools: a larger value would silently mean strict.
        if not isinstance(prefix, int) or not 1 <= prefix <= len(required):
            failures.append({"reason": "bad_call_order_prefix", "call_order_prefix": prefix})
    elif prefix is not None:
        failures.append({"reason": "call_order_prefix_without_prefix_order"})

    required_set = set(required)
    seen_required: list[str] = []
    for call in expected_calls:
        name = str(call.get("function_name"))
        if name in required_set and name not in seen_required:
            seen_required.append(name)
    if call_order == "strict" and required and seen_required != required:
        failures.append(
            {
                "reason": "call_order_mismatch",
                "call_order": "strict",
                "expected": required,
                "got": seen_required,
            }
        )
    elif call_order == "prefix" and isinstance(prefix, int) and required:
        expected_prefix = required[:prefix]
        got_prefix = seen_required[:prefix]
        if got_prefix != expected_prefix:
            failures.append(
                {
                    "reason": "call_order_mismatch",
                    "call_order": "prefix",
                    "expected": expected_prefix,
                    "got": got_prefix,
                }
            )
        if sorted(seen_required[prefix:]) != sorted(required[prefix:]):
            failures.append(
                {
                    "reason": "call_order_remainder_mismatch",
                    "expected": sorted(required[prefix:]),
                    "got": sorted(seen_required[prefix:]),
                }
            )

    for group, turns in group_turns.items():
        if len(turns) != 1:
            failures.append({"reason": "call_group_spans_turns", "call_group": group})
    for (turn, group), positions in group_positions.items():
        if sorted(positions) != list(range(len(positions))):
            failures.append(
                {
                    "reason": "non_contiguous_group_positions",
                    "turn_index": turn,
                    "call_group": group,
                    "positions": positions,
                }
            )

    if task.get("turn_policy") == "irrelevant" and expected_calls:
        failures.append({"reason": "irrelevant_task_has_calls"})
    for tool_name in required:
        if tool_name not in {call.get("function_name") for call in expected_calls}:
            failures.append({"reason": "required_tool_not_called", "tool": tool_name})
    return failures


def run_schema_validation(
    config: BfclConfig,
    pack: LoadedPack,
    tasks: list[dict[str, Any]],
    traces: dict[str, list[dict[str, Any]]],
    *,
    skipped: dict[str, str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Validate every trace and cache the per-task failures.

    ``skipped`` carries task ids dropped before this stage (for example a
    dependent_call bind failure). Those rows stay in the stage table so joins
    across artifacts still show every expanded ``task_id``.
    """
    tool_index = _tool_index(pack)
    skipped = skipped or {}
    failures: dict[str, list[dict[str, Any]]] = {}
    rows: list[dict[str, Any]] = []
    for task in tasks:
        task_id = str(task["task_id"])
        if task_id in skipped:
            failures[task_id] = [{"reason": "trace_not_derived", "detail": skipped[task_id]}]
            rows.append(schema_validated_row(task, failures[task_id], "trace_not_derived"))
            continue
        task_failures = validate_task(pack, task, traces[task_id], tool_index)
        failures[task_id] = task_failures
        rows.append(schema_validated_row(task, task_failures, REJECT_REASON))

    write_stage_table(
        stage_cache_dir(config) / SCHEMA_VALIDATED_TRACES,
        rows,
        schema_validated_traces_schema(),
    )
    rejected = sum(1 for items in failures.values() if items)
    logger.info("BFCL schema_validation checked %d tasks (%d rejected)", len(failures), rejected)
    return failures
