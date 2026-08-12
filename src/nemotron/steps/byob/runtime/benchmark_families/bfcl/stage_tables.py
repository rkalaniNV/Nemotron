"""Arrow schemas for the per-stage artifacts every generation stage leaves behind.

Each table is keyed by ``task_id`` and holds one row per task, so a run can be
inspected stage by stage and the tables can be joined without bookkeeping. Nested
content — slot bindings, plan steps, rendered turns, derived calls, failure lists —
is stored as canonical JSON text for the same reason the benchmark row does it: an
inferred Arrow struct unions keys across rows and pads the absent ones with nulls,
which turns "this task has no such field" into "this task's field is null".
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nemotron.steps.byob.runtime.benchmark_families.bfcl.row_schema import canonical_json

TASK_INSTANCES = "task_instances.parquet"
CONVERSATION_PLANS = "conversation_plans.parquet"
REFERENCE_SAMPLES = "reference_samples.parquet"
RENDERED_CONVERSATIONS = "rendered_conversations.parquet"
EXPECTED_TRACES = "expected_traces.parquet"
SCHEMA_VALIDATED_TRACES = "schema_validated_traces.parquet"
REPLAY_VALIDATED_TASKS = "replay_validated_tasks.parquet"

STAGE_TABLES = (
    TASK_INSTANCES,
    CONVERSATION_PLANS,
    RENDERED_CONVERSATIONS,
    EXPECTED_TRACES,
    SCHEMA_VALIDATED_TRACES,
    REPLAY_VALIDATED_TASKS,
)


def reference_samples_schema() -> Any:
    import pyarrow as pa

    return pa.schema(
        [
            ("sample_id", pa.string()),
            ("language", pa.string()),
            ("messages", pa.string()),
            ("tags", pa.list_(pa.string())),
            ("source_hash", pa.string()),
        ]
    )


def write_stage_table(path: Path, rows: list[dict[str, Any]], schema: Any) -> Path:
    """Write one stage artifact with an explicit schema, empty runs included."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)
    return path


def task_instances_schema() -> Any:
    import pyarrow as pa

    string_list = pa.list_(pa.string())
    return pa.schema(
        [
            ("task_id", pa.string()),
            ("base_task_id", pa.string()),
            ("template_id", pa.string()),
            ("pack_id", pa.string()),
            ("pack_version", pa.string()),
            ("intent", pa.string()),
            ("category", pa.string()),
            ("difficulty", pa.string()),
            ("turn_policy", pa.string()),
            ("call_order", pa.string()),
            ("call_order_prefix", pa.int32()),
            ("variant_index", pa.int32()),
            ("seed", pa.uint64()),
            ("mutates", pa.bool_()),
            ("required_tools", string_list),
            ("tools_present", string_list),
            ("success_assertions", string_list),
            ("fixture_refs", string_list),
            ("slots", pa.string()),
            ("slots_initial", pa.string()),
            ("slot_updates", pa.string()),
        ]
    )


def task_instance_row(task: dict[str, Any]) -> dict[str, Any]:
    """Project one locked task instance, slot timeline included."""
    return {
        "task_id": str(task["task_id"]),
        "base_task_id": str(task.get("base_task_id") or task["task_id"]),
        "template_id": str(task.get("template_id")),
        "pack_id": str(task.get("pack_id")),
        "pack_version": str(task.get("pack_version")),
        "intent": _optional_str(task.get("intent")),
        "category": _optional_str(task.get("category")),
        "difficulty": _optional_str(task.get("difficulty")),
        "turn_policy": _optional_str(task.get("turn_policy")),
        "call_order": str(task.get("call_order", "strict")),
        "call_order_prefix": task.get("call_order_prefix"),
        "variant_index": int(task.get("variant_index", 0)),
        "seed": int(task.get("seed", 0)),
        "mutates": bool(task.get("mutates", False)),
        "required_tools": [str(name) for name in task.get("required_tools") or []],
        "tools_present": [str(name) for name in task.get("tools_present") or []],
        "success_assertions": [str(name) for name in task.get("success_assertions") or []],
        "fixture_refs": [str(ref) for ref in task.get("fixture_refs") or []],
        "slots": canonical_json(task.get("slots") or {}),
        "slots_initial": canonical_json(task.get("slots_initial") or task.get("slots") or {}),
        "slot_updates": canonical_json(task.get("slot_updates") or []),
    }


def conversation_plans_schema() -> Any:
    import pyarrow as pa

    return pa.schema(
        [
            ("task_id", pa.string()),
            ("template_id", pa.string()),
            ("turn_policy", pa.string()),
            ("num_user_turns", pa.int32()),
            ("num_tool_calls", pa.int32()),
            ("is_multi_turn", pa.bool_()),
            ("has_user_confirmation", pa.bool_()),
            ("has_slot_correction", pa.bool_()),
            ("steps", pa.string()),
        ]
    )


def conversation_plan_row(task: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    """Project one conversation plan, with its ordered steps as JSON."""
    steps = [
        {
            key: value
            for key, value in (
                ("kind", step["kind"]),
                ("source", step.get("source")),
                ("milestone_type", step.get("milestone_type")),
                ("call_group", step.get("call_group")),
                ("update_index", step.get("update_index")),
                (
                    "tools",
                    [str(milestone.get("tool")) for milestone in step.get("milestones") or []] or None,
                ),
            )
            if value is not None
        }
        for step in plan["steps"]
    ]
    return {
        "task_id": str(plan["task_id"]),
        "template_id": str(plan["template_id"]),
        "turn_policy": _optional_str(task.get("turn_policy")),
        "num_user_turns": sum(1 for step in plan["steps"] if step["kind"] == "user"),
        "num_tool_calls": int(plan["num_tool_calls"]),
        "is_multi_turn": bool(plan["is_multi_turn"]),
        "has_user_confirmation": bool(plan["has_user_confirmation"]),
        "has_slot_correction": bool(plan["has_slot_correction"]),
        "steps": canonical_json(steps),
    }


def rendered_conversations_schema() -> Any:
    import pyarrow as pa

    return pa.schema(
        [
            ("task_id", pa.string()),
            ("base_task_id", pa.string()),
            ("variant_index", pa.int32()),
            ("source", pa.string()),
            ("language", pa.string()),
            ("system_prompt_id", pa.string()),
            ("paraphrase_model", pa.string()),
            ("paraphrase_model_canonical", pa.string()),
            ("profile_hash", pa.string()),
            ("num_user_turns", pa.int32()),
            ("accepted", pa.bool_()),
            ("guard_violations", pa.string()),
            ("turns", pa.string()),
        ]
    )


def rendered_conversation_row(surface: dict[str, Any]) -> dict[str, Any]:
    """Project one rendered conversation and its guard verdict."""
    violations = surface["guard_violations"]
    return {
        "task_id": str(surface["task_id"]),
        "base_task_id": str(surface.get("base_task_id") or surface["task_id"]),
        "variant_index": int(surface.get("variant_index", 0)),
        "source": str(surface.get("source", "template")),
        "language": str(surface["language"]),
        "system_prompt_id": str(surface["system_prompt_id"]),
        "paraphrase_model": _optional_str(surface.get("paraphrase_model")),
        "paraphrase_model_canonical": _optional_str(
            surface.get("paraphrase_model_canonical")
        ),
        "profile_hash": _optional_str(surface.get("profile_hash")),
        "num_user_turns": sum(1 for step in surface["steps"] if step["kind"] == "user"),
        "accepted": not violations,
        "guard_violations": canonical_json(violations),
        "turns": canonical_json(surface["steps"]),
    }


def expected_traces_schema() -> Any:
    import pyarrow as pa

    return pa.schema(
        [
            ("task_id", pa.string()),
            ("template_id", pa.string()),
            ("turn_policy", pa.string()),
            ("derived", pa.bool_()),
            ("drop_reason", pa.string()),
            ("num_tool_calls", pa.int32()),
            ("expected_tool_calls", pa.string()),
        ]
    )


def expected_trace_row(
    task: dict[str, Any],
    calls: list[dict[str, Any]],
    drop_reason: str | None = None,
) -> dict[str, Any]:
    """Project one derived trace; a task with no call, or a dropped one, keeps a row."""
    return {
        "task_id": str(task["task_id"]),
        "template_id": str(task.get("template_id")),
        "turn_policy": _optional_str(task.get("turn_policy")),
        "derived": drop_reason is None,
        "drop_reason": drop_reason,
        "num_tool_calls": len(calls),
        "expected_tool_calls": canonical_json(calls),
    }


def schema_validated_traces_schema() -> Any:
    import pyarrow as pa

    return pa.schema(
        [
            ("task_id", pa.string()),
            ("template_id", pa.string()),
            ("valid", pa.bool_()),
            ("reject_reason", pa.string()),
            ("failure_count", pa.int32()),
            ("failures", pa.string()),
        ]
    )


def schema_validated_row(
    task: dict[str, Any], failures: list[dict[str, Any]], reject_reason: str
) -> dict[str, Any]:
    """Project one schema verdict, keeping the failure detail next to it."""
    return {
        "task_id": str(task["task_id"]),
        "template_id": str(task.get("template_id")),
        "valid": not failures,
        "reject_reason": reject_reason if failures else None,
        "failure_count": len(failures),
        "failures": canonical_json(failures),
    }


def replay_validated_tasks_schema() -> Any:
    import pyarrow as pa

    return pa.schema(
        [
            ("task_id", pa.string()),
            ("template_id", pa.string()),
            ("valid", pa.bool_()),
            ("reason", pa.string()),
            ("detail", pa.string()),
            ("num_tool_results", pa.int32()),
            ("tool_results", pa.string()),
            ("assertions", pa.string()),
        ]
    )


def replay_validated_row(task: dict[str, Any], verdict: dict[str, Any]) -> dict[str, Any]:
    """Project one replay verdict.

    The final backend state is deliberately absent: it is the whole world, and only
    its equality across the two replays carries information, which ``reason`` already
    reports.
    """
    results = verdict.get("results") or []
    return {
        "task_id": str(task["task_id"]),
        "template_id": str(task.get("template_id")),
        "valid": bool(verdict.get("passed")),
        "reason": _optional_str(verdict.get("reason")),
        "detail": _optional_str(verdict.get("detail")),
        "num_tool_results": len(results),
        "tool_results": canonical_json(results),
        "assertions": canonical_json(verdict.get("assertions") or []),
    }


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)
