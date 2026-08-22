"""Recover the gold episodes A0 produced, in the shape the worker wants them.

The stage tables serialise `slots`, `slots_initial` and `expected_tool_calls` as
canonical JSON strings. Assertions receive the pre-serialisation task dict, so
reading the parquet back without decoding those columns would hand every assertion
a string where it expects a mapping and turn the whole gate into a crash table.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from bfcl_ablation import common

_JSON_COLUMNS = ("slots", "slots_initial", "slot_updates")


def load_config_and_pack(config_path: Path) -> tuple[Any, Any]:
    common.bootstrap()
    from nemotron.steps.byob.runtime.benchmark_families.bfcl import pack_loader
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.config import BfclConfig

    config = BfclConfig.from_yaml(config_path)
    return config, pack_loader.load_pack(config)


def load_tasks(stage_cache: Path) -> list[dict[str, Any]]:
    tasks = []
    for row in common.read_parquet(stage_cache / "task_instances.parquet"):
        task = dict(row)
        for column in _JSON_COLUMNS:
            if isinstance(task.get(column), str):
                task[column] = json.loads(task[column])
        task["success_assertions"] = list(task.get("success_assertions") or [])
        tasks.append(task)
    return tasks


def load_traces(stage_cache: Path) -> dict[str, list[dict[str, Any]]]:
    return {
        row["task_id"]: json.loads(row["expected_tool_calls"])
        for row in common.read_parquet(stage_cache / "expected_traces.parquet")
        if row.get("derived")
    }


def replayed_task_ids(stage_cache: Path) -> set[str]:
    """Only tasks A0 actually replayed can be corrupted; the rest have no gold episode."""
    return {
        row["task_id"]
        for row in common.read_parquet(stage_cache / "replay_validated_tasks.parquet")
        if row.get("valid")
    }
