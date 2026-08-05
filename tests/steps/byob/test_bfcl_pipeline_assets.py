"""End-to-end assets produced by the BFCL generation slice on the tiny pack."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from nemotron.steps.byob.runtime.benchmark_families.bfcl.pipeline import generate_bfcl
from nemotron.steps.byob.runtime.benchmark_families.bfcl.row_schema import (
    decode_arguments,
    decode_tools,
)

BYOB_ROOT = Path(__file__).resolve().parents[3] / "src" / "nemotron" / "steps" / "byob"
BFCL_CONFIG_DIR = BYOB_ROOT / "bfcl" / "config"
BANKING_PACK_ROOT = BYOB_ROOT / "data" / "banking_vn_oracle_pack"


def _run_tiny(tmp_path: Path) -> tuple[list[dict[str, Any]], Path]:
    import pyarrow.parquet as pq

    config_data = yaml.safe_load((BFCL_CONFIG_DIR / "tiny.yaml").read_text(encoding="utf-8"))
    config_data["output_dir"] = str(tmp_path / "output")
    config_path = tmp_path / "tiny.yaml"
    config_path.write_text(yaml.safe_dump(config_data), encoding="utf-8")

    benchmark_path = generate_bfcl(config_path)
    assert benchmark_path is not None
    return pq.read_table(benchmark_path).to_pylist(), benchmark_path.parent


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[list[dict[str, Any]], Path]:
    return _run_tiny(tmp_path_factory.mktemp("tiny_slice"))


@pytest.fixture(scope="module")
def banking_run(tmp_path_factory: pytest.TempPathFactory) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    tmp_path = tmp_path_factory.mktemp("banking_slice")
    config_data = yaml.safe_load((BFCL_CONFIG_DIR / "banking_vn.yaml").read_text(encoding="utf-8"))
    config_data["output_dir"] = str(tmp_path / "output")
    config_path = tmp_path / "banking.yaml"
    config_path.write_text(yaml.safe_dump(config_data), encoding="utf-8")
    benchmark_path = generate_bfcl(config_path)
    assert benchmark_path is not None
    return pq.read_table(benchmark_path).to_pylist()


@pytest.fixture(scope="module")
def third_pack_run(tmp_path_factory: pytest.TempPathFactory) -> list[dict[str, Any]]:
    """Build a non-bundled domain pack to prove the runtime reads only the contract."""
    import pyarrow.parquet as pq

    tmp_path = tmp_path_factory.mktemp("third_pack_slice")
    pack = tmp_path / "asset_oracle"
    pack.mkdir()
    (pack / "manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "pack_id": "warehouse_assets",
                "version": "1.0",
                "languages": ["en"],
                "paths": {
                    "tools": "tools.json",
                    "backend": "backend.py",
                    "fixtures": "fixtures.json",
                    "templates": "task_templates.yaml",
                    "assertions": "assertions.py",
                    "validation_cases": "validation_cases.yaml",
                },
                "assistant_turn_templates": {"final_answer": {"en": "Inspection complete."}},
                "absent_ids": {"assets": ["ASSET-MISSING"]},
            }
        ),
        encoding="utf-8",
    )
    tools = [
        {
            "type": "function",
            "function": {
                "name": "inspect_asset",
                "description": "Read one warehouse asset.",
                "parameters": {
                    "type": "object",
                    "properties": {"asset_id": {"type": "string"}},
                    "required": ["asset_id"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    (pack / "tools.json").write_text(json.dumps(tools), encoding="utf-8")
    (pack / "fixtures.json").write_text(
        json.dumps({"assets": [{"asset_id": "ASSET-7", "condition": "ready"}]}),
        encoding="utf-8",
    )
    templates = [
        {
            "template_id": "asset_inspection",
            "intent": "inspect_asset_condition",
            "category": "warehouse",
            "difficulty": "easy",
            "turn_policy": "single_turn",
            "required_tools": ["inspect_asset"],
            "tools_present": ["inspect_asset"],
            "slots": {
                "asset_id": {
                    "source": "fixture:assets.asset_id",
                    "visible_in_first_turn": True,
                }
            },
            "success_assertions": ["assert_asset_returned"],
            "user_turn_templates": {"en": "What is the condition of asset ASSET-7?"},
            "assistant_milestones": [
                {"type": "tool_call", "tool": "inspect_asset"},
                {"type": "final_answer"},
            ],
        }
    ]
    (pack / "task_templates.yaml").write_text(
        yaml.safe_dump(templates, sort_keys=False),
        encoding="utf-8",
    )
    cases = [
        {
            "id": "asset_success",
            "tool": "inspect_asset",
            "arguments": {"asset_id": "ASSET-7"},
            "expect": {"result_class": "success", "condition": "ready"},
        },
        {
            "id": "asset_missing",
            "tool": "inspect_asset",
            "arguments": {"asset_id": "ASSET-MISSING"},
            "expect": {"result_class": "structured_error", "error_code": "not_found"},
        },
    ]
    (pack / "validation_cases.yaml").write_text(
        yaml.safe_dump(cases, sort_keys=False),
        encoding="utf-8",
    )
    (pack / "backend.py").write_text(
        """
from copy import deepcopy

_state = {}

def list_tools():
    return ["inspect_asset"]

def reset(*, ctx, fixtures=None):
    global _state
    _state = deepcopy(fixtures or {})

def call_tool(name, arguments, *, ctx):
    if name != "inspect_asset":
        return {"error": {"code": "unknown_tool"}}
    asset_id = arguments.get("asset_id")
    for asset in _state.get("assets", []):
        if asset.get("asset_id") == asset_id:
            return dict(asset)
    return {"error": {"code": "not_found"}}

def get_state():
    return deepcopy(_state)
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (pack / "assertions.py").write_text(
        """
def assert_asset_returned(*, state, trace, task, ctx):
    assert len(trace) == 1
    assert trace[0]["result"]["asset_id"] == task["slots"]["asset_id"]

ASSERTIONS = {"assert_asset_returned": assert_asset_returned}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    config_data = yaml.safe_load((BFCL_CONFIG_DIR / "tiny.yaml").read_text(encoding="utf-8"))
    config_data["expt_name"] = "third_pack_validation"
    config_data["output_dir"] = str(tmp_path / "output")
    config_data["oracle_pack"] = {"manifest_path": str(pack / "manifest.yaml")}
    config_data["oracle_runtime"]["allowed_roots"] = [str(pack)]
    config_data["task_generation"]["tasks_per_category"] = 1
    config_path = tmp_path / "third-pack.yaml"
    config_path.write_text(yaml.safe_dump(config_data), encoding="utf-8")

    benchmark_path = generate_bfcl(config_path)
    assert benchmark_path is not None
    return pq.read_table(benchmark_path).to_pylist()


def _row(rows: list[dict[str, Any]], template_id: str) -> dict[str, Any]:
    matches = [row for row in rows if row["template_id"] == template_id]
    assert matches, f"no row for template {template_id}"
    return matches[0]


def test_non_bundled_oracle_pack_runs_end_to_end(third_pack_run) -> None:
    assert len(third_pack_run) == 1
    row = third_pack_run[0]
    assert row["pack_id"] == "warehouse_assets"
    assert row["template_id"] == "asset_inspection"
    assert row["expected_tool_calls"][0]["function_name"] == "inspect_asset"
    assert decode_arguments(row["expected_tool_calls"][0]["arguments"]) == {
        "asset_id": "ASSET-7"
    }


def test_tiny_slice_produces_replay_validated_non_publishable_rows(tiny_run) -> None:
    rows, _ = tiny_run
    assert rows
    for row in rows:
        assert row["tier"] == "gold"
        assert row["gold_eligible"] is False
        assert "schema" in row["validated_by"]
        assert "replay" in row["validated_by"]
        # Null, not False: no held-out set was configured, so the row cannot claim it
        # was checked against one.
        assert row["held_out_hit"] is None
        assert row["task_id"].startswith("tiny_library__")
        assert row["messages"][0]["role"] == "system"
        assert row["system_prompt_id"].startswith("sha256:")


def test_every_template_reaches_the_benchmark(tiny_run) -> None:
    rows, _ = tiny_run
    assert {row["template_id"] for row in rows} == {
        "lib_status_single",
        "lib_checkout_confirm",
        "lib_status_parallel",
        "lib_irrelevant_renew",
    }


def test_parallel_group_is_one_assistant_message_with_two_calls(tiny_run) -> None:
    rows, _ = tiny_run
    row = _row(rows, "lib_status_parallel")

    assistant_with_calls = [
        message for message in row["messages"] if message["role"] == "assistant" and message["tool_calls"]
    ]
    assert len(assistant_with_calls) == 1
    assert len(assistant_with_calls[0]["tool_calls"]) == 2

    calls = row["expected_tool_calls"]
    assert [call["call_group"] for call in calls] == [0, 0]
    assert [call["position_in_group"] for call in calls] == [0, 1]
    assert {call["turn_index"] for call in calls} == {0}

    tool_messages = [message for message in row["messages"] if message["role"] == "tool"]
    issued = [call["id"] for call in assistant_with_calls[0]["tool_calls"]]
    assert [message["tool_call_id"] for message in tool_messages] == issued


def test_irrelevant_template_declines_without_calling_a_tool(tiny_run) -> None:
    rows, _ = tiny_run
    row = _row(rows, "lib_irrelevant_renew")

    assert row["expected_tool_calls"] == []
    assert row["num_tool_calls"] == 0
    assert row["required_tools"] == []
    assert all(not message["tool_calls"] for message in row["messages"])
    assert row["messages"][-1]["role"] == "assistant"
    assert row["messages"][-1]["content"]


def test_confirmation_trace_confirms_once_after_a_user_reply(tiny_run) -> None:
    rows, _ = tiny_run
    row = _row(rows, "lib_checkout_confirm")

    calls = row["expected_tool_calls"]
    assert len(calls) == 1
    arguments = decode_arguments(calls[0]["arguments"])
    assert arguments["confirm"] is True
    assert row["is_multi_turn"] is True

    roles = [message["role"] for message in row["messages"]]
    assert roles.count("user") == 2
    # The confirming call is issued by the second assistant message.
    assert calls[0]["turn_index"] == 1


def test_arguments_and_tools_round_trip_without_null_padding(tiny_run) -> None:
    rows, _ = tiny_run
    status_row = _row(rows, "lib_status_single")

    arguments = decode_arguments(status_row["expected_tool_calls"][0]["arguments"])
    assert arguments == {"book_id": "BK-100"}

    tools = decode_tools(status_row["tools"])
    assert {tool["function"]["name"] for tool in tools} == set(status_row["tools_present"])
    status_tool = next(tool for tool in tools if tool["function"]["name"] == "get_book_status")
    assert set(status_tool["function"]["parameters"]["properties"]) == {"book_id"}


def test_banking_rows_expose_only_declared_tools_and_confirm_mutations(banking_run) -> None:
    assert banking_run
    for row in banking_run:
        names = {tool["function"]["name"] for tool in decode_tools(row["tools"])}
        assert names == set(row["tools_present"])
        assert row["gold_eligible"] is False

    for template_id in ("bn_create_transfer_single", "bn_create_dispute_single"):
        row = _row(banking_run, template_id)
        assert row["is_multi_turn"] is True
        assert [message["role"] for message in row["messages"]].count("user") == 2
        assert row["expected_tool_calls"][0]["turn_index"] == 1


def test_banking_categories_share_one_budget(banking_run) -> None:
    config = yaml.safe_load((BFCL_CONFIG_DIR / "banking_vn.yaml").read_text(encoding="utf-8"))
    budget = config["task_generation"]["tasks_per_category"]
    templates = yaml.safe_load(
        (BANKING_PACK_ROOT / "task_templates.yaml").read_text(encoding="utf-8")
    )

    counts: dict[str, int] = {}
    for row in banking_run:
        counts[row["category"]] = counts.get(row["category"], 0) + 1
    assert counts
    assert max(counts.values()) <= budget
    # Splitting a budget across templates must not silently drop a template.
    assert {row["template_id"] for row in banking_run} == {
        template["template_id"] for template in templates
    }


def test_banking_covers_every_supported_policy_edge(banking_run) -> None:
    policies = {row["turn_policy"] for row in banking_run}
    assert policies == {
        "single_turn",
        "missing_slot",
        "confirmation",
        "correction",
        "multi_tool",
        "dependent_call",
        "negative_path",
        "clarify_only",
        "irrelevant",
    }
    assert any(row["num_tool_calls"] >= 2 for row in banking_run)
    assert all(
        set(row["required_tools"]) <= set(row["tools_present"]) for row in banking_run
    )
    assert any(set(row["required_tools"]) < set(row["tools_present"]) for row in banking_run)


def test_banking_withheld_slot_stays_out_of_the_first_user_turn(banking_run) -> None:
    row = _row(banking_run, "bn_balance_withheld_account")
    account_id = decode_arguments(row["expected_tool_calls"][0]["arguments"])["account_id"]
    user_turns = [message["content"] for message in row["messages"] if message["role"] == "user"]

    assert len(user_turns) == 2
    assert account_id not in user_turns[0]
    assert account_id in user_turns[1]
    assert row["expected_tool_calls"][0]["turn_index"] == 1


def test_banking_dependent_call_reads_its_id_from_the_first_result(banking_run) -> None:
    row = _row(banking_run, "bn_latest_txn_status_dependent")
    first, second = row["expected_tool_calls"]

    assert first["function_name"] == "list_recent_transactions"
    assert second["function_name"] == "get_transaction_status"
    assert first["call_group"] < second["call_group"]
    assert (first["turn_index"], second["turn_index"]) == (0, 1)

    listed = json.loads(
        next(message for message in row["messages"] if message["role"] == "tool")["content"]
    )
    checked_id = decode_arguments(second["arguments"])["transaction_id"]
    assert checked_id == listed["transactions"][0]["transaction_id"]
    assert checked_id not in decode_arguments(first["arguments"]).values()


def test_banking_correction_transfers_only_the_replacement_amount(banking_run) -> None:
    row = _row(banking_run, "bn_transfer_amount_corrected")
    (call,) = row["expected_tool_calls"]
    arguments = decode_arguments(call["arguments"])
    user_turns = [message["content"] for message in row["messages"] if message["role"] == "user"]

    assert arguments["confirm"] is True
    # The user stated one amount, replaced it, and confirmed again; only the
    # replacement may reach the mutating call.
    assert str(arguments["amount_vnd"]) in user_turns[1]
    assert str(arguments["amount_vnd"]) not in user_turns[0]
    assert len(user_turns) == 3
    assert call["turn_index"] == 2

    committed = json.loads(
        next(message for message in row["messages"] if message["role"] == "tool")["content"]
    )
    assert committed["status"] == "succeeded"
    assert committed["amount_vnd"] == arguments["amount_vnd"]


def test_banking_negative_paths_keep_the_failure_in_the_trace(banking_run) -> None:
    not_found = _row(banking_run, "bn_txn_status_unknown_id")
    tool_message = next(
        message for message in not_found["messages"] if message["role"] == "tool"
    )
    assert json.loads(tool_message["content"])["error"]["code"] == "not_found"

    rejected = _row(banking_run, "bn_transfer_short_of_funds")
    tool_message = next(message for message in rejected["messages"] if message["role"] == "tool")
    assert json.loads(tool_message["content"])["status"] == "rejected_insufficient_funds"
    assert decode_arguments(rejected["expected_tool_calls"][0]["arguments"])["confirm"] is True
    assert rejected["expected_tool_calls"][0]["turn_index"] == 1


def test_each_generation_stage_leaves_a_joinable_artifact(tiny_run) -> None:
    import pyarrow.parquet as pq

    rows, output_dir = tiny_run
    cache = output_dir / "stage_cache"
    tables = {
        name: pq.read_table(cache / f"{name}.parquet").to_pylist()
        for name in (
            "task_instances",
            "conversation_plans",
            "rendered_conversations",
            "expected_traces",
            "schema_validated_traces",
            "replay_validated_tasks",
        )
    }

    # Every stage keeps one row per task under the same key, so the intermediates can
    # be joined and a dropped row can be traced to the stage that dropped it.
    task_ids = {row["task_id"] for row in tables["task_instances"]}
    assert task_ids >= {row["task_id"] for row in rows}
    for name, table in tables.items():
        assert {row["task_id"] for row in table} == task_ids, name
        assert len(table) == len(task_ids), name

    plans = {row["task_id"]: row for row in tables["conversation_plans"]}
    traces = {row["task_id"]: row for row in tables["expected_traces"]}
    for row in rows:
        assert plans[row["task_id"]]["num_tool_calls"] == row["num_tool_calls"]
        assert traces[row["task_id"]]["num_tool_calls"] == len(row["expected_tool_calls"])
        assert json.loads(plans[row["task_id"]]["steps"])[0]["source"] == "first_turn"
    assert all(row["valid"] for row in tables["schema_validated_traces"])
    assert all(row["valid"] and row["reason"] is None for row in tables["replay_validated_tasks"])


def test_run_manifest_records_smoke_lineage_and_artifact_hashes(tiny_run) -> None:
    _, output_dir = tiny_run
    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))

    assert manifest["generation_mode"] == "smoke_no_publication"
    assert manifest["lineage_policy"] == "smoke_no_publication"
    assert manifest["tier"] == "gold"
    assert manifest["gold_eligible"] is False
    assert manifest["pack"]["pack_id"] == "tiny_library"
    assert manifest["models"]["paraphrase"]["enabled"] is False
    assert manifest["models"]["paraphrase"]["canonical_id"] is None
    assert manifest["generation_config_hash"] != manifest["resolved_config_hash"]
    assert manifest["stage_counts"]["expanded"] >= manifest["stage_counts"]["replay_passed"]
    assert manifest["stage_counts"]["trace_dropped"] == 0
    assert manifest["trace_drop_rejections"] == {"count": 0, "by_reason": {}}
    for artifact in (
        "benchmark_raw_parquet",
        "benchmark_parquet",
        "tools_model_facing",
        "oracle_validation_report",
        "pack_manifest",
        "fixtures_normalized",
        "task_templates_normalized",
        "task_instances",
        "conversation_plans",
        "rendered_conversations",
        "expected_traces",
        "schema_validated_traces",
        "replay_validated_tasks",
    ):
        assert manifest["artifacts"][artifact]["content_hash"].startswith("sha256:")
    assert manifest["stage_counts"]["replay_passed"] == manifest["stage_counts"]["published"]
    assert (output_dir / "benchmark_raw.parquet").exists()


def test_slice_is_deterministic_across_runs(tmp_path: Path, tiny_run) -> None:
    rows, _ = tiny_run
    rerun_rows, _ = _run_tiny(tmp_path)

    def fingerprint(items: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
        return sorted(
            (row["task_id"], row["seed"], json.dumps(row["messages"], sort_keys=True, default=str))
            for row in items
        )

    assert fingerprint(rows) == fingerprint(rerun_rows)
