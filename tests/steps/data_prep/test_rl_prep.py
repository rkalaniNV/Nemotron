# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Static checks for ``steps/data_prep/rl_prep``."""

import json
from copy import deepcopy
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from nemotron.data_prep.blend import DataBlend, Dataset
from nemotron.steps.data_prep.rl_prep.released_blend import (
    DAPO_QUESTION_DATASET,
    QUESTION_PLACEHOLDER_KEY,
    SKYWORK_QUESTION_DATASET,
    question_placeholder_restore_is_current,
    restore_question_placeholder_jsonl,
    restore_question_placeholder_row,
    run_released_jsonl_blend,
    split_local_jsonl,
)

from .._step_helpers import assert_step_static, step_dir


def test_rl_prep_static() -> None:
    assert_step_static(
        step_dir(__file__, "data_prep", "rl_prep"),
        expected_name="steps/data_prep/rl_prep",
        expected_launch="python",
        expected_default_config="default",
    )


def test_lightning35_prep_config_uses_released_blend_and_lepton_paths() -> None:
    config = OmegaConf.load(step_dir(__file__, "data_prep", "rl_prep") / "config" / "lightning35.yaml")
    blend = json.loads((step_dir(__file__, "data_prep", "rl_prep") / "data" / "blend_lightning35.json").read_text())

    assert config.blend_path == "data/blend_lightning35.json"
    assert config.prep_backend == "released_jsonl"
    assert config.val_holdout == 1000
    assert set(config.allowed_agent_names) == {
        "math_with_judge_simple_agent",
        "code_gen_simple_agent",
        "single_step_tool_use_with_argument_comparison_agent",
        "workplace_assistant_simple_agent",
        "mcqa_simple_agent",
        "instruction_following_simple_agent",
        "structured_outputs_simple_agent",
    }
    assert blend["datasets"][0]["path"] == "hf://nvidia/Nemotron-RL-Lightning-Training-Blend"


def test_split_local_jsonl_filters_agent_refs_before_holdout(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    rows = [
        {"id": 1, "agent_ref": {"type": "responses_api_agents", "name": "supported_a"}},
        {"id": 2, "agent_ref": {"type": "responses_api_agents", "name": "external_judge"}},
        {"id": 3, "agent_ref": {"type": "responses_api_agents", "name": "supported_b"}},
        {"id": 4, "agent_ref": {"type": "responses_api_agents", "name": "supported_a"}},
        {"id": 5, "agent_ref": {"type": "responses_api_agents", "name": "external_judge"}},
    ]
    source.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    result = split_local_jsonl(
        source,
        tmp_path / "output",
        val_holdout=1,
        allowed_agent_names=["supported_a", "supported_b"],
    )

    train_rows = [json.loads(line) for line in Path(result.train_path).read_text().splitlines()]
    val_rows = [json.loads(line) for line in Path(result.val_path or "").read_text().splitlines()]
    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))

    assert [row["id"] for row in train_rows] == [1, 3]
    assert [row["id"] for row in val_rows] == [4]
    assert manifest["source_rows"] == 5
    assert manifest["matched_rows"] == 3
    assert manifest["filtered_rows"] == 2
    assert manifest["agent_counts"] == {"supported_a": 2, "supported_b": 1}
    assert manifest["eligible_agent_counts"] == {"supported_a": 2, "supported_b": 1}


def test_restore_jsonl_strips_nemo_gym_runtime_metadata(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    rows = [
        {
            "id": 1,
            "_ng_task_index": 42,
            "_ng_rollout_index": 3,
            "metadata": {"_ng_task_index": "user-owned nested value"},
        },
        {
            "id": 2,
            "_ng_task_index": 99,
            "_ng_rollout_index": 7,
        },
    ]
    source.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    output = tmp_path / "restored.jsonl"
    restore_question_placeholder_jsonl(source, output, sources={})

    normalized = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    marker = json.loads(output.with_name("restored.jsonl.restore.json").read_text(encoding="utf-8"))

    assert "_ng_task_index" not in normalized[0]
    assert "_ng_rollout_index" not in normalized[0]
    assert "_ng_task_index" not in normalized[1]
    assert "_ng_rollout_index" not in normalized[1]
    assert normalized[0]["metadata"]["_ng_task_index"] == "user-owned nested value"
    assert marker["schema"] == 2
    assert marker["stripped_runtime_metadata_counts"] == {
        "_ng_rollout_index": 2,
        "_ng_task_index": 2,
    }


def test_split_local_jsonl_rejects_sample_that_leaves_no_training_rows(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text(
        "".join(json.dumps({"id": row_id}) + "\n" for row_id in range(3)),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=r"sample/max_rows \(2\) must be greater than val_holdout \(2\)",
    ):
        split_local_jsonl(
            source,
            tmp_path / "output",
            sample=2,
            val_holdout=2,
        )


def test_split_cache_rejects_truncated_output(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text(
        "".join(json.dumps({"id": row_id}) + "\n" for row_id in range(3)),
        encoding="utf-8",
    )
    output_dir = tmp_path / "output"
    first = split_local_jsonl(source, output_dir, val_holdout=1)
    Path(first.train_path).write_text("", encoding="utf-8")

    repaired = split_local_jsonl(source, output_dir, val_holdout=1)

    assert repaired.train_rows == 2
    assert len(Path(repaired.train_path).read_text(encoding="utf-8").splitlines()) == 2
    manifest = json.loads(Path(repaired.manifest_path).read_text(encoding="utf-8"))
    assert manifest["train_bytes"] == Path(repaired.train_path).stat().st_size


def test_released_snapshot_refresh_is_repo_keyed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, Path]] = []

    def fake_snapshot_download(**kwargs: object) -> str:
        repo_id = str(kwargs["repo_id"])
        local_dir = Path(str(kwargs["local_dir"]))
        calls.append((repo_id, local_dir))
        local_dir.mkdir(parents=True, exist_ok=True)
        rows = [{"repo": repo_id, "row": 1}, {"repo": repo_id, "row": 2}]
        (local_dir / "rlvr.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        return str(local_dir)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    output_dir = tmp_path / "output"
    first_blend = DataBlend.from_datasets(Dataset(name="first", path="hf://org/first", split="train"))
    second_blend = DataBlend.from_datasets(Dataset(name="second", path="hf://org/second", split="train"))

    run_released_jsonl_blend(blend=first_blend, output_dir=output_dir, val_holdout=1)
    second = run_released_jsonl_blend(blend=second_blend, output_dir=output_dir, val_holdout=1)

    assert [repo_id for repo_id, _ in calls] == ["org/first", "org/second"]
    assert calls[0][1] != calls[1][1]
    assert json.loads(Path(second.train_path).read_text(encoding="utf-8"))["repo"] == "org/second"


def test_released_blend_rejects_max_rows_before_download_when_no_train_split(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match=r"sample/max_rows \(1000\) must be greater than val_holdout \(1000\)",
    ):
        run_released_jsonl_blend(
            blend=object(),  # type: ignore[arg-type]
            output_dir=tmp_path,
            sample=1000,
            val_holdout=1000,
        )


def test_restore_dapo_question_placeholder() -> None:
    row = {
        QUESTION_PLACEHOLDER_KEY: {
            "dataset": DAPO_QUESTION_DATASET,
            "split": "train",
            "row": 0,
            "mode": "canonical",
            "lead": "Work carefully:\n",
            "trail": "\nPut the result in a box.",
        },
        "question": "masked",
        "expected_answer": "masked",
        "responses_create_params": {
            "input": [{"role": "user", "content": "masked"}],
        },
        "matched_sources": [{"expected_answer": "masked", "name": "dapo"}],
    }
    original = deepcopy(row)
    source_question = (
        "Solve the following math problem step by step. The last line of your response "
        "should be of the form Answer: $Answer (without quotes) where $Answer is the "
        "answer to the problem.\n\nWhat is 40 + 2?\n\n"
        'Remember to put your answer on its own line after "Answer:".'
    )
    sources = {
        (DAPO_QUESTION_DATASET, "train"): [
            {
                "prompt": [{"role": "user", "content": source_question}],
                "reward_model": {"ground_truth": "42"},
            }
        ]
    }

    restored = restore_question_placeholder_row(row, sources)

    expected_question = "Work carefully:\nWhat is 40 + 2?\nPut the result in a box."
    assert QUESTION_PLACEHOLDER_KEY not in restored
    assert restored["question"] == expected_question
    assert restored["expected_answer"] == "42"
    assert restored["responses_create_params"]["input"][0]["content"] == expected_question
    assert restored["matched_sources"][0]["expected_answer"] == "42"
    assert row == original


def test_restore_jsonl_is_atomic_and_cache_checked(tmp_path: Path) -> None:
    input_path = tmp_path / "masked.jsonl"
    output_path = tmp_path / "restored.jsonl"
    masked = {
        QUESTION_PLACEHOLDER_KEY: {
            "dataset": SKYWORK_QUESTION_DATASET,
            "split": "math",
            "row": 0,
            "prefix": "[",
            "suffix": "]",
        }
    }
    input_path.write_text(
        json.dumps({"id": "plain"}) + "\n" + json.dumps(masked) + "\n",
        encoding="utf-8",
    )
    sources = {
        (SKYWORK_QUESTION_DATASET, "math"): [
            {
                "prompt": [{"content": "question"}],
                "reward_model": {"ground_truth": '["answer"]'},
            }
        ]
    }

    assert restore_question_placeholder_jsonl(
        input_path,
        output_path,
        sources=sources,
    ) == (2, 1)
    assert question_placeholder_restore_is_current(input_path, output_path)
    assert [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()] == [
        {"id": "plain"},
        {"question": "[question]", "expected_answer": "answer"},
    ]

    output_path.write_text("truncated\n", encoding="utf-8")
    assert not question_placeholder_restore_is_current(input_path, output_path)
