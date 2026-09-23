# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Static and runner checks for ``steps/rl/nemo_rl/rlvr``."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from nemotron.steps._runners import nemo_rl as nemo_rl_runner
from nemotron.steps._runners.nemo_rl import load_nemo_rl_step_config
from nemotron.steps._runners.nemo_rl_lightning35 import (
    _reject_unsupported_modes,
    set_lightning35_nemo_gym_validation_size,
    validate_lightning35_nemo_gym_data,
    validate_lightning35_pretrained_checkpoint,
)
from tests.steps._step_helpers import assert_step_static, step_dir

RLVR_STEP_DIR = step_dir(__file__, "rl", "nemo_rl", "rlvr")


def test_rl_rlvr_static() -> None:
    assert_step_static(
        RLVR_STEP_DIR,
        expected_name="steps/rl/nemo_rl/rlvr",
        expected_launch="ray",
        expected_default_config="default",
        require_workdir=True,
    )


@pytest.mark.parametrize("config_name", ["default", "tiny"])
def test_rlvr_sets_force_hf_for_automodel_weight_sync(config_name: str) -> None:
    cfg = load_nemo_rl_step_config(RLVR_STEP_DIR / "config" / f"{config_name}.yaml")

    assert OmegaConf.select(cfg, "policy.dtensor_cfg.automodel_kwargs.force_hf") is True


def test_lightning35_config_matches_pinned_runtime_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = "nvcr.io/example/nemo-rl:lightning35-patched"
    monkeypatch.setenv("LIGHTNING35_RL_IMAGE", image)
    cfg = load_nemo_rl_step_config(RLVR_STEP_DIR / "config" / "lightning35.yaml")

    assert OmegaConf.select(cfg, "nemotron.runner") == "lightning35"
    assert OmegaConf.select(cfg, "env.should_use_nemo_gym") is True
    assert OmegaConf.select(cfg, "policy.model_name") == ("nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16")
    assert OmegaConf.select(cfg, "policy.megatron_cfg.mtp_num_layers") == 5
    assert OmegaConf.select(cfg, "policy.megatron_cfg.expert_model_parallel_size") == 16
    assert OmegaConf.select(cfg, "policy.generation.val_temperature") == 1.0
    assert OmegaConf.select(cfg, "grpo.val_num_generations_per_prompt") == 4
    assert OmegaConf.select(cfg, "grpo.async_grpo.enabled") is False
    assert OmegaConf.select(cfg, "checkpointing.pretrained_checkpoint.format") == ("megatron_bridge")
    assert OmegaConf.select(cfg, "run.env.container_image") == image
    assert OmegaConf.select(cfg, "run.env.nodes") == 31
    assert OmegaConf.select(cfg, "cluster.num_nodes") == 32

    unresolved = OmegaConf.to_container(cfg, resolve=False)
    assert str(unresolved["data"]["train"]["data_path"]).startswith("${manifest:")


def test_lightning35_disables_incompatible_gradient_overlap() -> None:
    cfg = load_nemo_rl_step_config(RLVR_STEP_DIR / "config" / "lightning35.yaml")
    ddp = "policy.megatron_cfg.distributed_data_parallel_config"
    overrides = "policy.megatron_cfg.model_overrides"

    assert OmegaConf.select(cfg, f"{ddp}.overlap_grad_reduce") is False
    assert OmegaConf.select(cfg, f"{ddp}.overlap_param_gather") is True
    for callback in (
        "timers",
        "finalize_model_grads_func",
        "grad_scale_func",
        "moe_grad_scale_func",
        "mtp_grad_scale_func",
        "no_sync_func",
        "grad_sync_func",
        "param_sync_func",
    ):
        assert OmegaConf.select(cfg, f"{overrides}.{callback}") is None


def test_lightning35_dispatches_to_step_runner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "lightning35.yaml"
    config_path.write_text("nemotron:\n  runner: lightning35\n", encoding="utf-8")
    captured: dict = {}

    monkeypatch.setattr(
        nemo_rl_runner,
        "parse_nemo_rl_args",
        lambda **_kwargs: (
            argparse.Namespace(config=str(config_path)),
            ["grpo.max_num_steps=1"],
        ),
    )
    monkeypatch.setattr(
        "nemotron.steps._runners.nemo_rl_lightning35.run_lightning35_grpo",
        lambda **kwargs: captured.update(kwargs),
    )

    nemo_rl_runner.exec_or_run_nemo_rl_grpo(
        default_config=config_path,
        upstream_script="/unused.py",
        description="test",
    )

    assert captured == {
        "config_path": config_path,
        "overrides": ["grpo.max_num_steps=1"],
    }


def test_lightning35_runner_has_no_recipe_dependency() -> None:
    runner_path = RLVR_STEP_DIR.parents[2] / "_runners" / "nemo_rl_lightning35.py"
    source = runner_path.read_text(encoding="utf-8")

    assert "nemotron.recipes" not in source


def test_lightning35_validation_keeps_bounded_batches() -> None:
    config = SimpleNamespace(max_val_samples=None, val_batch_size=64)

    set_lightning35_nemo_gym_validation_size(config, [None] * 1000)

    assert config.max_val_samples == 1024
    assert config.val_batch_size == 64


def test_lightning35_validation_rejects_nonpositive_batch() -> None:
    config = SimpleNamespace(max_val_samples=None, val_batch_size=0)

    with pytest.raises(ValueError, match="must be positive"):
        set_lightning35_nemo_gym_validation_size(config, [None])


@pytest.mark.parametrize(
    "config",
    [
        {"grpo": {"async_grpo": {"enabled": True}}, "env": {"nemo_gym": {}}},
        {
            "grpo": {"async_grpo": {"enabled": False}},
            "env": {"nemo_gym": {"is_trajectory_collection": True}},
        },
    ],
)
def test_lightning35_rejects_unsupported_modes(config: dict) -> None:
    with pytest.raises(NotImplementedError):
        _reject_unsupported_modes(config)


def test_lightning35_removes_disabled_trajectory_flag() -> None:
    config = {
        "grpo": {"async_grpo": {"enabled": False}},
        "env": {"nemo_gym": {"is_trajectory_collection": False}},
    }

    _reject_unsupported_modes(config)

    assert "is_trajectory_collection" not in config["env"]["nemo_gym"]


def test_lightning35_validates_checkpoint_mount(tmp_path: Path) -> None:
    checkpoint = tmp_path / "iter_0000100"
    checkpoint.mkdir()

    validate_lightning35_pretrained_checkpoint(
        {
            "checkpointing": {
                "pretrained_checkpoint": {
                    "path": str(checkpoint),
                    "format": "megatron_bridge",
                }
            }
        }
    )

    with pytest.raises(FileNotFoundError, match="was not found"):
        validate_lightning35_pretrained_checkpoint(
            {
                "checkpointing": {
                    "pretrained_checkpoint": {
                        "path": str(tmp_path / "missing"),
                        "format": "megatron_bridge",
                    }
                }
            }
        )


def _lightning35_preflight_config(
    tmp_path: Path,
    *,
    train_agent: str = "supported_agent",
    runtime_metadata: bool = False,
) -> dict:
    gym_config = tmp_path / "gym.yaml"
    gym_config.write_text(
        "supported_agent:\n  responses_api_agents:\n    simple_agent:\n      entrypoint: app.py\n",
        encoding="utf-8",
    )

    row = {
        "agent_ref": {
            "type": "responses_api_agents",
            "name": train_agent,
        }
    }
    if runtime_metadata:
        row["_ng_task_index"] = 17
    for split in ("train", "validation"):
        (tmp_path / f"{split}.jsonl").write_text(
            json.dumps(row) + "\n",
            encoding="utf-8",
        )

    return {
        "env": {
            "should_use_nemo_gym": True,
            "nemo_gym": {"config_paths": [str(gym_config)]},
        },
        "data": {
            "train": {"data_path": str(tmp_path / "train.jsonl")},
            "validation": {"data_path": str(tmp_path / "validation.jsonl")},
        },
    }


def test_lightning35_data_preflight_accepts_matching_agents(tmp_path: Path) -> None:
    validate_lightning35_nemo_gym_data(_lightning35_preflight_config(tmp_path))


def test_lightning35_data_preflight_rejects_runtime_metadata(tmp_path: Path) -> None:
    config = _lightning35_preflight_config(tmp_path, runtime_metadata=True)

    with pytest.raises(ValueError, match="_ng_task_index"):
        validate_lightning35_nemo_gym_data(config)


def test_lightning35_data_preflight_rejects_missing_agent(tmp_path: Path) -> None:
    config = _lightning35_preflight_config(tmp_path, train_agent="missing_agent")

    with pytest.raises(ValueError, match="missing_agent"):
        validate_lightning35_nemo_gym_data(config)
