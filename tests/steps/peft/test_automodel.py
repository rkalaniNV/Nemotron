# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Static checks for ``steps/peft/automodel``."""

from omegaconf import OmegaConf

from nemo_runspec.config.loader import apply_dotlist_overrides, extract_train_config, load_config

from .._step_helpers import assert_step_static, step_dir


def test_peft_automodel_static() -> None:
    assert_step_static(
        step_dir(__file__, "peft", "automodel"),
        expected_name="steps/peft/automodel",
        expected_launch="torchrun",
        expected_default_config="default",
    )


def test_content_safety_presets_share_data_and_preserve_native_model() -> None:
    steps = step_dir(__file__, "peft", "automodel").parents[1]
    sft = load_config(steps / "sft/automodel/config/content_safety.yaml")
    lora = load_config(steps / "peft/automodel/config/content_safety.yaml")

    assert sft.peft is None
    assert lora.peft.dim == 16
    assert lora.peft.alpha == 32
    assert not lora.peft.match_all_linear
    assert all("language_model" in pattern for pattern in lora.peft.target_modules)
    assert sft.checkpoint.save_consolidated == "final"
    assert lora.checkpoint.save_consolidated is False
    for key in ("model", "dataset", "validation_dataset", "distributed", "step_scheduler", "run"):
        assert OmegaConf.to_container(sft[key], resolve=False) == OmegaConf.to_container(lora[key], resolve=False)
    assert sft.model._target_ == "nemotron.steps._runners.automodel.load_image_text_model"
    assert "vision_tower" in " ".join(sft.model.freeze_modules)
    assert "multi_modal_projector" in " ".join(sft.model.freeze_modules)
    assert sft.run.env.nodes * sft.run.env.gpus_per_node == sft.distributed.dp_size == 32


def test_content_safety_compiled_config_is_self_contained_and_overridable(monkeypatch) -> None:
    monkeypatch.delenv("WANDB_NAME", raising=False)
    config = load_config(step_dir(__file__, "peft", "automodel") / "config/content_safety.yaml")
    config = apply_dotlist_overrides(config, ["wandb.name=unique-lora-run", "step_scheduler.max_steps=20"])
    compiled = extract_train_config(config, for_remote=True)
    resolved = OmegaConf.to_container(compiled, resolve=True)

    assert "defaults" not in resolved
    assert "run" not in resolved
    assert resolved["step_scheduler"]["max_steps"] == 20
    assert resolved["wandb"]["mode"] == "online"
    assert "unique-lora-run/checkpoints" in resolved["checkpoint"]["checkpoint_dir"]
    assert resolved["dataset"]["tokenizer"]["revision"] == resolved["model"]["revision"]
    assert resolved["validation_dataset"]["tokenizer"] == resolved["dataset"]["tokenizer"]
    assert resolved["validation_dataset"]["split"] == "validation"
    assert resolved["dataset"]["path_or_dataset_id"] == "nvidia/Nemotron-3.5-Content-Safety-Dataset"
    assert "content_safety_automodel" not in str(resolved)
    assert "content_safety_tracking" not in str(resolved)
