# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Static and CPU-only runtime checks for the shared AutoModel runner."""

import importlib
import os
import re
import sys
import tomllib
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from nemotron.steps._runners import automodel

from .._step_helpers import assert_step_static, step_dir


def test_sft_automodel_static() -> None:
    assert_step_static(
        step_dir(__file__, "sft", "automodel"),
        expected_name="steps/sft/automodel",
        expected_launch="torchrun",
        expected_default_config="default",
    )


def test_sft_automodel_default_is_full_sft_not_peft() -> None:
    directory = step_dir(__file__, "sft", "automodel")
    manifest = tomllib.loads((directory / "step.toml").read_text(encoding="utf-8"))
    config = yaml.safe_load((directory / "config" / "default.yaml").read_text(encoding="utf-8"))

    peft_param = next(param for param in manifest["parameters"] if param["name"] == "peft")
    assert peft_param["default"] == "null"
    assert "peft" not in config


class _Tokenizer:
    pad_token_id = 0

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return list(map(ord, text))


def test_preformatted_data_preserves_native_prompt_and_supervises_complete_output():
    prompt, output = "native prompt\u2028context", "User Safety: safe<EOT>"
    data = automodel.PreformattedTextDataset([{"input": prompt, "output": output}], _Tokenizer())
    sample = data[0]
    assert sample["input_ids"] == list(map(ord, prompt + output))[:-1]
    assert sample["labels"] == ([-100] * len(prompt) + list(map(ord, output)))[1:]
    assert sample["labels"][len(prompt) - 1] == ord(output[0])
    assert sample["labels"][-1] == ord(output[-1])
    assert sum(value != -100 for value in sample["labels"]) == len(output)
    sample["input_ids"].clear()
    sample.pop("___PAD_TOKEN_IDS___")
    assert data[0]["input_ids"]
    assert data[0]["___PAD_TOKEN_IDS___"]["labels"] == -100


@pytest.mark.parametrize("record,sequence_length,error", [
    ({"input": "abc", "output": "xy"}, 3, "exceeds"),
    ({"input": "abc", "output": ""}, 100, "nonempty"),
    ({"input": "abc", "output": "xy", "metadata": {"input_tokens": 4}}, 100, "tokenizer mismatch"),
])
def test_preformatted_data_rejects_truncation_and_tokenizer_mismatch(record, sequence_length, error):
    with pytest.raises(ValueError, match=error):
        automodel.PreformattedTextDataset(records=[record], tokenizer=_Tokenizer(), seq_length=sequence_length)


def test_image_text_factory_retains_full_model_and_freezes_only_selected_modules(monkeypatch):
    parameters = {}
    for name in ("model.language_model.layers.0.q_proj.weight", "lm_head.weight",
                 "model.vision_tower.encoder.weight", "model.multi_modal_projector.weight"):
        parameter = SimpleNamespace(requires_grad=True)
        parameter.requires_grad_ = lambda flag, target=parameter: setattr(target, "requires_grad", flag)
        parameters[name] = parameter
    model = SimpleNamespace(named_parameters=lambda: parameters.items(), enable_input_require_grads=Mock())
    config = SimpleNamespace(use_cache=True, text_config=SimpleNamespace(use_cache=True))
    transformers = ModuleType("transformers")
    transformers.AutoConfig = SimpleNamespace(from_pretrained=Mock(return_value=config))
    transformers.AutoModelForImageTextToText = SimpleNamespace(from_pretrained=Mock(return_value=model))
    torch = ModuleType("torch")
    torch.bfloat16 = object()
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    result = automodel.load_image_text_model(
        "original-model", revision="pinned-revision", freeze_modules=["*vision_tower*", "*multi_modal_projector*"]
    )
    assert result is model
    assert parameters["model.language_model.layers.0.q_proj.weight"].requires_grad
    assert parameters["lm_head.weight"].requires_grad
    assert not parameters["model.vision_tower.encoder.weight"].requires_grad
    assert not parameters["model.multi_modal_projector.weight"].requires_grad
    assert not config.use_cache and not config.text_config.use_cache
    model.enable_input_require_grads.assert_called_once()
    arguments = transformers.AutoModelForImageTextToText.from_pretrained.call_args.kwargs
    assert arguments["revision"] == "pinned-revision"
    assert arguments["torch_dtype"] is torch.bfloat16


def test_runner_import_needs_no_training_framework(monkeypatch):
    for name in ("torch", "transformers", "nemo_automodel", "wandb"):
        monkeypatch.setitem(sys.modules, name, None)
    importlib.reload(automodel)
    assert automodel.PreformattedTextDataset(records=[{"input": "a", "output": "b"}], tokenizer=_Tokenizer())


def test_direct_runner_resolves_inheritance_and_bare_overrides_before_interpolation(tmp_path, monkeypatch):
    base = tmp_path / "base.yaml"
    base.write_text("step_scheduler:\n  max_steps: 100\n  local_batch_size: 1\nmodel:\n  label: base\npeft:\n  dim: 16\n")
    child = tmp_path / "child.yaml"
    child.write_text("defaults: base.yaml\nreported_steps: ${step_scheduler.max_steps}\n")
    parser = ModuleType("nemo_automodel.components.config._arg_parser")

    def upstream_parse(default, *, argv):
        # Upstream accepts only dashed overrides; the old runner silently
        # discarded bare Nemotron dotlist arguments.
        dotlist = []
        index = 0
        while index < len(argv):
            token = argv[index]
            if token.startswith("--"):
                if "=" in token:
                    dotlist.append(token[2:])
                else:
                    dotlist.append(token[2:] + "=" + argv[index + 1])
                    index += 1
            index += 1
        return default, dotlist

    parser.parse_cli_argv = upstream_parse
    parser._resolve_wandb_enable = Mock()
    loader = ModuleType("nemo_automodel.components.config.loader")
    loader.ConfigNode = dict
    # Match upstream's surprising treatment of unquoted YAML null as a string.
    loader.translate_value = lambda value: value if value == "null" else yaml.safe_load(value)
    loader.resolve_yaml_env_vars = lambda value: value
    config_module = ModuleType("nemo_automodel.components.config")
    config_module._arg_parser = parser
    monkeypatch.setitem(sys.modules, "nemo_automodel", ModuleType("nemo_automodel"))
    monkeypatch.setitem(sys.modules, "nemo_automodel.components", ModuleType("nemo_automodel.components"))
    monkeypatch.setitem(sys.modules, "nemo_automodel.components.config", config_module)
    monkeypatch.setitem(sys.modules, "nemo_automodel.components.config.loader", loader)
    monkeypatch.setattr(sys, "argv", ["step.py", "--config", str(child), "step_scheduler.max_steps=20",
                                    "peft=null", "--literal", '"null"',
                                    "--model.label", "/cache/value=unchanged"])
    cfg = automodel._load_automodel_config(base)
    assert cfg["step_scheduler"] == {"max_steps": 20, "local_batch_size": 1}
    assert cfg["reported_steps"] == 20
    assert cfg["model"]["label"] == "/cache/value=unchanged"
    assert cfg["peft"] is None
    assert cfg["literal"] == "null"
    assert "defaults" not in cfg
    parser._resolve_wandb_enable.assert_called_once_with(cfg)


def test_config_values_preserve_legacy_environment_defaults_and_literal_dollars(monkeypatch):
    from omegaconf import OmegaConf

    def native_env(text):
        def braced(match):
            name, _, default = match.group(1).partition(",")
            return os.environ.get(name, default)
        text = re.sub(r"\$\{([^}]+)\}", braced, text)
        return re.sub(r"\$([A-Za-z_][A-Za-z0-9_]*)", lambda match: os.environ[match.group(1)], text)

    monkeypatch.setenv("LEGACY_MODEL", "/models/literal$VALUE")
    monkeypatch.setenv("OTHER_PATH", "ordinary")
    monkeypatch.delenv("ABSENT_SETTING", raising=False)
    config = OmegaConf.create({
        "model": {"path": "${LEGACY_MODEL}"},
        "copied_path": "${model.path}/$OTHER_PATH",
        "fallback": "${ABSENT_SETTING,default$literal}",
        "ordinary": "$OTHER_PATH",
        "selected": "${model.path}",
    })
    resolved = automodel._resolve_config_values(config, native_env)
    assert resolved["model"]["path"] == "/models/literal$VALUE"
    assert resolved["copied_path"] == "/models/literal$VALUE/ordinary"
    assert resolved["fallback"] == "default$literal"
    assert resolved["ordinary"] == "$OTHER_PATH"  # Native ConfigNode resolves this later.
    assert resolved["selected"] == "/models/literal$VALUE"
    assert resolved["selected"]._no_env_resolve is True
