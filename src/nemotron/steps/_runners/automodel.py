# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared AutoModel runner used by sft/peft/pretrain step.py wrappers.

The recipe class can be overridden in YAML via a top-level
``_step_recipe: "module.path:ClassName"`` key. We deliberately avoid the
``recipe._target_`` slot because AutoModel's own config loader interprets
``_target_`` values as ``file/path.py:ClassName`` (file-style) — sharing the
slot causes a parse error.
"""

from __future__ import annotations

import importlib
import os
import re
import uuid
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Iterable

_STEP_RECIPE_KEY = "_step_recipe"


class PreformattedTextDataset:
    """Map-style dataset for already-rendered prompt/completion records.

    Tokenize the two fields separately, preserving the exact prompt tokens.
    AutoModel consumes next-token labels already shifted by one position. Only
    the completion is supervised; overlong records are rejected, never silently
    truncated. This class deliberately needs no training framework at import.
    """

    def __init__(
        self,
        records: Iterable[dict],
        tokenizer,
        seq_length: int = 2048,
    ):
        if tokenizer is None or tokenizer.pad_token_id is None:
            raise ValueError("A tokenizer with pad_token_id is required")
        if seq_length <= 0:
            raise ValueError("seq_length must be positive")
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.pad_token_id = tokenizer.pad_token_id
        self.samples = []
        self.rows = []

        for number, row in enumerate(records, 1):
            prompt, completion = row.get("input"), row.get("output")
            if not isinstance(prompt, str) or not prompt or not isinstance(completion, str) or not completion:
                raise ValueError(f"Record {number} requires nonempty input/output strings")
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            completion_ids = tokenizer.encode(completion, add_special_tokens=False)
            if not prompt_ids or not completion_ids:
                raise ValueError(f"Record {number} has empty tokenized input/output")
            ids = prompt_ids + completion_ids
            if len(ids) - 1 > seq_length:
                raise ValueError(f"Record {number} exceeds seq_length={seq_length}; prepare a shorter record")
            metadata = row.get("metadata") or {}
            for key, count in (("input_tokens", len(prompt_ids)), ("output_tokens", len(completion_ids))):
                if key in metadata and metadata[key] != count:
                    raise ValueError(f"Record {number} has a tokenizer mismatch for {key}")
            self.rows.append(row)
            self.samples.append({
                "input_ids": ids[:-1],
                "labels": ([-100] * len(prompt_ids) + completion_ids)[1:],
                "attention_mask": [1] * (len(ids) - 1),
            })
        if not self.samples:
            raise ValueError("No preformatted training records")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        # AutoModel's collator pops padding metadata. Return fresh containers.
        sample = {key: values.copy() for key, values in self.samples[index].items()}
        sample["___PAD_TOKEN_IDS___"] = {"input_ids": self.pad_token_id, "attention_mask": 0, "labels": -100}
        return sample


def load_image_text_model(
    pretrained_model_name_or_path: str,
    torch_dtype: str = "bfloat16",
    attn_implementation: str = "sdpa",
    trust_remote_code: bool = False,
    revision: str | None = None,
    freeze_modules: Iterable[str] = (),
    use_cache: bool = False,
    **model_kwargs,
):
    """Load a full HF image/text model for AutoModel's generic factory path.

    The existing AutoModel recipe applies its distributed and PEFT machinery
    after this factory returns. Explicit wildcard patterns freeze, for example,
    a vision tower and projector while leaving the complete language backbone
    trainable. The original architecture and all weights remain available.
    """
    import torch
    from transformers import AutoConfig, AutoModelForImageTextToText

    kwargs = {"trust_remote_code": trust_remote_code}
    if revision is not None:
        kwargs["revision"] = revision
    config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
    config.use_cache = use_cache
    if getattr(config, "text_config", None) is not None:
        config.text_config.use_cache = use_cache
    dtype = torch_dtype
    if isinstance(dtype, str) and dtype != "auto":
        dtype = getattr(torch, dtype.removeprefix("torch."))
    model = AutoModelForImageTextToText.from_pretrained(
        pretrained_model_name_or_path,
        config=config,
        torch_dtype=dtype,
        attn_implementation=attn_implementation,
        **kwargs,
        **model_kwargs,
    )
    patterns = [freeze_modules] if isinstance(freeze_modules, str) else list(freeze_modules)
    for name, parameter in model.named_parameters():
        if any(fnmatchcase(name, pattern) for pattern in patterns):
            parameter.requires_grad_(False)
    # Reentrant activation checkpointing needs differentiable embedding output
    # even when PEFT later freezes the base embedding parameters.
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    return model


def _normalize_overrides(arguments: list[str]) -> list[str]:
    """Translate Nemotron dotlist arguments for AutoModel's dashed CLI parser."""
    result = []
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument.startswith("-"):
            result.append(argument)
            # Keep a dashed option's separate value intact, including '=' in
            # paths or strings. Bare flags retain upstream parser semantics.
            if "=" not in argument and index + 1 < len(arguments) and not arguments[index + 1].startswith("--"):
                result.append(arguments[index + 1])
                index += 1
        else:
            result.append("--" + argument if "=" in argument else argument)
        index += 1
    return result


class _ResolvedEnvironment(str):
    # AutoModel respects this marker when constructing or instantiating nodes.
    _no_env_resolve = True


def _resolve_config_values(config, resolve_env):
    """Keep native ${VAR} environment syntax alongside OmegaConf references."""
    from omegaconf import OmegaConf

    values = {}
    marker = "__NEMOTRON_ENV_" + uuid.uuid4().hex + "_"
    legacy = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?:,([^{}]*))?\}")

    def walk(value, transform):
        if isinstance(value, dict):
            return {key: walk(item, transform) for key, item in value.items()}
        if isinstance(value, list):
            return [walk(item, transform) for item in value]
        return transform(value) if isinstance(value, str) else value

    def protect(text):
        def replace(match):
            name, default = match.groups()
            if default is None and name in config:
                return match.group(0)  # An ordinary OmegaConf config reference.
            # Resolve an existing variable through the native unbraced path:
            # re.sub then leaves dollar signs inside its value uninterpreted.
            if name in os.environ:
                value = resolve_env("$" + name)
            else:
                suffix = "" if default is None else "," + default.replace("$", marker + "dollar")
                safe = "${" + name + suffix + "}"
                value = resolve_env(safe).replace(marker + "dollar", "$")
            token = marker + str(len(values)) + "__"
            values[token] = value
            return token
        return legacy.sub(replace, text)

    def restore(text):
        if not any(token in text for token in values):
            return text
        # Original $VAR references still follow upstream semantics. Restore
        # protected environment values afterwards so their '$' stays literal.
        text = resolve_env(text)
        for token, value in values.items():
            text = text.replace(token, value)
        return _ResolvedEnvironment(text)

    raw = walk(OmegaConf.to_container(config, resolve=False), protect)
    resolved = OmegaConf.to_container(OmegaConf.create(raw), resolve=True)
    return walk(resolved, restore)


def _load_automodel_config(default_config: Path):
    from nemo_automodel.components.config import _arg_parser
    from nemo_automodel.components.config.loader import ConfigNode, resolve_yaml_env_vars, translate_value
    from nemo_runspec.config.loader import load_config
    from omegaconf import OmegaConf

    from nemotron.kit.train_script import parse_config_and_overrides

    config_path, overrides = parse_config_and_overrides(default_config=default_config)
    config_path, dotlist = _arg_parser.parse_cli_argv(
        str(config_path), argv=_normalize_overrides(overrides)
    )
    merged = load_config(Path(config_path))
    for override in dotlist:
        key, value = override.split("=", 1)
        # Nemotron documents YAML ``null`` for disabling PEFT. Upstream's
        # Python-oriented caster only recognizes ``None``/``none``; preserve
        # quoted "null" strings while accepting the unquoted YAML spelling.
        parsed = None if value.strip().lower() == "null" else translate_value(value)
        OmegaConf.update(merged, key, parsed, merge=True)
    # Resolve after overrides so references such as ${model.revision} record
    # the actual configured run, including direct step.py invocations.
    cfg = ConfigNode(_resolve_config_values(merged, resolve_yaml_env_vars))
    toggle = getattr(_arg_parser, "_resolve_wandb_enable", None)
    if toggle is not None:
        toggle(cfg)
    return cfg


def _patch_automodel_compat() -> None:
    """Patch known container-version mismatches before AutoModel setup.

    Some AutoModel images pair a Megatron dataset builder that reads
    ``GPTDatasetConfig.multiple_validation_sets`` with a config class that does
    not define that field yet. Setting a class-level default keeps single
    validation-set configs on the old behavior and avoids patching site-packages.
    """
    for module_path in (
        "megatron.core.datasets.gpt_dataset",
        "nemo_automodel.components.datasets.llm.megatron.gpt_dataset",
        "nemo_automodel.components.datasets.llm.megatron_dataset",
    ):
        try:
            module = importlib.import_module(module_path)
        except ImportError:
            continue

        config_cls = getattr(module, "GPTDatasetConfig", None)
        if config_cls is not None and not hasattr(config_cls, "multiple_validation_sets"):
            setattr(config_cls, "multiple_validation_sets", False)


def _resolve_recipe(default_target: str, cfg) -> type:
    """Return the recipe class. YAML override via ``_step_recipe`` key, else default."""
    target = default_target
    override = None
    if hasattr(cfg, _STEP_RECIPE_KEY):
        override = getattr(cfg, _STEP_RECIPE_KEY)
    elif isinstance(cfg, dict):
        override = cfg.get(_STEP_RECIPE_KEY)
    if override:
        target = override

    module_path, _, class_name = target.partition(":")
    if not class_name:
        # Allow Python dotted form: "pkg.mod.Class".
        module_path, _, class_name = target.rpartition(".")
    if not class_name:
        raise ValueError(f"Could not resolve recipe target: {target!r}")

    return getattr(importlib.import_module(module_path), class_name)


def run_automodel(*, default_target: str, default_config: Path) -> None:
    """Load YAML config, instantiate the AutoModel recipe class, run it.

    Args:
        default_target: ``"module.path:ClassName"`` of the recipe used when the
            YAML doesn't supply ``_step_recipe``.
        default_config: Default config path (the step's ``config/default.yaml``).
    """
    cfg = _load_automodel_config(default_config)

    _patch_automodel_compat()

    recipe_cls = _resolve_recipe(default_target, cfg)
    recipe = recipe_cls(cfg)
    recipe.setup()
    recipe.run_train_validation_loop()
