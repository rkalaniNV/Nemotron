# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Synchronous NeMo-RL runner for the pinned Lightning 3.5 step image."""

from __future__ import annotations

import json
import pprint
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

_RUNTIME_METADATA_KEYS = frozenset({"_ng_rollout_index", "_ng_task_index"})


def run_lightning35_grpo(
    *,
    config_path: Path,
    overrides: list[str] | None = None,
) -> None:
    """Run synchronous Lightning 3.5 NeMo-Gym GRPO on the pinned API."""
    from nemo_rl.algorithms.grpo import MasterConfig, _should_use_nemo_gym, grpo_train, setup
    from nemo_rl.algorithms.utils import get_tokenizer
    from nemo_rl.data.utils import setup_response_data
    from nemo_rl.distributed.virtual_cluster import init_ray
    from nemo_rl.environments.nemo_gym import setup_nemo_gym_config
    from nemo_rl.models.generation import configure_generation_config
    from nemo_rl.utils.config import register_omegaconf_resolvers
    from nemo_rl.utils.logger import get_next_experiment_dir

    from nemo_runspec.config.resolvers import register_manifest_resolver
    from nemotron.steps._runners.nemo_rl import load_nemo_rl_step_config

    register_omegaconf_resolvers()
    register_manifest_resolver()

    config_path = Path(config_path)
    config_omega = load_nemo_rl_step_config(config_path, overrides or [])
    config_dict = OmegaConf.to_container(config_omega, resolve=True)
    if not isinstance(config_dict, dict):
        raise TypeError(f"{config_path}: expected a mapping at the config root")

    # These fields control the Nemotron step launcher, not NeMo-RL itself.
    config_dict.pop("nemotron", None)
    config_dict.pop("run", None)
    config_dict.pop("nemo_rl_workdir", None)

    _reject_unsupported_modes(config_dict)
    validate_lightning35_pretrained_checkpoint(config_dict)
    validate_lightning35_nemo_gym_data(config_dict)

    config = MasterConfig(**config_dict)
    config.logger["log_dir"] = get_next_experiment_dir(config.logger["log_dir"])
    print(f"Using log directory: {config.logger['log_dir']}")
    if config.checkpointing["enabled"]:
        print(f"Using checkpoint directory: {config.checkpointing['checkpoint_dir']}")

    tokenizer = get_tokenizer(config.policy["tokenizer"])
    generation_config = config.policy["generation"]
    if generation_config is None:
        raise ValueError("A generation config is required for GRPO")

    megatron_config = config.policy.get("megatron_cfg") or {}
    config.policy["generation"] = configure_generation_config(
        generation_config,
        tokenizer,
        has_refit_draft_weights=bool(config.policy.get("draft") and config.policy["draft"].get("enabled")),
        trains_mtp=bool(megatron_config.get("enabled") and megatron_config.get("mtp_num_layers")),
    )

    setup_nemo_gym_config(config, tokenizer)
    if not _should_use_nemo_gym(config):
        raise ValueError("Lightning 3.5 requires env.should_use_nemo_gym=true")

    train_dataset, val_dataset = setup_response_data(
        tokenizer,
        config.data,
        env_configs=None,
    )
    set_lightning35_nemo_gym_validation_size(config.grpo, val_dataset)

    print("Final config:")
    pprint.pprint(config.model_dump())

    init_ray()
    (
        policy,
        policy_generation,
        nemo_gym,
        _cluster,
        dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        grpo_state,
        master_config,
        _teacher_worker_groups,
        _alias_to_group_alias,
    ) = setup(config, tokenizer, train_dataset, val_dataset)

    task_to_env = {"nemo_gym": nemo_gym}
    grpo_train(
        policy,
        policy_generation,
        dataloader,
        val_dataloader,
        tokenizer,
        loss_fn,
        task_to_env,
        task_to_env,
        logger,
        checkpointer,
        grpo_state,
        master_config,
    )


def _reject_unsupported_modes(config: dict[str, Any]) -> None:
    grpo = config.get("grpo") or {}
    async_config = grpo.get("async_grpo") or {}
    if async_config.get("enabled"):
        raise NotImplementedError(
            "The Lightning 3.5 step runner supports synchronous GRPO only; set grpo.async_grpo.enabled=false"
        )

    nemo_gym = (config.get("env") or {}).get("nemo_gym") or {}
    if nemo_gym.pop("is_trajectory_collection", False):
        raise NotImplementedError(
            "The Lightning 3.5 step runner does not support trajectory collection; "
            "set env.nemo_gym.is_trajectory_collection=false"
        )


def validate_lightning35_pretrained_checkpoint(config: dict[str, Any]) -> None:
    """Fail before Ray startup when the native Megatron checkpoint is unmounted."""
    checkpointing = config.get("checkpointing")
    pretrained = checkpointing.get("pretrained_checkpoint") if isinstance(checkpointing, dict) else None
    if not isinstance(pretrained, dict):
        raise ValueError("checkpointing.pretrained_checkpoint must be a mapping")
    if pretrained.get("format") != "megatron_bridge":
        raise ValueError("Lightning 3.5 requires checkpointing.pretrained_checkpoint.format=megatron_bridge")

    configured_path = pretrained.get("path")
    if not isinstance(configured_path, str) or not configured_path.strip():
        raise ValueError("checkpointing.pretrained_checkpoint.path must be a non-empty directory")
    checkpoint_path = Path(configured_path).expanduser()
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"Lightning 3.5 pretrained checkpoint directory was not found: {checkpoint_path}")


def validate_lightning35_nemo_gym_data(config: dict[str, Any]) -> None:
    """Reject stale collection IDs and agent/config mismatches before Ray startup."""
    from nemotron.steps._runners.nemo_rl_grpo_nemo_gym import materialize_nemo_gym_config_paths

    env = config.get("env") or {}
    nemo_gym = env.get("nemo_gym")
    if not env.get("should_use_nemo_gym") or not isinstance(nemo_gym, dict):
        raise ValueError("Lightning 3.5 requires an env.nemo_gym mapping")

    materialized = materialize_nemo_gym_config_paths(nemo_gym)
    if materialized.get("config_paths"):
        raise FileNotFoundError("Unable to resolve every env.nemo_gym.config_paths entry from the NeMo-RL workdir")
    configured_agents = {
        name
        for name, value in materialized.items()
        if isinstance(value, dict) and isinstance(value.get("responses_api_agents"), dict)
    }
    if not configured_agents:
        raise ValueError("env.nemo_gym.config_paths did not define any responses_api_agents")

    used_agents: dict[str, str] = {}
    runtime_metadata: dict[str, str] = {}
    data = config.get("data") or {}
    for split in ("train", "validation"):
        split_config = data.get(split)
        data_path = split_config.get("data_path") if isinstance(split_config, dict) else None
        if not isinstance(data_path, str) or not data_path.strip():
            raise ValueError(f"data.{split}.data_path must be a non-empty JSONL path")
        path = Path(data_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"NeMo-Gym {split} data file not found: {path}")

        with path.open(encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}: {error.msg}") from error
                if not isinstance(row, dict):
                    raise ValueError(f"Expected a JSON object at {path}:{line_number}")

                location = f"{path}:{line_number}"
                for key in _RUNTIME_METADATA_KEYS.intersection(row):
                    runtime_metadata.setdefault(key, location)

                agent_ref = row.get("agent_ref")
                agent_type = agent_ref.get("type") if isinstance(agent_ref, dict) else None
                agent_name = agent_ref.get("name") if isinstance(agent_ref, dict) else None
                if agent_type != "responses_api_agents" or not isinstance(agent_name, str):
                    raise ValueError(f"Invalid NeMo-Gym agent_ref at {location}: {agent_ref!r}")
                used_agents.setdefault(agent_name, location)

    if runtime_metadata:
        details = ", ".join(f"{key} at {location}" for key, location in sorted(runtime_metadata.items()))
        raise ValueError(
            "Synchronous NeMo-Gym data contains collection-time runtime metadata "
            f"({details}); rerun `data_prep/rl_prep -c lightning35`"
        )

    missing_agents = set(used_agents) - configured_agents
    if missing_agents:
        details = ", ".join(f"{name} at {used_agents[name]}" for name in sorted(missing_agents))
        raise ValueError(f"Prepared NeMo-Gym data references agents absent from env.nemo_gym.config_paths: {details}")


def set_lightning35_nemo_gym_validation_size(
    grpo_config: Any,
    val_dataset: Any,
) -> None:
    """Keep a bounded validation batch while retaining the final partial batch."""
    if grpo_config.max_val_samples is not None:
        raise ValueError("grpo.max_val_samples must be null for NeMo-Gym response data")

    val_size = len(val_dataset)
    if val_size == 0:
        raise ValueError("The NeMo-Gym validation dataset must not be empty")
    val_batch_size = grpo_config.val_batch_size
    if val_batch_size is None:
        val_batch_size = val_size
    if val_batch_size <= 0:
        raise ValueError("grpo.val_batch_size must be positive")

    val_batch_size = min(val_batch_size, val_size)
    grpo_config.val_batch_size = val_batch_size
    grpo_config.max_val_samples = ((val_size + val_batch_size - 1) // val_batch_size) * val_batch_size
