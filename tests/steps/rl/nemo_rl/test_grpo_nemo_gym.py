# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Focused checks for the NeMo-Gym GRPO runner and configs."""

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, sentinel

import pytest
from omegaconf import OmegaConf

from nemotron.steps._runners import nemo_rl_grpo_nemo_gym as gym_runner
from nemotron.steps._runners.nemo_rl import (
    load_nemo_rl_step_config,
    should_use_nemo_gym_config,
)
from nemotron.steps._runners.nemo_rl_grpo_nemo_gym import (
    materialize_nemo_gym_data_manifest,
    materialize_nemo_gym_response_data,
    materialize_nemo_gym_runtime_config,
    set_nemo_gym_validation_size,
    setup_initial_checkpoint,
    validate_async_grpo_config,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
RLVR_CONFIG = REPO_ROOT / "src/nemotron/steps/rl/nemo_rl/rlvr/config"
RLHF_CONFIG = REPO_ROOT / "src/nemotron/steps/rl/nemo_rl/rlhf/config"


@pytest.fixture
def nemo_rl_runtime(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Model both upstream contracts without importing GPU or Ray dependencies."""

    class NativeMasterConfig(SimpleNamespace):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.grpo = SimpleNamespace(**kwargs["grpo"])

        def model_dump(self):
            return vars(self) | {"grpo": vars(self.grpo)}

    generation = SimpleNamespace(
        cfg={"model_name": "policy"},
        dp_openai_server_base_urls=["http://policy/v1"],
    )
    legacy_env = SimpleNamespace(health_check=SimpleNamespace(remote=Mock(return_value=sentinel.health)))
    runtime = SimpleNamespace(
        master_config_type=NativeMasterConfig,
        master_config=Mock(side_effect=NativeMasterConfig),
        tokenizer=Mock(return_value=sentinel.tokenizer),
        generation=Mock(return_value=sentinel.generation_config),
        response_data=Mock(return_value=([sentinel.train], [1, 2, 3])),
        setup_gym=Mock(),
        setup=Mock(),
        train=Mock(),
        async_train=Mock(),
        init_ray=Mock(),
        create_env=Mock(return_value=legacy_env),
        env_config=Mock(return_value=sentinel.env_config),
        ray_get=Mock(),
        legacy_env=legacy_env,
        register_resolvers=Mock(),
        register_manifest=Mock(),
        register_artifacts=Mock(),
        clear_artifacts=Mock(),
        next_log_dir=Mock(return_value="logs/experiment_1"),
        setup_values=(
            sentinel.policy,
            generation,
            sentinel.cluster,
            sentinel.dataloader,
            sentinel.val_dataloader,
            sentinel.loss_fn,
            sentinel.logger,
            sentinel.checkpointer,
            sentinel.grpo_state,
            sentinel.master_config,
        ),
    )
    module_attrs = {
        "ray": {"get": runtime.ray_get},
        "nemo_rl.algorithms.grpo": {
            "MasterConfig": runtime.master_config,
            "_should_use_nemo_gym": lambda _config: True,
            "setup": runtime.setup,
            "grpo_train": runtime.train,
            "async_grpo_train": runtime.async_train,
        },
        "nemo_rl.algorithms.utils": {"get_tokenizer": runtime.tokenizer},
        "nemo_rl.data.utils": {"setup_response_data": runtime.response_data},
        "nemo_rl.distributed.virtual_cluster": {"init_ray": runtime.init_ray},
        "nemo_rl.environments.nemo_gym": {
            "NemoGymConfig": runtime.env_config,
            "setup_nemo_gym_config": runtime.setup_gym,
        },
        "nemo_rl.environments.utils": {"create_env": runtime.create_env},
        "nemo_rl.models.generation": {"configure_generation_config": runtime.generation},
        "nemo_rl.utils.config": {"register_omegaconf_resolvers": runtime.register_resolvers},
        "nemo_rl.utils.logger": {"get_next_experiment_dir": runtime.next_log_dir},
        "nemo_runspec.config.resolvers": {
            "register_manifest_resolver": runtime.register_manifest,
            "register_resolvers_from_config": runtime.register_artifacts,
            "clear_artifact_cache": runtime.clear_artifacts,
        },
    }
    modules = {}
    for name, attrs in module_attrs.items():
        parts = name.split(".")
        for index in range(1, len(parts) + 1):
            module_name = ".".join(parts[:index])
            if module_name not in modules:
                module = ModuleType(module_name)
                module.__path__ = []
                modules[module_name] = module
                monkeypatch.setitem(sys.modules, module_name, module)
        vars(modules[name]).update(attrs)

    runtime.legacy_hooks = {}
    for name in (
        "_maybe_chdir_to_nemo_rl_workdir",
        "materialize_nemo_gym_data_manifest",
        "materialize_nemo_gym_response_data",
        "_patch_wandb",
        "_setup_initial_policy",
    ):
        hook = Mock(name=name)
        monkeypatch.setattr(gym_runner, name, hook)
        runtime.legacy_hooks[name] = hook
    return runtime


@pytest.fixture
def nemo_gym_runner_config(tmp_path: Path) -> Path:
    data_path = tmp_path / "responses.jsonl"
    data_path.write_text(
        json.dumps({"agent_ref": {"type": "responses_api_agents", "name": "test_agent"}}) + "\n",
        encoding="utf-8",
    )
    checkpoint = tmp_path / "pretrained"
    checkpoint.mkdir()
    config = {
        "run": {"workdir": "/unused"},
        "nemo_rl_workdir": "/unused",
        "env": {
            "should_use_nemo_gym": True,
            "nemo_gym": {
                "is_trajectory_collection": False,
                "test_agent": {"responses_api_agents": {"simple_agent": {"entrypoint": "app.py"}}},
            },
        },
        "data": {"train": {"data_path": str(data_path)}, "validation": {"data_path": str(data_path)}},
        "grpo": {"max_val_samples": None, "val_batch_size": 2, "async_grpo": {"enabled": False}},
        "policy": {
            "tokenizer": {"name": "tokenizer"},
            "generation": {"backend": "vllm"},
            "draft": {"enabled": True},
            "megatron_cfg": {"enabled": True, "mtp_num_layers": 5},
        },
        "checkpointing": {
            "enabled": False,
            "pretrained_checkpoint": {"path": str(checkpoint), "format": "megatron_bridge"},
        },
        "logger": {"log_dir": "logs"},
    }
    config_path = tmp_path / "grpo.yaml"
    config_path.write_text(OmegaConf.to_yaml(config), encoding="utf-8")
    return config_path


@pytest.mark.parametrize(
    ("selector_override", "draft_enabled", "megatron_enabled", "mtp_layers"),
    [(False, True, True, 5), (True, False, True, 0), (True, True, False, 5)],
)
def test_shared_entrypoint_preserves_native_upstream_contract(
    nemo_rl_runtime: SimpleNamespace,
    nemo_gym_runner_config: Path,
    selector_override: bool,
    draft_enabled: bool,
    megatron_enabled: bool,
    mtp_layers: int,
) -> None:
    runtime = nemo_rl_runtime
    if not selector_override:
        config = OmegaConf.load(nemo_gym_runner_config)
        config.nemotron = {"runner": "lightning35"}
        nemo_gym_runner_config.write_text(OmegaConf.to_yaml(config), encoding="utf-8")
    overrides = [
        f"policy.draft.enabled={str(draft_enabled).lower()}",
        f"policy.megatron_cfg.enabled={str(megatron_enabled).lower()}",
        f"policy.megatron_cfg.mtp_num_layers={mtp_layers}",
    ]
    if selector_override:
        overrides.append("nemotron.runner=lightning35")
    runtime.setup.return_value = (
        *runtime.setup_values[:2],
        sentinel.upstream_gym,
        *runtime.setup_values[2:],
        sentinel.teacher_worker_groups,
        sentinel.alias_to_group_alias,
    )

    gym_runner.run_nemo_gym_grpo(config_path=nemo_gym_runner_config, overrides=overrides)

    runtime.master_config.assert_called_once()
    config = runtime.setup.call_args.args[0]
    assert isinstance(config, runtime.master_config_type)
    assert not {"run", "nemotron", "nemo_rl_workdir"}.intersection(runtime.master_config.call_args.kwargs)
    assert "is_trajectory_collection" not in config.env["nemo_gym"]
    assert config.grpo.max_val_samples == 4
    assert config.grpo.val_batch_size == 2
    runtime.generation.assert_called_once_with(
        {"backend": "vllm"},
        sentinel.tokenizer,
        has_refit_draft_weights=draft_enabled,
        trains_mtp=bool(megatron_enabled and mtp_layers),
    )
    runtime.setup_gym.assert_called_once_with(config, sentinel.tokenizer)
    runtime.response_data.assert_called_once_with(sentinel.tokenizer, config.data, env_configs=None)
    runtime.setup.assert_called_once_with(config, sentinel.tokenizer, [sentinel.train], [1, 2, 3])
    runtime.train.assert_called_once_with(
        *runtime.setup_values[:2],
        sentinel.dataloader,
        sentinel.val_dataloader,
        sentinel.tokenizer,
        sentinel.loss_fn,
        {"nemo_gym": sentinel.upstream_gym},
        {"nemo_gym": sentinel.upstream_gym},
        *runtime.setup_values[6:],
    )
    runtime.init_ray.assert_called_once_with()
    runtime.create_env.assert_not_called()
    runtime.env_config.assert_not_called()
    runtime.ray_get.assert_not_called()
    runtime.async_train.assert_not_called()
    runtime.register_artifacts.assert_not_called()
    for hook in runtime.legacy_hooks.values():
        hook.assert_not_called()


def test_shared_entrypoint_preserves_legacy_upstream_contract(
    nemo_rl_runtime: SimpleNamespace,
    nemo_gym_runner_config: Path,
) -> None:
    runtime = nemo_rl_runtime
    runtime.setup.return_value = runtime.setup_values

    gym_runner.run_nemo_gym_grpo(config_path=nemo_gym_runner_config)

    config = runtime.setup.call_args.args[0]
    assert isinstance(config, dict)
    assert config["grpo"]["max_val_samples"] == 3
    assert config["grpo"]["val_batch_size"] == 2
    runtime.master_config.assert_not_called()
    runtime.generation.assert_called_once_with({"backend": "vllm"}, sentinel.tokenizer)
    runtime.setup_gym.assert_called_once_with(config, sentinel.tokenizer)
    runtime.response_data.assert_called_once_with(
        tokenizer=sentinel.tokenizer, data_config=config["data"], env_configs=None
    )
    runtime.setup.assert_called_once_with(config, sentinel.tokenizer, [sentinel.train], [1, 2, 3])
    runtime.create_env.assert_called_once_with(env_name="nemo_gym", env_config=sentinel.env_config)
    runtime.env_config.assert_called_once_with(
        model_name="policy",
        base_urls=["http://policy/v1"],
        initial_global_config_dict={
            "test_agent": {"responses_api_agents": {"simple_agent": {"entrypoint": "app.py"}}}
        },
    )
    runtime.legacy_env.health_check.remote.assert_called_once_with()
    runtime.ray_get.assert_called_once_with(sentinel.health)
    runtime.train.assert_called_once_with(
        *runtime.setup_values[:2],
        sentinel.dataloader,
        sentinel.val_dataloader,
        sentinel.tokenizer,
        sentinel.loss_fn,
        {"nemo_gym": runtime.legacy_env},
        {"nemo_gym": runtime.legacy_env},
        *runtime.setup_values[6:],
    )
    runtime.init_ray.assert_called_once_with()
    runtime.async_train.assert_not_called()
    runtime.register_artifacts.assert_called_once()
    for hook in runtime.legacy_hooks.values():
        hook.assert_called_once_with(config)


def test_nemo_gym_dispatch_configs() -> None:
    assert should_use_nemo_gym_config(RLVR_CONFIG / "default.yaml") is False
    assert should_use_nemo_gym_config(RLVR_CONFIG / "nemo_gym.yaml") is True
    assert should_use_nemo_gym_config(RLHF_CONFIG / "default.yaml") is True
    assert (
        should_use_nemo_gym_config(
            RLHF_CONFIG / "default.yaml",
            ["env.should_use_nemo_gym=false"],
        )
        is False
    )


def test_rlhf_genrm_config_contract() -> None:
    for path in (RLHF_CONFIG / "default.yaml", RLHF_CONFIG / "tiny.yaml"):
        cfg = load_nemo_rl_step_config(path)
        resolved = OmegaConf.to_container(cfg, resolve=True)
        nemo_gym = resolved["env"]["nemo_gym"]

        genrm_impls = nemo_gym["genrm_model"]["responses_api_models"]
        assert resolved["env"]["use_genrm_compare"] is True
        assert list(genrm_impls) == ["genrm_model"]
        assert nemo_gym["genrm_model_name"]
        assert (
            nemo_gym["genrm_compare"]["resources_servers"]["genrm_compare"]["num_rollouts_per_prompt"]
            == resolved["grpo"]["num_generations_per_prompt"]
        )
        assert resolved["data"]["validation"]["repeat"] == resolved["grpo"]["num_generations_per_prompt"]

    tiny = OmegaConf.to_container(load_nemo_rl_step_config(RLHF_CONFIG / "tiny.yaml"), resolve=True)
    assert tiny["cluster"]["num_nodes"] == 1
    assert tiny["policy"]["generation"]["colocated"]["resources"]["num_nodes"] == 1
    assert tiny["env"]["nemo_gym"]["num_gpu_nodes"] == 1


def test_nemo_gym_configs_follow_upstream_runtime_contract() -> None:
    for path in (RLHF_CONFIG / "default.yaml", RLVR_CONFIG / "nemo_gym.yaml"):
        cfg = load_nemo_rl_step_config(path)
        assert OmegaConf.select(cfg, "grpo.max_val_samples") is None
        assert OmegaConf.select(cfg, "grpo.val_batch_size") is None
        assert OmegaConf.select(cfg, "data.max_input_seq_length") is None
        assert OmegaConf.select(cfg, "data.num_workers") == 0
        assert OmegaConf.select(cfg, "env.should_log_nemo_gym_responses") is True
        assert OmegaConf.select(cfg, "env.nemo_gym.rollout_max_attempts_to_avoid_lp_nan") == 1
        assert OmegaConf.select(cfg, "env.nemo_gym.is_trajectory_collection") is False
        assert OmegaConf.select(cfg, "policy.generation.vllm_cfg.async_engine") is False
        assert OmegaConf.select(cfg, "policy.generation.vllm_cfg.enforce_eager") is True
        assert (
            OmegaConf.select(
                cfg,
                "env.nemo_gym.policy_model.responses_api_models.vllm_model.uses_reasoning_parser",
            )
            is False
        )
        assert (
            OmegaConf.select(
                cfg,
                "env.nemo_gym.policy_model.responses_api_models.vllm_model.extra_body.chat_template_kwargs.enable_thinking",
            )
            is False
        )

    rlhf_config_paths = OmegaConf.select(
        load_nemo_rl_step_config(RLHF_CONFIG / "default.yaml"),
        "env.nemo_gym.config_paths",
    )
    assert (
        "resources_servers/single_step_tool_use_with_argument_comparison/configs/"
        "single_step_tool_use_with_argument_comparison.yaml" not in rlhf_config_paths
    )


@pytest.mark.parametrize(
    ("manifest", "allow_train_as_validation", "expected_validation"),
    [
        ({"train": "/data/train.jsonl", "validation": "/data/val.jsonl"}, False, "/data/val.jsonl"),
        ({"train": "/data/train.jsonl"}, True, "/data/train.jsonl"),
    ],
)
def test_materialize_nemo_gym_data_manifest(
    tmp_path: Path,
    manifest: dict[str, str],
    allow_train_as_validation: bool,
    expected_validation: str,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    config = {
        "data": {
            "manifest_path": str(manifest_path),
            "allow_train_as_validation": allow_train_as_validation,
            "train": {"split": "train"},
            "validation": {"split": "validation"},
        }
    }

    materialize_nemo_gym_data_manifest(config)

    assert config["data"]["train"] == {"split": "train", "data_path": "/data/train.jsonl"}
    assert config["data"]["validation"] == {
        "split": "validation",
        "data_path": expected_validation,
    }
    assert "manifest_path" not in config["data"]
    assert "allow_train_as_validation" not in config["data"]


def test_materialize_nemo_gym_response_data_normalizes_rows(tmp_path: Path) -> None:
    train_path = tmp_path / "train.jsonl"
    train_path.write_text(
        json.dumps({"question": "What is 2+2?", "tools": [{"type": "function"}]}) + "\n",
        encoding="utf-8",
    )
    val_path = tmp_path / "val.jsonl"
    val_path.write_text(
        json.dumps({"extra_env_info": {"responses_create_params": {"input": [{"role": "user", "content": "hello"}]}}})
        + "\n",
        encoding="utf-8",
    )
    config = {
        "data": {
            "normalized_data_dir": str(tmp_path / "normalized"),
            "default_agent_ref": "genrm_simple_agent",
            "train": {"data_path": str(train_path)},
            "validation": {"data_path": str(val_path)},
        }
    }

    materialize_nemo_gym_response_data(config)

    normalized_train_path = Path(config["data"]["train"]["data_path"])
    normalized_val_path = Path(config["data"]["validation"]["data_path"])
    train_row = json.loads(normalized_train_path.read_text(encoding="utf-8"))
    val_row = json.loads(normalized_val_path.read_text(encoding="utf-8"))

    assert normalized_train_path.parent == tmp_path / "normalized"
    assert train_row["responses_create_params"] == {
        "input": [{"role": "user", "content": "What is 2+2?"}],
        "tools": [{"type": "function"}],
    }
    assert val_row["responses_create_params"]["input"][0]["content"] == "hello"
    assert train_row["agent_ref"] == {
        "type": "responses_api_agents",
        "name": "genrm_simple_agent",
    }
    assert "default_agent_ref" not in config["data"]


def test_materialize_nemo_gym_response_data_requires_agent_ref(tmp_path: Path) -> None:
    data_path = tmp_path / "train.jsonl"
    data_path.write_text(json.dumps({"question": "hello"}) + "\n", encoding="utf-8")
    config = {"data": {"train": {"data_path": str(data_path)}}}

    with pytest.raises(ValueError, match="agent_ref"):
        materialize_nemo_gym_response_data(config)


def test_set_nemo_gym_validation_size() -> None:
    config = {"grpo": {"max_val_samples": None, "val_batch_size": 2}}
    set_nemo_gym_validation_size(config, [object(), object(), object()])
    assert config["grpo"] == {"max_val_samples": 3, "val_batch_size": 2}

    with pytest.raises(ValueError, match="max_val_samples"):
        set_nemo_gym_validation_size(
            {"grpo": {"max_val_samples": 2, "val_batch_size": 1}},
            [object()],
        )


def test_validate_async_grpo_config_rejects_unsupported_features() -> None:
    config = {
        "grpo": {
            "use_dynamic_sampling": True,
            "reward_scaling": {"enabled": False},
            "reward_shaping": {"enabled": False},
        },
        "data": {"use_multiple_dataloader": False},
    }

    with pytest.raises(NotImplementedError, match="use_dynamic_sampling"):
        validate_async_grpo_config(config)


def test_materialize_nemo_gym_runtime_config_normalizes_genrm_and_policy_alias(
    tmp_path: Path,
) -> None:
    inherited_config = tmp_path / "genrm_compare.yaml"
    inherited_config.write_text(
        """
genrm_simple_agent_reasoning_off:
  responses_api_agents:
    simple_agent:
      model:
        type: responses_api_models
        name: policy_model_reasoning_off
genrm_model:
  responses_api_models:
    genrm_model:
      entrypoint: app.py
      model: ${genrm_model_name}
""",
        encoding="utf-8",
    )
    config = {
        "config_paths": [str(inherited_config)],
        "genrm_model_name": "reward",
        "genrm_model": {
            "responses_api_models": {
                "vllm_model": {
                    "entrypoint": "app.py",
                    "model": "stale",
                    "spinup_server": True,
                }
            }
        },
    }

    materialized = materialize_nemo_gym_runtime_config(
        config,
        policy_model_name="policy",
        policy_base_urls=["http://policy-0/v1"],
    )

    assert "config_paths" not in materialized
    assert "policy_model_reasoning_off" not in config
    assert materialized["genrm_model"]["responses_api_models"] == {
        "genrm_model": {
            "entrypoint": "app.py",
            "model": "reward",
        }
    }

    policy_alias = materialized["policy_model_reasoning_off"]["responses_api_models"]["vllm_model"]
    assert policy_alias["base_url"] == ["http://policy-0/v1"]
    assert policy_alias["model"] == "policy"
    assert policy_alias["return_token_id_information"] is True
    assert policy_alias["uses_reasoning_parser"] is False
    assert policy_alias["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False


def test_materialize_nemo_gym_runtime_config_does_not_add_unused_policy_alias() -> None:
    materialized = materialize_nemo_gym_runtime_config(
        {"policy_model": {"responses_api_models": {"vllm_model": {"model": "policy"}}}},
        policy_model_name="policy",
        policy_base_urls=["http://policy-0/v1"],
    )

    assert "policy_model_reasoning_off" not in materialized


def test_materialize_nemo_gym_runtime_config_collapses_generic_duplicate_model_server() -> None:
    materialized = materialize_nemo_gym_runtime_config(
        {
            "reward_model_name": "reward",
            "reward_model": {
                "responses_api_models": {
                    "reward_model": {
                        "entrypoint": "app.py",
                        "model": "${reward_model_name}",
                    },
                    "vllm_model": {
                        "entrypoint": "app.py",
                        "model": "stale",
                    },
                }
            },
        },
        policy_model_name="policy",
        policy_base_urls=["http://policy-0/v1"],
    )

    assert materialized["reward_model"]["responses_api_models"] == {
        "reward_model": {
            "entrypoint": "app.py",
            "model": "reward",
        }
    }


def test_runner_uses_upstream_nemo_rl_gym_primitives() -> None:
    source = (REPO_ROOT / "src/nemotron/steps/_runners/nemo_rl_grpo_nemo_gym.py").read_text(encoding="utf-8")

    assert "setup_response_data" in source
    assert "create_env" in source
    assert "async_grpo_train" in source
    assert "_should_use_async_rollouts" not in source
    assert "CompatNemoGym" not in source
    assert "RolloutCollectionHelper" not in source
    assert "setup_nemo_gym_jsonl_dataset" not in source


def test_setup_initial_checkpoint_writes_complete_marker(tmp_path: Path) -> None:
    source = tmp_path / "source" / "iter_0000001"
    source.mkdir(parents=True)
    (source / "model_optim_rng.pt").write_text("weights", encoding="utf-8")
    checkpoint_dir = tmp_path / "checkpoints"

    setup_initial_checkpoint(str(source.parent), str(checkpoint_dir))

    step_dir = checkpoint_dir / "step_0"
    target = step_dir / "policy" / "weights" / "model_optim_rng.pt"
    assert target.is_symlink()
    assert target.resolve() == source / "model_optim_rng.pt"
    assert (step_dir / ".complete").read_text(encoding="utf-8") == "ok\n"
    info = json.loads((step_dir / "training_info.json").read_text(encoding="utf-8"))
    assert info["initial_checkpoint"] == str(source.parent)

    setup_initial_checkpoint(str(source.parent), str(checkpoint_dir))


def test_setup_initial_checkpoint_refuses_incomplete_existing_view(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    checkpoint_dir = tmp_path / "checkpoints"
    (checkpoint_dir / "step_0" / "policy" / "weights").mkdir(parents=True)

    with pytest.raises(RuntimeError, match=r"\.complete is missing"):
        setup_initial_checkpoint(str(source), str(checkpoint_dir))
