# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Static checks for Nemotron Customizer airgap compilation."""

from __future__ import annotations

from pathlib import Path

import yaml
from typer.testing import CliRunner

from nemo_runspec.execution import DEFAULT_OFFLINE_WHEELHOUSE
from nemotron.cli.commands.step import airgap_cmd
from nemotron.cli.commands.step.airgap_cmd import airgap_app, fetch_airgap
from nemotron.steps.airgap import (
    AIRGAP_CONTAINER_WHEELHOUSE,
    AirgapCompiler,
    AirgapTarget,
    build_delivery_plan,
    lock_to_dict,
    verify_lock,
)
from nemotron.steps.index import discover_steps


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_airgap_stage_is_discoverable() -> None:
    steps = {step.id: step for step in discover_steps()}

    assert "env/airgap" in steps
    assert steps["env/airgap"].produces[0].type == "airgap_lock"


def test_airgap_lock_includes_task_container_readiness() -> None:
    data = lock_to_dict(
        AirgapCompiler(repo_root=_repo_root()).compile(step_id="prep/sft_packing", config_name="tiny")
    )

    task = data["runtime"]["task_containers"][0]
    assert task["step_id"] == "prep/sft_packing"
    assert {"cosmos_xenna", "obstore"} <= set(task["candidate_imports"])
    packages = {hint.split("==", 1)[0] for hint in task["package_hints"]}
    assert {"cosmos-xenna", "obstore"} <= packages
    assert any(item["field"] == "output_dir" for item in data["manual_inputs"])


def test_airgap_wheelhouse_default_matches_runspec_fallback() -> None:
    assert AIRGAP_CONTAINER_WHEELHOUSE == DEFAULT_OFFLINE_WHEELHOUSE


def test_airgap_workflow_merges_task_container_readiness() -> None:
    data = AirgapCompiler(repo_root=_repo_root()).compile_many(
        [
            AirgapTarget("prep/sft_packing", "tiny"),
            AirgapTarget("sft/megatron_bridge", "tiny"),
        ],
        workflow_name="nano3-sft",
    )

    step_ids = {item["step_id"] for item in data["runtime"]["task_containers"]}
    assert {"prep/sft_packing", "sft/megatron_bridge"} <= step_ids
    assert data["delivery_plan"]["task_containers"]


def test_airgap_hf_field_local_path_override_becomes_manual_input() -> None:
    data = lock_to_dict(
        AirgapCompiler(repo_root=_repo_root()).compile(
            step_id="sft/megatron_bridge",
            config_name="tiny",
            overrides=["hf_model_path=/mnt/lustre-shared/models/nano"],
        )
    )

    assert any(item["id"] == "/mnt/lustre-shared/models/nano" for item in data["manual_inputs"])
    assert not any(asset["id"] == "/mnt/lustre-shared/models/nano" for asset in data["assets"])


def test_airgap_delivery_plan_has_task_image_stage() -> None:
    plan = build_delivery_plan(
        lock_to_dict(AirgapCompiler(repo_root=_repo_root()).compile(step_id="sft/megatron_bridge", config_name="tiny"))
    )

    assert plan["runtime_image"]["scope"] == "submitter"
    assert any(stage["stage"] == "3. Probe and rebuild task images" for stage in plan["stages"])
    assert any(item["repo_mounts"] for item in plan["task_containers"])


def test_airgap_probe_task_images_plan_writes_repo_mount_map(tmp_path: Path) -> None:
    lockfile = tmp_path / "airgap.lock.yaml"
    repo_mounts = tmp_path / "task-repo-mounts.json"
    data = lock_to_dict(
        AirgapCompiler(repo_root=_repo_root()).compile(step_id="sft/megatron_bridge", config_name="tiny")
    )
    lockfile.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    result = CliRunner().invoke(
        airgap_app,
        [
            "probe-task-images",
            str(lockfile),
            "--repo-mounts-output",
            str(repo_mounts),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Task image probe plan" in result.output
    mounts = yaml.safe_load(repo_mounts.read_text(encoding="utf-8"))
    assert {"repo": "Megatron-Bridge", "target": "/opt/Megatron-Bridge"} in mounts


def test_airgap_build_task_image_prints_checked_in_dockerfile(tmp_path: Path) -> None:
    lockfile = tmp_path / "airgap.lock.yaml"
    requirements = tmp_path / "task-requirements.txt"
    repo_mounts = tmp_path / "task-repo-mounts.json"
    requirements.write_text("", encoding="utf-8")
    repo_mounts.write_text("[]\n", encoding="utf-8")
    data = lock_to_dict(
        AirgapCompiler(repo_root=_repo_root()).compile(step_id="sft/megatron_bridge", config_name="tiny")
    )
    lockfile.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    result = CliRunner().invoke(
        airgap_app,
        [
            "build-task-image",
            str(lockfile),
            "--dockerfile",
            str(_repo_root() / "deploy/nemotron-customizer/airgap/Dockerfile.task"),
            "--task-requirements",
            str(requirements),
            "--task-repo-mounts",
            str(repo_mounts),
            "--tag",
            "customer/task:airgap",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Dockerfile.task" in result.output
    assert "TASK_REQUIREMENTS=" in result.output
    assert "customer/task:airgap" in result.output


def test_airgap_build_task_image_allows_missing_repo_mount_map(tmp_path: Path) -> None:
    lockfile = tmp_path / "airgap.lock.yaml"
    requirements = tmp_path / "task-requirements.txt"
    repo_mounts = tmp_path / "task-repo-mounts.json"
    requirements.write_text("", encoding="utf-8")
    data = lock_to_dict(
        AirgapCompiler(repo_root=_repo_root()).compile(step_id="prep/sft_packing", config_name="tiny")
    )
    lockfile.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    result = CliRunner().invoke(
        airgap_app,
        [
            "build-task-image",
            str(lockfile),
            "--dockerfile",
            str(_repo_root() / "deploy/nemotron-customizer/airgap/Dockerfile.task"),
            "--task-requirements",
            str(requirements),
            "--task-repo-mounts",
            str(repo_mounts),
            "--tag",
            "customer/task:airgap",
        ],
    )

    assert result.exit_code == 0, result.output
    assert repo_mounts.read_text(encoding="utf-8") == "[]\n"
    assert "wrote empty mount map" in result.output


def test_airgap_fetch_include_repos_does_not_fetch_hf_assets(monkeypatch, tmp_path: Path) -> None:
    lockfile = tmp_path / "airgap.lock.yaml"
    lockfile.write_text(
        yaml.safe_dump(
            {
                "assets": [
                    {
                        "kind": "hf_model",
                        "id": "org/model",
                        "delivery": "external",
                        "bundle_path": "assets/hf-cache/hub/models--org--model",
                    },
                    {
                        "kind": "git_repo",
                        "id": "Megatron-Bridge",
                        "delivery": "external",
                        "note": "https://example.invalid/Megatron-Bridge.git",
                        "revision": "main",
                    },
                ],
                "runtime": {"python": {"manager": "uv"}},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(airgap_cmd, "_fetch_hf", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError))
    monkeypatch.setattr(airgap_cmd, "_fetch_git", lambda asset, **_kwargs: f"git {asset['id']}")

    fetch_airgap(
        lockfile=lockfile,
        bundle_dir=tmp_path / "bundle",
        dry_run=False,
        include_wheels=False,
        include_repos=True,
        include_assets=False,
        tighten_lock=False,
    )

    assert (tmp_path / "bundle/assets/repos").exists()
    assert not (tmp_path / "bundle/assets/hf-cache/hub/models--org--model").exists()


def test_airgap_task_dockerfile_avoids_baking_hf_cache() -> None:
    dockerfile = (_repo_root() / "deploy/nemotron-customizer/airgap/Dockerfile.task").read_text(encoding="utf-8")

    assert "TASK_REPO_MOUNTS" in dockerfile
    assert "COPY ${AIRGAP_BUNDLE}/assets/repos/" in dockerfile
    assert "COPY ${AIRGAP_BUNDLE}/assets/hf-cache" not in dockerfile


def test_airgap_verify_checks_python_bundle_files(tmp_path: Path) -> None:
    data = lock_to_dict(
        AirgapCompiler(repo_root=_repo_root()).compile(step_id="prep/sft_packing", config_name="tiny")
    )

    codes = {issue.code for issue in verify_lock(data, bundle_dir=tmp_path)}

    assert "bundle_wheelhouse_missing" in codes
    assert "bundle_requirements_missing" in codes
    assert "bundle_offline_env_missing" in codes
