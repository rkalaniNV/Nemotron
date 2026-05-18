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

"""Tests for :mod:`nemo_runspec.data_mover`."""

from __future__ import annotations

import os
import tarfile
from pathlib import Path

import pytest

from nemo_runspec.data_mover import Plan, SourcePackager, _auto_includes, plan_for

# ── _auto_includes ───────────────────────────────────────────────────────────


def _write_fake_repo(root: Path) -> None:
    """Build a synthetic repo layout mirroring this project."""
    src = root / "src"
    # nemo_runspec package (no recipes/)
    (src / "nemo_runspec").mkdir(parents=True)
    (src / "nemo_runspec" / "__init__.py").write_text("")
    (src / "nemo_runspec" / "foo.py").write_text("x=1")
    # nemotron package with two recipe families
    (src / "nemotron").mkdir()
    (src / "nemotron" / "__init__.py").write_text("")
    (src / "nemotron" / "kit").mkdir()
    (src / "nemotron" / "kit" / "__init__.py").write_text("")
    recipes = src / "nemotron" / "recipes"
    recipes.mkdir()
    (recipes / "__init__.py").write_text("")
    (recipes / "nano3").mkdir()
    (recipes / "nano3" / "__init__.py").write_text("")
    (recipes / "super3").mkdir()
    (recipes / "super3" / "__init__.py").write_text("")
    # Noise that must be filtered from the packager
    (src / "nemotron" / "__pycache__").mkdir()
    (src / "nemotron" / "__pycache__" / "x.pyc").write_text("bytecode")


def test_auto_includes_scopes_to_active_recipe_family(tmp_path):
    _write_fake_repo(tmp_path)
    includes = _auto_includes(tmp_path, script_path="src/nemotron/recipes/nano3/stage1_sft/data_prep.py")
    # Active family included, inactive one omitted.
    assert "src/nemotron/recipes/nano3" in includes
    assert "src/nemotron/recipes/super3" not in includes
    # Sibling package always shipped.
    assert "src/nemo_runspec" in includes
    # Non-recipes children of nemotron shipped.
    assert "src/nemotron/kit" in includes


def test_auto_includes_ships_all_families_without_hint(tmp_path):
    _write_fake_repo(tmp_path)
    includes = _auto_includes(tmp_path, script_path=None)
    # No family hint → ship both.
    assert "src/nemotron/recipes/nano3" in includes
    assert "src/nemotron/recipes/super3" in includes


def test_auto_includes_scopes_to_active_step_subtree(tmp_path):
    _write_fake_repo(tmp_path)
    steps = tmp_path / "src" / "nemotron" / "steps"
    (steps / "_runners").mkdir(parents=True)
    (steps / "_runners" / "__init__.py").write_text("")
    (steps / "index.py").write_text("")
    (steps / "sft" / "automodel").mkdir(parents=True)
    (steps / "sft" / "__init__.py").write_text("")
    (steps / "sft" / "automodel" / "step.py").write_text("")
    (steps / "data_prep").mkdir()
    (steps / "data_prep" / "__init__.py").write_text("")
    (steps / "data_prep" / "_common.py").write_text("")
    (steps / "data_prep" / "sft_packing").mkdir()
    (steps / "data_prep" / "sft_packing" / "step.py").write_text("")
    (steps / "rl" / "nemo_rl").mkdir(parents=True)
    (steps / "rl" / "nemo_rl" / "step.py").write_text("")

    includes = _auto_includes(tmp_path, script_path="src/nemotron/steps/sft/automodel/step.py")

    assert "src/nemotron/steps/index.py" in includes
    assert "src/nemotron/steps/sft" in includes
    assert "src/nemotron/steps/_runners" in includes
    assert "src/nemotron/steps/rl" not in includes
    assert "src/nemotron/recipes/nano3" not in includes


def test_auto_includes_ships_active_step_ancestor_helpers(tmp_path):
    _write_fake_repo(tmp_path)
    steps = tmp_path / "src" / "nemotron" / "steps"
    (steps / "data_prep").mkdir(parents=True)
    (steps / "data_prep" / "__init__.py").write_text("")
    (steps / "data_prep" / "_common.py").write_text("")
    (steps / "data_prep" / "sft_packing").mkdir()
    (steps / "data_prep" / "sft_packing" / "step.py").write_text("")
    (steps / "sft" / "automodel").mkdir(parents=True)
    (steps / "sft" / "automodel" / "step.py").write_text("")

    includes = _auto_includes(tmp_path, script_path="src/nemotron/steps/data_prep/sft_packing/step.py")

    assert "src/nemotron/steps/data_prep" in includes
    assert "src/nemotron/steps/sft" not in includes


def test_source_packager_ships_active_branch_support_paths(tmp_path):
    _write_fake_repo(tmp_path)
    steps = tmp_path / "src" / "nemotron" / "steps"
    byob = steps / "byob"
    (byob / "mcq").mkdir(parents=True)
    (byob / "legacy").mkdir()
    (byob / "__init__.py").write_text("from nemotron.steps.byob.adapter import x\n")
    (byob / "adapter.py").write_text("x = 1\n")
    (byob / "scripts").mkdir()
    (byob / "scripts" / "run.py").write_text("")
    (byob / "runtime").mkdir()
    (byob / "runtime" / "config.py").write_text("")
    (byob / "assets").mkdir()
    (byob / "assets" / "tiny.txt").write_text("")
    (byob / "mcq" / "step.py").write_text("")
    (byob / "legacy" / "step.py").write_text("")
    (steps / "sft" / "automodel").mkdir(parents=True)
    (steps / "sft" / "automodel" / "step.py").write_text("")

    pkg = SourcePackager(
        repo_root=str(tmp_path),
        script_path="src/nemotron/steps/byob/mcq/step.py",
    )
    out = pkg.package(None, str(tmp_path), "test")
    with tarfile.open(out) as tf:
        names = set(tf.getnames())

    assert "src/nemotron/steps/byob/__init__.py" in names
    assert "src/nemotron/steps/byob/adapter.py" in names
    assert "src/nemotron/steps/byob/scripts/run.py" in names
    assert "src/nemotron/steps/byob/runtime/config.py" in names
    assert "src/nemotron/steps/byob/assets/tiny.txt" in names
    assert "src/nemotron/steps/byob/mcq/step.py" in names
    assert "src/nemotron/steps/sft/automodel/step.py" not in names


def test_auto_includes_scopes_active_script_collection_without_step_names(tmp_path):
    _write_fake_repo(tmp_path)
    flows = tmp_path / "src" / "nemotron" / "workflows"
    benchmark = flows / "benchmark"
    (benchmark / "mcq").mkdir(parents=True)
    (benchmark / "legacy").mkdir()
    (flows / "other").mkdir()
    (benchmark / "__init__.py").write_text("")
    (benchmark / "adapter.py").write_text("x = 1\n")
    (benchmark / "assets").mkdir()
    (benchmark / "assets" / "tiny.txt").write_text("")
    (benchmark / "mcq" / "run.py").write_text("")
    (benchmark / "legacy" / "run.py").write_text("")
    (flows / "other" / "run.py").write_text("")

    includes = _auto_includes(tmp_path, script_path="src/nemotron/workflows/benchmark/mcq/run.py")

    assert "src/nemotron/workflows/benchmark" in includes
    assert "src/nemotron/workflows/other" not in includes


def test_auto_includes_raises_when_src_missing(tmp_path):
    with pytest.raises(ValueError, match="No src/"):
        _auto_includes(tmp_path, script_path=None)


# ── SourcePackager ──────────────────────────────────────────────────────────


def test_source_packager_filters_pycache_and_pyc(tmp_path):
    _write_fake_repo(tmp_path)
    artifacts = tmp_path / "src" / "nemotron" / "kit" / "artifacts"
    artifacts.mkdir()
    for name in (
        "records.parquet",
        "table.arrow",
        "weights.safetensors",
        "checkpoint.ckpt",
        "optimizer.pt",
        "tensor.npy",
        "index.idx",
        "model.onnx",
        "model.h5",
    ):
        (artifacts / name).write_text("large artifact")
    pkg = SourcePackager(
        repo_root=str(tmp_path),
        script_path="src/nemotron/recipes/nano3/x.py",
    )
    out = pkg.package(None, str(tmp_path), "test")
    assert out is not None and os.path.exists(out)
    with tarfile.open(out) as tf:
        names = tf.getnames()
    # Pyc + __pycache__ stripped.
    assert not any(n.endswith(".pyc") for n in names)
    assert not any("__pycache__" in n for n in names)
    assert not any("/artifacts/" in n for n in names)
    # Real package files present.
    assert any(n.endswith("src/nemo_runspec/__init__.py") for n in names)


def test_source_packager_warns_when_tarball_exceeds_limit(tmp_path, monkeypatch, capsys):
    _write_fake_repo(tmp_path)
    monkeypatch.setenv("NEMOTRON_SRC_TARBALL_WARN_BYTES", "1")
    pkg = SourcePackager(
        repo_root=str(tmp_path),
        script_path="src/nemotron/recipes/nano3/x.py",
    )

    pkg.package(None, str(tmp_path), "test")

    captured = capsys.readouterr()
    assert "source tarball is" in captured.err
    assert "NEMOTRON_SRC_TARBALL_WARN_BYTES=0" in captured.err


# ── plan_for ─────────────────────────────────────────────────────────────────


def test_plan_for_lepton_chunks_source_into_env_vars(tmp_path, monkeypatch):
    _write_fake_repo(tmp_path)
    # Skip the nemo-run patch side-effects — not relevant here.
    monkeypatch.setattr("nemo_runspec.run.patch_cloud_data_mover_skip_configs", lambda: None)
    monkeypatch.setattr(
        "nemo_runspec.run.patch_dgxcloud_strip_source_chunks_from_exports",
        lambda: None,
    )
    env_vars: dict[str, str] = {}
    plan = plan_for(
        executor_type="lepton",
        env_vars=env_vars,
        script_path="src/nemotron/recipes/nano3/x.py",
        pod_nemotron_home="/mnt/foo/_nemotron",
        repo_root=tmp_path,
    )
    assert isinstance(plan, Plan)
    assert plan.pod_src_root.startswith("/mnt/foo/_nemotron/src-")
    assert plan.source_ready_marker == f"{plan.pod_src_root}/.nemotron-src-ready"
    # Env vars populated with chunk count + at least one chunk.
    n = int(env_vars["_NEMOTRON_SRC_CHUNKS"])
    assert n >= 1
    assert "_NEMOTRON_SRC_CHUNK_0" in env_vars
    # NODE_RANK gate is present so multi-pod NFS runs don't race.
    script = plan.pre_script_cmds[0]
    assert "NODE_RANK" in script and "tar -xz" in script
    assert plan.source_ready_marker is not None
    assert plan.source_ready_marker in script
    assert ".nemotron-src-failed" in script
    assert 'while [ "$i" -lt 600 ]' in script
    assert "timed out waiting for" in script
    assert plan.needs_pwd_symlinks is True


def test_plan_for_cloud_ready_marker_is_unique_per_submission(tmp_path, monkeypatch):
    _write_fake_repo(tmp_path)
    monkeypatch.setattr("nemo_runspec.run.patch_cloud_data_mover_skip_configs", lambda: None)
    monkeypatch.setattr(
        "nemo_runspec.run.patch_dgxcloud_strip_source_chunks_from_exports",
        lambda: None,
    )

    env_a: dict[str, str] = {}
    env_b: dict[str, str] = {}
    plan_a = plan_for(
        executor_type="lepton",
        env_vars=env_a,
        script_path="src/nemotron/recipes/nano3/x.py",
        pod_nemotron_home="/mnt/foo/_nemotron",
        repo_root=tmp_path,
    )
    plan_b = plan_for(
        executor_type="lepton",
        env_vars=env_b,
        script_path="src/nemotron/recipes/nano3/x.py",
        pod_nemotron_home="/mnt/foo/_nemotron",
        repo_root=tmp_path,
    )

    assert env_a["_NEMOTRON_SRC_SHA256"] == env_b["_NEMOTRON_SRC_SHA256"]
    assert ".nemotron-src-ready-${_NEMOTRON_SRC_CHUNKS}" not in plan_a.pre_script_cmds[0]
    assert plan_a.source_ready_marker != plan_b.source_ready_marker
    assert plan_a.pod_src_root != plan_b.pod_src_root
    assert plan_a.pre_script_cmds[0] != plan_b.pre_script_cmds[0]


def test_plan_for_dgxcloud_chunks_source_into_env_vars(tmp_path, monkeypatch):
    """DGXCloud uses the same env-var chunking as Lepton now — env vars travel
    via ``environmentVariables`` in the Job spec (bypassing the 10 KiB Args cap)
    and a companion patch strips them from the ``torchrun_job.sh`` exports so
    the launcher file stays small."""
    _write_fake_repo(tmp_path)
    monkeypatch.setattr("nemo_runspec.run.patch_cloud_data_mover_skip_configs", lambda: None)
    monkeypatch.setattr(
        "nemo_runspec.run.patch_dgxcloud_strip_source_chunks_from_exports",
        lambda: None,
    )
    env_vars: dict[str, str] = {}
    plan = plan_for(
        executor_type="dgxcloud",
        env_vars=env_vars,
        script_path="src/nemotron/recipes/nano3/x.py",
        pod_nemotron_home="/workspace/_nemotron",
        repo_root=tmp_path,
    )
    assert plan.pod_src_root.startswith("/workspace/_nemotron/src-")
    # Env vars populated, no file-based PVC path.
    assert int(env_vars["_NEMOTRON_SRC_CHUNKS"]) >= 1
    assert "_NEMOTRON_SRC_CHUNK_0" in env_vars
    assert plan.needs_pwd_symlinks is True


def test_plan_for_fallback_uses_native_packager_path(tmp_path, monkeypatch):
    _write_fake_repo(tmp_path)
    monkeypatch.setattr("nemo_runspec.run.patch_cloud_data_mover_skip_configs", lambda: None)
    monkeypatch.setattr(
        "nemo_runspec.run.patch_dgxcloud_strip_source_chunks_from_exports",
        lambda: None,
    )
    plan = plan_for(
        executor_type="slurm",
        env_vars={},
        script_path=None,
        pod_nemotron_home="/unused",
        repo_root=tmp_path,
    )
    # Native extraction path; caller must symlink under $PWD/src.
    assert plan.pod_src_root == "/nemo_run/code/src"
    assert plan.needs_pwd_symlinks is True
    assert plan.pre_script_cmds == []
