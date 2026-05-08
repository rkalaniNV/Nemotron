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

"""Airgap lockfile compiler for ``src/nemotron/steps``.

The compiler is intentionally static and conservative. It turns one
``step.py`` + ``step.toml`` + selected config YAML into a portable manifest of
the runtime, assets, service endpoints, and manual inputs that need to be made
available before a customer runs the step in a disconnected environment.

It does not execute step code and it does not need heavyweight training
frameworks installed. That keeps ``nemotron step airgap lock`` usable from a
plain development checkout.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import sys
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import yaml
from omegaconf import OmegaConf

from nemo_runspec import Runspec
from nemo_runspec import parse as parse_runspec
from nemo_runspec.config import apply_dotlist_overrides, find_config_file, load_config
from nemo_runspec.env import find_env_file, load_env_profile
from nemotron.steps.index import StepInfo, discover_steps

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore[no-redef]


AIRGAP_SCHEMA_VERSION = "1"
AIRGAP_UV_VERSION = "0.11.1"
AIRGAP_RUNTIME_DIR = "runtime"
AIRGAP_ASSETS_DIR = "assets"
AIRGAP_CONTAINER_ROOT = "/opt/nemotron-airgap"
AIRGAP_CONTAINER_WHEELHOUSE = f"{AIRGAP_CONTAINER_ROOT}/wheels"
AIRGAP_CONTAINER_ASSET_ROOT = f"{AIRGAP_CONTAINER_ROOT}/assets"
AIRGAP_CONTAINER_HF_HOME = f"{AIRGAP_CONTAINER_ASSET_ROOT}/hf-cache"
AIRGAP_CONTAINER_HF_CACHE = f"{AIRGAP_CONTAINER_HF_HOME}/hub"
AIRGAP_CONTAINER_REPOS = f"{AIRGAP_CONTAINER_ASSET_ROOT}/repos"
LEPTON_INIT_SCRIPT_URL = "https://raw.githubusercontent.com/leptonai/scripts/main/lepton_env_to_pytorch.sh"
LEPTON_INIT_SCRIPT_RELATIVE_PATH = "lepton/lepton_env_to_pytorch.sh"
LEPTON_INIT_SCRIPT_BUNDLE_PATH = f"{AIRGAP_ASSETS_DIR}/{LEPTON_INIT_SCRIPT_RELATIVE_PATH}"
LEPTON_INIT_SCRIPT_CONTAINER_PATH = f"{AIRGAP_CONTAINER_ASSET_ROOT}/{LEPTON_INIT_SCRIPT_RELATIVE_PATH}"
REMOTE_EXECUTORS = {"slurm", "lepton", "dgxcloud"}
PIP_INSTALL_MODES = {
    "none",
    "offline_wheelhouse",
    "online",
    "online_best_effort",
    "preinstalled",
}
ONLINE_STARTUP_RE = re.compile(
    r"\b("
    r"pip\s+install|"
    r"uv\s+pip\s+install|"
    r"wget\s+https?://|"
    r"curl\b.*https?://|"
    r"git\s+clone\s+https?://"
    r")\b",
    re.IGNORECASE,
)

OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "PIP_NO_INDEX": "1",
    "PIP_FIND_LINKS": AIRGAP_CONTAINER_WHEELHOUSE,
    "UV_FIND_LINKS": AIRGAP_CONTAINER_WHEELHOUSE,
    "UV_NO_INDEX": "1",
    "UV_OFFLINE": "1",
    "UV_LINK_MODE": "copy",
    "UV_NO_SYNC": "1",
    "UV_PYTHON_DOWNLOADS": "never",
    "WANDB_MODE": "offline",
    "HF_HOME": AIRGAP_CONTAINER_HF_HOME,
    "HF_HUB_CACHE": AIRGAP_CONTAINER_HF_CACHE,
    "NEMO_RUNSPEC_OFFLINE": "1",
    "NEMO_RUNSPEC_REPO_CACHE": AIRGAP_CONTAINER_REPOS,
    "NEMO_RUNSPEC_LEPTON_INIT_DEFAULT_SCRIPT": LEPTON_INIT_SCRIPT_CONTAINER_PATH,
    "NEMOTRON_AIRGAP_ASSETS": AIRGAP_CONTAINER_ASSET_ROOT,
    "NEMOTRON_AIRGAP_REPOS": AIRGAP_CONTAINER_REPOS,
}

LOCALHOST_NAMES = {
    "0.0.0.0",
    "127.0.0.1",
    "localhost",
    "nim-llm",
}

HF_ID_KEYS = {
    "pretrained_model_name_or_path",
    "hf_model_path",
    "hf_model_id",
    "hf_model_name_or_path",
    "teacher_hf_path",
    "student_hf_path",
    "student_hf_model",
    "model_name",
    "model",
    "name",
    "path_or_dataset_id",
    "repo_id",
}

TOKENIZER_KEYS = {
    "tokenizer",
    "tokenizer_path",
    "tokenizer_name",
    "tokenizer_model",
}

PATH_KEYS = {
    "path",
    "paths",
    "blend_path",
    "input_glob",
    "output_dir",
    "output_path",
    "checkpoint_dir",
    "checkpoint_path",
    "save",
    "load",
    "pretrained_checkpoint",
    "hf_export_path",
    "output_hf_path",
    "megatron_save_path",
    "nano3_packed_sft_dir",
    "data_paths",
    "prompts",
}

SERVICE_KEYS = {
    "url",
    "address",
    "endpoint",
    "base_url",
    "base_urls",
}

HF_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]+/[A-Za-z0-9][A-Za-z0-9_.\-/]+$")
URL_RE = re.compile(r"^https?://", re.IGNORECASE)
GIT_REF_RE = re.compile(r"^git\+(?P<url>.+)@(?P<ref>[^@]+)$")
AUTO_MOUNT_RE = re.compile(r"\$\{auto_mount:(git\+[^,}]+)(?:,([^}]+))?\}")
OC_ENV_RE = re.compile(r"\$\{oc\.env:([^,}]+)(?:,([^}]*))?\}")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)
DOCKER_DIGEST_RE = re.compile(r"@sha256:[0-9a-f]{64}$", re.IGNORECASE)
PINNED_HF_REVISIONS = {"main", "master", "latest", ""}
IMPLICIT_ENV_VARS = {"PWD", "HOME", "USER"}
LOCAL_IMPORT_ROOTS = {"nemotron", "nemo_runspec"}
COMMON_TASK_IMPORTS = {
    "prep": ["cosmos_xenna", "cattrs", "obstore", "portpicker"],
    "curate": ["cosmos_xenna"],
    "pretrain": ["typer", "rich", "pydantic_settings", "omegaconf"],
    "sft": ["typer", "rich", "pydantic_settings", "omegaconf"],
    "peft": ["typer", "rich", "pydantic_settings", "omegaconf"],
    "rl": ["typer", "rich", "pydantic_settings", "omegaconf"],
    "eval": ["typer", "rich", "pydantic_settings", "omegaconf"],
}
PACKAGE_IMPORT_ALIASES = {
    "cosmos-xenna": "cosmos_xenna",
    "pydantic-settings": "pydantic_settings",
    "huggingface-hub": "huggingface_hub",
    "pyyaml": "yaml",
}


AssetKind = Literal[
    "hf_model",
    "hf_dataset",
    "git_repo",
    "docker_image",
    "python_git",
    "url",
    "local_path",
    "service",
    "env",
]
AssetDelivery = Literal["runtime", "external", "manual"]


@dataclass(frozen=True)
class AirgapAsset:
    """One external or local thing needed by an offline run."""

    kind: AssetKind
    id: str
    source: str
    revision: str | None = None
    repo_type: str | None = None
    target: str | None = None
    field: str | None = None
    required: bool = True
    note: str | None = None
    delivery: AssetDelivery | None = None
    bundle_path: str | None = None
    # Content hash for assets where we can compute one (currently URL payloads).
    # Recorded after fetch and re-checked by verify_lock(bundle_dir=...) to
    # catch tampered or corrupted transfers.
    sha256: str | None = None
    # Optional caller-supplied expectation; if set, fetch refuses to write a
    # blob whose sha256 does not match.
    expected_sha256: str | None = None


@dataclass(frozen=True)
class AirgapIssue:
    """A warning or failure candidate in the compiled lock."""

    severity: Literal["info", "warning", "error"]
    code: str
    message: str
    source: str | None = None


@dataclass(frozen=True)
class FileFingerprint:
    """Small provenance hash for source files that influenced the lock."""

    path: str
    sha256: str


@dataclass(frozen=True)
class AirgapLock:
    """Serializable airgap lockfile."""

    schema_version: str
    generated_at: str
    step: dict[str, Any]
    executors: list[dict[str, Any]]
    runspec: dict[str, Any]
    config: dict[str, Any]
    runtime: dict[str, Any]
    assets: list[dict[str, Any]]
    services: list[dict[str, Any]]
    manual_inputs: list[dict[str, Any]]
    unresolved_env: list[dict[str, Any]]
    issues: list[dict[str, Any]]
    delivery_plan: dict[str, Any]
    provenance: dict[str, Any]
    offline_env: dict[str, str] = field(default_factory=lambda: dict(OFFLINE_ENV))


@dataclass(frozen=True)
class AirgapTarget:
    """One step/config target inside a workflow airgap lock."""

    step_id: str
    config_name: str | None = None
    overrides: tuple[str, ...] = ()


class AirgapCompiler:
    """Compile a step/config pair into an :class:`AirgapLock`."""

    def __init__(self, *, repo_root: str | Path | None = None) -> None:
        self.repo_root = Path(repo_root or Path.cwd()).resolve()

    def compile(
        self,
        *,
        step_id: str,
        config_name: str | None = None,
        overrides: Iterable[str] = (),
        profiles: Iterable[str] = (),
        env_file: str | Path | None = None,
    ) -> AirgapLock:
        step = _resolve_step(step_id)
        step_py = step.path / "step.py"
        spec = parse_runspec(step_py)

        selected_config = config_name or spec.config.default
        config_path = find_config_file(selected_config, spec.config_dir)
        config = apply_dotlist_overrides(load_config(config_path), list(overrides))
        config_data = OmegaConf.to_container(config, resolve=False)
        if not isinstance(config_data, dict):
            raise TypeError(f"{config_path}: top-level config must be a mapping")

        manifest = _load_step_manifest(step.path / "step.toml")
        airgap_hints = _load_step_airgap(step.path)
        manifest = _merge_step_airgap(manifest, airgap_hints)
        assets: list[AirgapAsset] = []
        services: list[AirgapAsset] = []
        manual_inputs: list[AirgapAsset] = []
        unresolved_env: list[AirgapAsset] = []
        issues: list[AirgapIssue] = []

        _add_runtime_assets(spec=spec, config_data=config_data, assets=assets, issues=issues)
        _add_pyproject_assets(self.repo_root, assets=assets, issues=issues)
        _scan_manifest_airgap(manifest, assets=assets, services=services, issues=issues)
        _scan_airgap_config_fields(
            manifest,
            config_data,
            assets=assets,
            services=services,
            manual_inputs=manual_inputs,
            issues=issues,
        )
        _scan_training_smoke_config(config_data, issues=issues)

        scanner = _ConfigScanner(
            repo_root=self.repo_root,
            config_path=config_path,
            assets=assets,
            services=services,
            manual_inputs=manual_inputs,
            unresolved_env=unresolved_env,
            issues=issues,
        )
        scanner.scan(config_data)

        self._scan_blends(config_data, config_path=config_path, scanner=scanner)

        assets = _prepare_assets_for_lock(_dedupe_assets(assets))
        services = _prepare_assets_for_lock(_dedupe_assets(services))
        manual_inputs = _prepare_assets_for_lock(_dedupe_assets(manual_inputs))
        unresolved_env = _prepare_assets_for_lock(_dedupe_assets(unresolved_env))
        issues = _dedupe_issues(issues)

        config_hash = hashlib.sha256(
            yaml.safe_dump(config_data, sort_keys=True).encode("utf-8")
        ).hexdigest()

        executor_data = _compile_executor_profiles(
            self.repo_root,
            profile_names=profiles,
            env_file=Path(env_file) if env_file is not None else None,
        )

        assets = _prepare_assets_for_lock(_dedupe_assets([*assets, *executor_data["assets"]]))
        services = _prepare_assets_for_lock(_dedupe_assets([*services, *executor_data["services"]]))
        manual_inputs = _prepare_assets_for_lock(_dedupe_assets([*manual_inputs, *executor_data["manual_inputs"]]))
        unresolved_env = _prepare_assets_for_lock(_dedupe_assets([*unresolved_env, *executor_data["unresolved_env"]]))
        issues = _dedupe_issues([*issues, *executor_data["issues"]])
        provenance_files = _provenance_files(self.repo_root, step.path, config_path)
        provenance_files.extend(executor_data["provenance_files"])

        runtime = {
            "base_images": [asdict(a) for a in assets if a.kind == "docker_image"],
            "python": _python_runtime(self.repo_root, spec=spec, manifest=manifest, issues=issues),
            "task_containers": _task_container_readiness(
                self.repo_root,
                step=step,
                spec=spec,
                manifest=manifest,
                config_data=config_data,
                base_images=[a for a in assets if a.kind == "docker_image"],
            ),
            "bundle_layout": _bundle_layout(),
        }
        # ``_python_runtime`` may have appended new issues; re-dedupe to keep
        # the lock summary stable.
        issues = _dedupe_issues(issues)
        asset_dicts = [asdict(a) for a in assets if a.kind != "docker_image"]
        service_dicts = [asdict(a) for a in services]
        manual_input_dicts = [asdict(a) for a in manual_inputs]
        unresolved_env_dicts = [asdict(a) for a in unresolved_env]

        return AirgapLock(
            schema_version=AIRGAP_SCHEMA_VERSION,
            generated_at=datetime.now(timezone.utc).isoformat(),
            step={
                "id": step.id,
                "name": step.name,
                "category": step.category,
                "path": _relpath(step.path, self.repo_root),
            },
            executors=executor_data["executors"],
            runspec=_runspec_to_dict(spec, self.repo_root),
            config={
                "name": selected_config,
                "path": _relpath(config_path, self.repo_root),
                "sha256": config_hash,
                "overrides": list(overrides),
            },
            runtime=runtime,
            assets=asset_dicts,
            services=service_dicts,
            manual_inputs=manual_input_dicts,
            unresolved_env=unresolved_env_dicts,
            issues=[asdict(i) for i in issues],
            delivery_plan=build_delivery_plan(
                {
                    "runtime": runtime,
                    "assets": asset_dicts,
                    "services": service_dicts,
                    "manual_inputs": manual_input_dicts,
                    "unresolved_env": unresolved_env_dicts,
                }
            ),
            provenance={
                "repo_root": str(self.repo_root),
                "files": [asdict(fp) for fp in _dedupe_fingerprints(provenance_files)],
            },
        )

    def compile_many(
        self,
        targets: Iterable[AirgapTarget],
        *,
        workflow_name: str = "workflow",
        profiles: Iterable[str] = (),
        env_file: str | Path | None = None,
    ) -> dict[str, Any]:
        """Compile several step/config targets into one workflow lock."""

        target_list = list(targets)
        if not target_list:
            raise ValueError("At least one airgap target is required")
        locks = [
            lock_to_dict(
                self.compile(
                    step_id=target.step_id,
                    config_name=target.config_name,
                    overrides=target.overrides,
                )
            )
            for target in target_list
        ]
        merged = merge_locks(locks, workflow_name=workflow_name, repo_root=self.repo_root)
        return add_executor_profiles(
            merged,
            repo_root=self.repo_root,
            profiles=profiles,
            env_file=Path(env_file) if env_file is not None else None,
        )

    def _scan_blends(self, config_data: dict[str, Any], *, config_path: Path, scanner: _ConfigScanner) -> None:
        for path_value in _values_for_key(config_data, "blend_path"):
            if not isinstance(path_value, str):
                continue
            for expanded in _expand_oc_env_defaults(path_value):
                if expanded != path_value:
                    scanner.scan_value(expanded, path=("blend_path",))
            blend_path = _resolve_local_path(path_value, base_dir=config_path.parent, repo_root=self.repo_root)
            if blend_path is None or not blend_path.exists() or not blend_path.is_file():
                continue
            try:
                data = json.loads(blend_path.read_text(encoding="utf-8"))
            except Exception as exc:
                scanner.issues.append(
                    AirgapIssue(
                        severity="warning",
                        code="blend_unreadable",
                        message=f"Could not parse data blend {blend_path}: {type(exc).__name__}: {exc}",
                        source=str(blend_path),
                    )
                )
                continue
            scanner.scan(data, path=("blend", _relpath(blend_path, self.repo_root)))


class _ConfigScanner:
    """Best-effort scanner for known step config conventions."""

    def __init__(
        self,
        *,
        repo_root: Path,
        config_path: Path,
        assets: list[AirgapAsset],
        services: list[AirgapAsset],
        manual_inputs: list[AirgapAsset],
        unresolved_env: list[AirgapAsset],
        issues: list[AirgapIssue],
    ) -> None:
        self.repo_root = repo_root
        self.config_path = config_path
        self.assets = assets
        self.services = services
        self.manual_inputs = manual_inputs
        self.unresolved_env = unresolved_env
        self.issues = issues

    def scan(self, value: Any, path: tuple[str, ...] = ()) -> None:
        if isinstance(value, Mapping):
            skip_keys = self._scan_mapping(value, path=path)
            for key, child in value.items():
                if str(key) in skip_keys:
                    continue
                self.scan(child, (*path, str(key)))
            return
        if isinstance(value, list):
            for index, child in enumerate(value):
                self.scan(child, (*path, str(index)))
            return
        if isinstance(value, str):
            self.scan_value(value, path=path)

    def _scan_mapping(self, value: Mapping[str, Any], *, path: tuple[str, ...]) -> set[str]:
        skip_keys: set[str] = set()
        provider = value.get("provider")
        if isinstance(provider, str):
            field = ".".join((*path, "provider")) if path else "provider"
            provider_normalized = provider.lower()
            if provider_normalized in {"nvidia", "openai"}:
                self.services.append(
                    AirgapAsset(
                        kind="service",
                        id=f"{provider_normalized}-provider",
                        source=field,
                        field=field,
                        note=(
                            "Provider-backed generation must be replaced with an in-network endpoint "
                            "for customer airgap."
                        ),
                    )
                )
                if provider_normalized == "nvidia":
                    self.issues.append(
                        AirgapIssue(
                            severity="warning",
                            code="external_provider",
                            message=(
                                "Data generation config uses the NVIDIA cloud provider; airgap delivery "
                                "needs a local endpoint override."
                            ),
                            source=field,
                        )
                    )

        repo_id = _maybe_str(value.get("repo_id") or value.get("path_or_dataset_id"))
        if repo_id and _looks_like_hf_id(repo_id):
            repo_key = "repo_id" if value.get("repo_id") else "path_or_dataset_id"
            repo_type = _maybe_str(value.get("repo_type")) or ("dataset" if _is_dataset_context(path) else "model")
            field = ".".join((*path, repo_key)) if path else repo_key
            self.assets.append(
                AirgapAsset(
                    kind="hf_dataset" if repo_type == "dataset" else "hf_model",
                    id=repo_id,
                    source=field,
                    revision=_maybe_str(value.get("revision")),
                    repo_type=repo_type,
                    field=field,
                    target=_maybe_str(value.get("local_dir")),
                    delivery="external",
                )
            )
            skip_keys.add(repo_key)

        return skip_keys

    def scan_value(self, value: str, *, path: tuple[str, ...]) -> None:
        field = ".".join(path)
        key = path[-1] if path else ""

        for env_name, default in _oc_env_defaults(value):
            if env_name in IMPLICIT_ENV_VARS:
                continue
            if default is None:
                self.unresolved_env.append(
                    AirgapAsset(
                        kind="env",
                        id=env_name,
                        source=field,
                        field=field,
                        note="Required environment variable has no config default.",
                    )
                )
            elif default:
                self.scan_value(default, path=path)

        for spec, target in _auto_mounts(value):
            git = _parse_git_spec(spec)
            if git is not None:
                url, ref = git
                self.assets.append(
                    AirgapAsset(
                        kind="git_repo",
                        id=_repo_name(url),
                        source=field,
                        revision=ref,
                        target=target,
                        note=url,
                    )
                )
                _warn_if_git_ref_floating(ref, field, self.issues)

        git = _parse_git_spec(value)
        if git is not None:
            url, ref = git
            self.assets.append(
                AirgapAsset(
                    kind="git_repo",
                    id=_repo_name(url),
                    source=field,
                    revision=ref,
                    note=url,
                )
            )
            _warn_if_git_ref_floating(ref, field, self.issues)
            return

        if URL_RE.match(value):
            kind = "service" if key in SERVICE_KEYS or _is_local_service_url(value) else "url"
            target = self.services if kind == "service" else self.assets
            target.append(
                AirgapAsset(
                    kind=kind,  # type: ignore[arg-type]
                    id=value,
                    source=field,
                    field=field,
                    note="Local endpoint must be reachable inside the airgapped environment."
                    if kind == "service"
                    else None,
                )
            )
            return

        if value.startswith("hf://"):
            repo_id = value.removeprefix("hf://").strip("/")
            if repo_id:
                self.assets.append(
                    AirgapAsset(
                        kind="hf_dataset",
                        id=repo_id,
                        source=field,
                        repo_type="dataset",
                        field=field,
                    )
                )
            return

        if _looks_like_path(value):
            self._record_path(value, field=field, key=key)
            return

        if _looks_like_hf_id(value) and _is_hf_context(path):
            repo_type = "dataset" if _is_dataset_context(path) else "model"
            self.assets.append(
                AirgapAsset(
                    kind="hf_dataset" if repo_type == "dataset" else "hf_model",
                    id=value,
                    source=field,
                    repo_type=repo_type,
                    field=field,
                )
            )
            return

    def _record_path(self, value: str, *, field: str, key: str) -> None:
        if value.startswith(("/path/to/", "./path/to/")):
            self.manual_inputs.append(
                AirgapAsset(
                    kind="local_path",
                    id=value,
                    source=field,
                    field=field,
                    note="Placeholder path must be replaced with a real local or mounted path.",
                )
            )
            self.issues.append(
                AirgapIssue(
                    severity="warning",
                    code="placeholder_path",
                    message=f"{field} uses placeholder path {value!r}.",
                    source=field,
                )
            )
            return

        if (
            key not in PATH_KEYS
            and key not in TOKENIZER_KEYS
            and not any(part in PATH_KEYS for part in field.split("."))
        ):
            return

        resolved = _resolve_local_path(value, base_dir=self.config_path.parent, repo_root=self.repo_root)
        note = None
        if resolved is not None and resolved.exists():
            asset_id = _relpath(resolved, self.repo_root)
        else:
            asset_id = value
            note = "Path must be mounted or bundled for offline execution."
        self.manual_inputs.append(
            AirgapAsset(
                kind="local_path",
                id=asset_id,
                source=field,
                field=field,
                note=note,
            )
        )


def lock_to_dict(lock: AirgapLock | Mapping[str, Any]) -> dict[str, Any]:
    """Convert an :class:`AirgapLock` to a plain mapping for YAML/JSON."""

    if isinstance(lock, Mapping):
        return dict(lock)
    return asdict(lock)


def write_lock(lock: AirgapLock | Mapping[str, Any], path: str | Path) -> Path:
    """Write a lockfile as deterministic YAML."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(lock_to_dict(lock), sort_keys=False), encoding="utf-8")
    return out


def merge_locks(
    locks: Iterable[Mapping[str, Any]], *, workflow_name: str = "workflow", repo_root: Path | None = None
) -> dict[str, Any]:
    """Merge several single-step locks into one workflow-level lock."""

    lock_list = [dict(lock) for lock in locks]
    if not lock_list:
        raise ValueError("At least one lock is required")

    assets = _merge_asset_dicts(_workflow_items(lock_list, "assets", prefix_source=True))
    services = _merge_asset_dicts(_workflow_items(lock_list, "services", prefix_source=True))
    manual_inputs = _merge_asset_dicts(_workflow_items(lock_list, "manual_inputs", prefix_source=True))
    unresolved_env = _merge_asset_dicts(_workflow_items(lock_list, "unresolved_env", prefix_source=True))
    base_images = _merge_asset_dicts(_workflow_base_images(lock_list))
    issues = _merge_issue_dicts(_workflow_items(lock_list, "issues", prefix_source=True))

    if len({image.get("id") for image in base_images if image.get("id")}) > 1:
        issues.append(
            asdict(
                AirgapIssue(
                    severity="warning",
                    code="multiple_base_images",
                    message=(
                        "Workflow targets reference multiple base images. The checked-in Dockerfile "
                        "uses the first image unless you override BASE_IMAGE or build per-runtime images."
                    ),
                    source="workflow",
                )
            )
        )

    steps = [
        {
            "step": lock.get("step", {}),
            "executors": lock.get("executors", []) or [],
            "config": lock.get("config", {}),
            "runspec": lock.get("runspec", {}),
        }
        for lock in lock_list
    ]
    config_hash = hashlib.sha256(yaml.safe_dump(steps, sort_keys=True).encode("utf-8")).hexdigest()

    data = {
        "schema_version": AIRGAP_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "kind": "workflow",
        "step": {
            "id": workflow_name,
            "name": workflow_name,
            "category": "workflow",
            "path": None,
        },
        "executors": _merge_executor_dicts(_workflow_items(lock_list, "executors")),
        "steps": steps,
        "runspec": {
            "name": workflow_name,
            "image": base_images[0].get("id") if base_images else None,
            "launch": None,
            "cmd": None,
            "workdir": None,
            "config_dir": None,
            "default_config": None,
            "resources": {},
        },
        "config": {
            "name": workflow_name,
            "path": None,
            "sha256": config_hash,
            "overrides": [],
        },
        "runtime": {
            "base_images": base_images,
            "python": _merge_python_runtime(lock_list),
            "task_containers": _merge_task_containers(lock_list),
            "bundle_layout": _bundle_layout(),
        },
        "assets": assets,
        "services": services,
        "manual_inputs": manual_inputs,
        "unresolved_env": unresolved_env,
        "issues": _merge_issue_dicts(issues),
        "provenance": _merge_provenance(lock_list, repo_root=repo_root),
        "offline_env": dict(OFFLINE_ENV),
    }
    data["delivery_plan"] = build_delivery_plan(data)
    return data


def add_executor_profiles(
    lock: Mapping[str, Any],
    *,
    repo_root: Path,
    profiles: Iterable[str] = (),
    env_file: Path | None = None,
) -> dict[str, Any]:
    """Attach executor profile metadata and closure checks to an existing lock."""

    profile_names = [profile for profile in profiles if profile]
    if not profile_names:
        return dict(lock)
    data = dict(lock)
    executor_data = _compile_executor_profiles(repo_root, profile_names=profile_names, env_file=env_file)

    data["executors"] = _merge_executor_dicts(
        [*(data.get("executors", []) or []), *executor_data["executors"]]
    )
    data["assets"] = _merge_asset_dicts([*(data.get("assets", []) or []), *executor_data["assets"]])
    data["services"] = _merge_asset_dicts([*(data.get("services", []) or []), *executor_data["services"]])
    data["manual_inputs"] = _merge_asset_dicts(
        [*(data.get("manual_inputs", []) or []), *executor_data["manual_inputs"]]
    )
    data["unresolved_env"] = _merge_asset_dicts(
        [*(data.get("unresolved_env", []) or []), *executor_data["unresolved_env"]]
    )
    data["issues"] = _merge_issue_dicts([*(data.get("issues", []) or []), *executor_data["issues"]])
    data["delivery_plan"] = build_delivery_plan(data)
    provenance = dict(data.get("provenance", {}) or {})
    provenance["repo_root"] = provenance.get("repo_root") or str(repo_root)
    provenance["files"] = [
        asdict(fp)
        for fp in _dedupe_fingerprints(
            [
                *[
                    FileFingerprint(path=str(item.get("path")), sha256=str(item.get("sha256")))
                    for item in provenance.get("files", []) or []
                    if isinstance(item, Mapping) and item.get("path") and item.get("sha256")
                ],
                *executor_data["provenance_files"],
            ]
        )
    ]
    data["provenance"] = provenance
    return data


def _compile_executor_profiles(
    repo_root: Path,
    *,
    profile_names: Iterable[str],
    env_file: Path | None = None,
) -> dict[str, Any]:
    profile_list = [name for name in profile_names if name]
    if not profile_list:
        return _empty_executor_data()

    env_path = env_file.expanduser().resolve() if env_file is not None else find_env_file(repo_root)
    if env_path is None:
        raise ValueError("Executor profiles were requested, but no env.toml was found")

    executors: list[dict[str, Any]] = []
    assets: list[AirgapAsset] = []
    services: list[AirgapAsset] = []
    manual_inputs: list[AirgapAsset] = []
    unresolved_env: list[AirgapAsset] = []
    issues: list[AirgapIssue] = []

    manual_inputs.append(
        AirgapAsset(
            kind="local_path",
            id=_relpath(env_path, repo_root),
            source="executor.env_file",
            field="env.toml",
            note="Executor profile file must be supplied in the customer runtime environment.",
        )
    )

    for profile_name in profile_list:
        profile = load_env_profile(profile_name, config_path=env_path)
        profile_data = OmegaConf.to_container(profile, resolve=True)
        if not isinstance(profile_data, Mapping):
            raise TypeError(f"{env_path}: executor profile {profile_name!r} must resolve to a mapping")
        executors.append(_executor_summary(profile_name, profile_data, env_path=env_path, repo_root=repo_root))
        _scan_executor_profile(
            profile_name,
            profile_data,
            assets=assets,
            services=services,
            manual_inputs=manual_inputs,
            unresolved_env=unresolved_env,
            issues=issues,
        )

    return {
        "executors": executors,
        "services": _dedupe_assets(services),
        "manual_inputs": _dedupe_assets(manual_inputs),
        "unresolved_env": _dedupe_assets(unresolved_env),
        "issues": _dedupe_issues(issues),
        "assets": _dedupe_assets(assets),
        "provenance_files": [_fingerprint(env_path, repo_root)] if env_path.exists() else [],
    }


def _empty_executor_data() -> dict[str, Any]:
    return {
        "executors": [],
        "services": [],
        "manual_inputs": [],
        "unresolved_env": [],
        "issues": [],
        "assets": [],
        "provenance_files": [],
    }


def _executor_summary(
    profile_name: str, profile: Mapping[str, Any], *, env_path: Path, repo_root: Path
) -> dict[str, Any]:
    return {
        "profile": profile_name,
        "executor": str(profile.get("executor") or "local"),
        "env_file": _relpath(env_path, repo_root),
        "env_file_sha256": _sha256(env_path) if env_path.exists() else None,
        "fields": sorted(str(key) for key in profile.keys()),
        "note": "Values are intentionally not copied here to avoid leaking site-specific credentials.",
    }


def _scan_executor_profile(
    profile_name: str,
    profile: Mapping[str, Any],
    *,
    assets: list[AirgapAsset],
    services: list[AirgapAsset],
    manual_inputs: list[AirgapAsset],
    unresolved_env: list[AirgapAsset],
    issues: list[AirgapIssue],
) -> None:
    executor = str(profile.get("executor") or "local").lower()
    source = f"executor:{profile_name}"

    if executor in REMOTE_EXECUTORS:
        issues.append(
            AirgapIssue(
                severity="info",
                code="executor_network",
                message=(
                    f"Executor profile {profile_name!r} uses {executor!r}; its control plane, "
                    "registry, shared filesystem, and service endpoints must be reachable inside "
                    "the customer network or VPN."
                ),
                source=source,
            )
        )
        _scan_remote_pip_policy(profile_name, profile, issues=issues)
        _scan_remote_startup_commands(profile_name, profile, issues=issues)

    if executor == "slurm" and isinstance(profile.get("host"), str):
        services.append(
            AirgapAsset(
                kind="service",
                id=f"ssh://{profile['host']}",
                source=f"{source}.host",
                field="host",
                note="Slurm login host must be reachable from the submit environment, usually over VPN.",
            )
        )
    elif executor in {"lepton", "dgxcloud"}:
        services.append(
            AirgapAsset(
                kind="service",
                id=f"{executor}:{profile_name}",
                source=source,
                field="executor",
                note="Cloud executor API endpoint must be reachable from the submit environment.",
            )
        )
        if executor == "lepton":
            env_vars = profile.get("env_vars") if isinstance(profile.get("env_vars"), Mapping) else {}
            init_mode = str(
                env_vars.get("NEMO_RUNSPEC_LEPTON_INIT_MODE")
                or env_vars.get("NEMOTRON_LEPTON_INIT_MODE")
                or ""
            )
            init_script = str(
                env_vars.get("NEMO_RUNSPEC_LEPTON_INIT_SCRIPT")
                or env_vars.get("NEMOTRON_LEPTON_INIT_SCRIPT")
                or ""
            )
            if init_mode != "skip":
                assets.append(
                    AirgapAsset(
                        kind="url",
                        id=LEPTON_INIT_SCRIPT_URL,
                        source=f"{source}.env_vars.NEMO_RUNSPEC_LEPTON_INIT_SCRIPT",
                        field="NEMO_RUNSPEC_LEPTON_INIT_SCRIPT",
                        note=(
                            "Lepton distributed launcher init script. Fetch during connected prep, "
                            f"stage it as {LEPTON_INIT_SCRIPT_RELATIVE_PATH} under the remote asset root, "
                            "then set NEMO_RUNSPEC_LEPTON_INIT_SCRIPT to that executor-visible path."
                        ),
                        delivery="external",
                        bundle_path=LEPTON_INIT_SCRIPT_BUNDLE_PATH,
                    )
                )
            if init_mode != "skip" and not init_script:
                issues.append(
                    AirgapIssue(
                        severity="warning",
                        code="lepton_init_script_unset",
                        message=(
                            f"Lepton profile {profile_name!r} does not set NEMO_RUNSPEC_LEPTON_INIT_SCRIPT "
                            "or NEMO_RUNSPEC_LEPTON_INIT_MODE=skip. The lock includes the Lepton init script "
                            f"as {LEPTON_INIT_SCRIPT_BUNDLE_PATH}; stage it on executor-visible storage "
                            "and set NEMO_RUNSPEC_LEPTON_INIT_SCRIPT to that staged path."
                        ),
                        source=f"{source}.env_vars",
                    )
                )

    for profile_field in ("remote_job_dir", "build_cache_dir", "sif_dir", "squash_dir"):
        value = profile.get(profile_field)
        if isinstance(value, str) and value:
            manual_inputs.append(
                AirgapAsset(
                    kind="local_path",
                    id=value,
                    source=f"{source}.{profile_field}",
                    field=profile_field,
                    note="Path must exist on the executor-side filesystem.",
                )
            )

    for mount in profile.get("mounts", []) or []:
        _record_executor_mount(profile_name, mount, manual_inputs=manual_inputs)
    for mount in profile.get("container_mounts", []) or []:
        _record_executor_mount(profile_name, mount, manual_inputs=manual_inputs)

    for path, value in _walk_mapping(profile):
        if isinstance(value, str):
            for env_name, default in _oc_env_defaults(value):
                if env_name in IMPLICIT_ENV_VARS:
                    continue
                if default is None:
                    unresolved_env.append(
                        AirgapAsset(
                            kind="env",
                            id=env_name,
                            source=f"{source}.{path}",
                            field=path,
                            note="Executor profile references an environment variable with no default.",
                        )
                    )
        if isinstance(value, str) and path.split(".")[-1] in SERVICE_KEYS and URL_RE.match(value):
            services.append(
                AirgapAsset(
                    kind="service",
                    id=value,
                    source=f"{source}.{path}",
                    field=path,
                    note="Executor service endpoint must be reachable inside the customer network.",
                )
            )


def _scan_remote_pip_policy(
    profile_name: str,
    profile: Mapping[str, Any],
    *,
    issues: list[AirgapIssue],
) -> None:
    pip_extras = profile.get("pip_extras") or []
    if not pip_extras:
        return

    source = f"executor:{profile_name}.pip_extras"
    mode = str(profile.get("pip_install_mode") or "").lower()
    if not mode:
        issues.append(
            AirgapIssue(
                severity="warning",
                code="remote_pip_extras_implicit",
                message=(
                    f"Executor profile {profile_name!r} has pip_extras but no pip_install_mode. "
                    "Implicit online best-effort pip installs are not airgap-ready; use "
                    "preinstalled, offline_wheelhouse, online, online_best_effort, or none."
                ),
                source=source,
            )
        )
        return

    if mode not in PIP_INSTALL_MODES:
        issues.append(
            AirgapIssue(
                severity="warning",
                code="remote_pip_install_mode_invalid",
                message=(
                    f"Executor profile {profile_name!r} has unsupported pip_install_mode={mode!r}; "
                    f"expected one of {sorted(PIP_INSTALL_MODES)}."
                ),
                source=f"executor:{profile_name}.pip_install_mode",
            )
        )
        return

    if mode in {"online", "online_best_effort"}:
        issues.append(
            AirgapIssue(
                severity="warning",
                code="remote_pip_online_install",
                message=(
                    f"Executor profile {profile_name!r} installs pip_extras online at job launch. "
                    "For airgap runs, bake them into the task image or use offline_wheelhouse."
                ),
                source=f"executor:{profile_name}.pip_install_mode",
            )
        )
    elif mode == "offline_wheelhouse" and not profile.get("pip_wheelhouse"):
        issues.append(
            AirgapIssue(
                severity="warning",
                code="remote_pip_wheelhouse_unspecified",
                message=(
                    f"Executor profile {profile_name!r} uses offline_wheelhouse but does not set "
                    f"pip_wheelhouse. The default is {AIRGAP_CONTAINER_WHEELHOUSE}; ensure it is "
                    "present in the task image or mounted into the job."
                ),
                source=f"executor:{profile_name}.pip_install_mode",
            )
        )
    elif mode == "offline_wheelhouse" and not _truthy(profile.get("pip_no_deps")):
        issues.append(
            AirgapIssue(
                severity="warning",
                code="remote_pip_wheelhouse_deps_mutable",
                message=(
                    f"Executor profile {profile_name!r} installs from an offline wheelhouse but does "
                    "not set pip_no_deps=true. Pip may upgrade task-image packages from the wheelhouse; "
                    "use pip_no_deps with explicit pip_required_imports, or pin a requirements/constraints pair."
                ),
                source=f"executor:{profile_name}.pip_install_mode",
            )
        )


def _scan_remote_startup_commands(
    profile_name: str,
    profile: Mapping[str, Any],
    *,
    issues: list[AirgapIssue],
) -> None:
    commands = profile.get("startup_commands") or []
    if isinstance(commands, str):
        commands = [commands]
    for index, command in enumerate(commands):
        command_text = str(command)
        if not ONLINE_STARTUP_RE.search(command_text):
            continue
        issues.append(
            AirgapIssue(
                severity="warning",
                code="remote_startup_online_command",
                message=(
                    f"Executor profile {profile_name!r} startup command appears to perform an online "
                    "install/download/clone. Move this dependency into the task image, mounted assets, "
                    "or offline_wheelhouse before customer airgap delivery."
                ),
                source=f"executor:{profile_name}.startup_commands.{index}",
            )
        )


def _scan_training_smoke_config(config_data: Mapping[str, Any], *, issues: list[AirgapIssue]) -> None:
    """Catch tiny Megatron smoke configs that fail before the first train step."""

    train = _select_mapping(config_data, ("train",))
    scheduler = _select_mapping(config_data, ("scheduler",))
    train_iters = _number_or_none(train.get("train_iters"))
    warmup_iters = _number_or_none(
        scheduler.get("lr_warmup_iters")
        if "lr_warmup_iters" in scheduler
        else scheduler.get("lr_warmup_steps")
    )
    decay_iters = _number_or_none(
        scheduler.get("lr_decay_iters")
        if "lr_decay_iters" in scheduler
        else scheduler.get("lr_decay_steps")
    )
    decay_iters = decay_iters if decay_iters is not None else train_iters

    if warmup_iters is None or decay_iters is None:
        return
    if warmup_iters < decay_iters:
        return
    issues.append(
        AirgapIssue(
            severity="warning",
            code="megatron_warmup_not_less_than_decay",
            message=(
                "Megatron requires scheduler warmup steps to be less than decay/train steps. "
                "For a one-iteration smoke, set scheduler.lr_warmup_iters=0; otherwise "
                "increase train.train_iters above scheduler.lr_warmup_iters."
            ),
            source="scheduler.lr_warmup_iters",
        )
    )


def _number_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text or "${" in text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _record_executor_mount(
    profile_name: str,
    mount: Any,
    *,
    manual_inputs: list[AirgapAsset],
) -> None:
    source = f"executor:{profile_name}.mounts"
    if isinstance(mount, str):
        host_path = mount.split(":", 1)[0]
    elif isinstance(mount, Mapping):
        host_path = str(mount.get("src") or mount.get("source") or mount.get("host_path") or "")
    else:
        return
    if not host_path:
        return
    manual_inputs.append(
        AirgapAsset(
            kind="local_path",
            id=host_path,
            source=source,
            field="mounts",
            note="Mounted host path must exist on the executor-side filesystem.",
        )
    )


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _walk_mapping(value: Mapping[str, Any], prefix: tuple[str, ...] = ()) -> Iterable[tuple[str, Any]]:
    for key, child in value.items():
        path = (*prefix, str(key))
        yield ".".join(path), child
        if isinstance(child, Mapping):
            yield from _walk_mapping(child, path)
        elif isinstance(child, list):
            for index, item in enumerate(child):
                item_path = (*path, str(index))
                yield ".".join(item_path), item
                if isinstance(item, Mapping):
                    yield from _walk_mapping(item, item_path)


def _workflow_items(
    locks: Iterable[Mapping[str, Any]], section: str, *, prefix_source: bool = False
) -> Iterable[dict[str, Any]]:
    for lock in locks:
        step_id = str((lock.get("step", {}) or {}).get("id") or "step")
        for item in lock.get(section, []) or []:
            if not isinstance(item, Mapping):
                continue
            data = dict(item)
            if prefix_source and data.get("source"):
                data["source"] = _workflow_source(step_id, str(data["source"]))
            yield data


def _workflow_base_images(locks: Iterable[Mapping[str, Any]]) -> Iterable[dict[str, Any]]:
    for lock in locks:
        step_id = str((lock.get("step", {}) or {}).get("id") or "step")
        for item in ((lock.get("runtime", {}) or {}).get("base_images", []) or []):
            if not isinstance(item, Mapping):
                continue
            data = dict(item)
            if data.get("source"):
                data["source"] = _workflow_source(step_id, str(data["source"]))
            yield data


def _workflow_source(step_id: str, source: str) -> str:
    if source == "pyproject.toml":
        return source
    return f"{step_id}:{source}"


def _merge_asset_dicts(items: Iterable[Mapping[str, Any] | AirgapAsset]) -> list[dict[str, Any]]:
    # Accept both already-serialized dicts (from prior lock merges) and live
    # AirgapAsset dataclasses (returned by ``_compile_executor_profiles``); the
    # workflow path mixes both when ``add_executor_profiles`` runs over a
    # merged workflow lock.
    fields = set(AirgapAsset.__dataclass_fields__)
    assets: list[AirgapAsset] = []
    for item in items:
        if isinstance(item, AirgapAsset):
            assets.append(item)
        elif isinstance(item, Mapping):
            assets.append(AirgapAsset(**{key: item.get(key) for key in fields}))
    return [asdict(asset) for asset in _prepare_assets_for_lock(_dedupe_assets(assets))]


def _merge_issue_dicts(items: Iterable[Mapping[str, Any] | AirgapIssue]) -> list[dict[str, Any]]:
    fields = set(AirgapIssue.__dataclass_fields__)
    issues: list[AirgapIssue] = []
    for item in items:
        if isinstance(item, AirgapIssue):
            issues.append(item)
        elif isinstance(item, Mapping):
            issues.append(AirgapIssue(**{key: item.get(key) for key in fields}))
    return [asdict(issue) for issue in _dedupe_issues(issues)]


def _merge_executor_dicts(items: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str | None]] = set()
    out: list[dict[str, Any]] = []
    for item in items:
        profile = str(item.get("profile") or "")
        executor = str(item.get("executor") or "")
        env_file = item.get("env_file")
        key = (profile, executor, str(env_file) if env_file is not None else None)
        if not profile or key in seen:
            continue
        seen.add(key)
        out.append(dict(item))
    return sorted(out, key=lambda item: (str(item.get("profile")), str(item.get("executor"))))


def _merge_python_runtime(locks: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    extras: set[str] = set()
    file_keys: set[tuple[str, str]] = set()
    files: list[dict[str, str]] = []
    for lock in locks:
        python = (lock.get("runtime", {}) or {}).get("python", {}) or {}
        extras.update(str(extra) for extra in python.get("extras", []) or [])
        for file in python.get("files", []) or []:
            if not isinstance(file, Mapping):
                continue
            path = str(file.get("path") or "")
            sha256 = str(file.get("sha256") or "")
            if not path or not sha256 or (path, sha256) in file_keys:
                continue
            file_keys.add((path, sha256))
            files.append({"path": path, "sha256": sha256})
    return {
        "manager": "uv",
        "uv_version": AIRGAP_UV_VERSION,
        "offline_wheelhouse": AIRGAP_CONTAINER_WHEELHOUSE,
        "extras": sorted(extras),
        "files": sorted(files, key=lambda item: item["path"]),
    }


def _merge_task_containers(locks: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Merge per-step remote task-image readiness hints for workflow locks."""

    seen: set[tuple[str, str]] = set()
    out: list[dict[str, Any]] = []
    for lock in locks:
        step_id = str((lock.get("step", {}) or {}).get("id") or "")
        runtime = lock.get("runtime", {}) or {}
        for item in runtime.get("task_containers", []) or []:
            if not isinstance(item, Mapping):
                continue
            data = dict(item)
            data["step_id"] = str(data.get("step_id") or step_id)
            image = str(data.get("image") or "")
            key = (data["step_id"], image)
            if not data["step_id"] or key in seen:
                continue
            seen.add(key)
            out.append(data)
    return sorted(out, key=lambda item: (str(item.get("step_id")), str(item.get("image"))))


def _merge_provenance(locks: Iterable[Mapping[str, Any]], *, repo_root: Path | None) -> dict[str, Any]:
    file_keys: set[tuple[str, str]] = set()
    files: list[dict[str, str]] = []
    first_repo_root: str | None = str(repo_root) if repo_root is not None else None
    for lock in locks:
        provenance = lock.get("provenance", {}) or {}
        if first_repo_root is None and provenance.get("repo_root"):
            first_repo_root = str(provenance["repo_root"])
        for file in provenance.get("files", []) or []:
            if not isinstance(file, Mapping):
                continue
            path = str(file.get("path") or "")
            sha256 = str(file.get("sha256") or "")
            if not path or not sha256 or (path, sha256) in file_keys:
                continue
            file_keys.add((path, sha256))
            files.append({"path": path, "sha256": sha256})
    return {
        "repo_root": first_repo_root,
        "files": sorted(files, key=lambda item: item["path"]),
    }


def read_lock(path: str | Path) -> dict[str, Any]:
    """Read an airgap lockfile."""

    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"{path}: airgap lock must be a YAML mapping")
    return data


def build_delivery_plan(lock: Mapping[str, Any]) -> dict[str, Any]:
    """Build a standard airgap preparation checklist from a lock mapping."""

    runtime = dict(lock.get("runtime", {}) or {})
    layout = dict(runtime.get("bundle_layout") or _bundle_layout())
    task_containers = [
        item for item in (runtime.get("task_containers", []) or []) if isinstance(item, Mapping)
    ]
    assets = [asset for asset in lock.get("assets", []) or [] if isinstance(asset, Mapping)]
    base_images = [
        image
        for image in (runtime.get("base_images", []) or [])
        if isinstance(image, Mapping) and image.get("id")
    ]
    services = [service for service in lock.get("services", []) or [] if isinstance(service, Mapping)]
    manual_inputs = [item for item in lock.get("manual_inputs", []) or [] if isinstance(item, Mapping)]
    unresolved_env = [item for item in lock.get("unresolved_env", []) or [] if isinstance(item, Mapping)]
    remote_executors = _remote_executor_summaries(lock)

    download_assets = [
        _asset_plan_item(asset, layout=layout)
        for asset in assets
        if asset.get("delivery") == "external" and _truthy(asset.get("required", True))
    ]
    optional_download_assets = [
        _asset_plan_item(asset, layout=layout)
        for asset in assets
        if asset.get("delivery") == "external" and not _truthy(asset.get("required", True))
    ]
    runtime_assets = [
        _asset_plan_item(asset, layout=layout)
        for asset in assets
        if asset.get("delivery") == "runtime"
    ]
    customer_inputs = [_customer_input_plan_item(item) for item in manual_inputs]
    service_items = [_service_plan_item(service) for service in services]
    env_items = [_env_plan_item(item) for item in unresolved_env]
    container_images = [_image_plan_item(image) for image in base_images]

    standard_mounts = []
    if download_assets:
        standard_mounts.append(
            {
                "id": "local_smoke_airgap_assets",
                "host_path": f"<airgap-bundle>/{layout['assets_dir']}",
                "container_path": layout["container_asset_root"],
                "mode": "ro",
                "required_for": sorted({str(item["kind"]) for item in download_assets}),
                "scope": "local_smoke_test",
            }
        )
        standard_mounts.append(
            {
                "id": "remote_airgap_assets",
                "host_path": "<customer-persistent-asset-root>",
                "container_path": layout["container_asset_root"],
                "mode": "ro",
                "required_for": sorted({str(item["kind"]) for item in download_assets}),
                "scope": "remote_execution",
            }
        )
    if customer_inputs:
        standard_mounts.append(
            {
                "id": "customer_inputs",
                "host_path": "<customer-selected-path>",
                "container_path": "<path expected by step config or override>",
                "mode": "ro/rw",
                "required_for": ["local_path"],
            }
        )

    return {
        "schema_version": AIRGAP_SCHEMA_VERSION,
        "stages": _delivery_stages(layout=layout),
        "bundle": {
            "runtime_dir": layout["runtime_dir"],
            "assets_dir": layout["assets_dir"],
            "container_asset_root": layout["container_asset_root"],
        },
        "asset_locations": {
            "connected_staging_dir": f"<airgap-bundle>/{layout['assets_dir']}",
            "local_smoke_mount": f"<airgap-bundle>/{layout['assets_dir']}:{layout['container_asset_root']}:ro",
            "remote_persistent_root": "<customer-persistent-asset-root>",
            "remote_container_mount": f"<customer-persistent-asset-root>:{layout['container_asset_root']}:ro",
            "note": (
                "The bundle assets directory is a transfer/staging layout. For remote execution, copy or sync "
                "the same contents to persistent storage visible to the executor."
            ),
        },
        "execution": {
            "mode": "remote" if remote_executors else "local_or_unknown",
            "remote_executors": remote_executors,
            "asset_fetch_default": "remote_stage" if remote_executors else "runtime_only",
            "note": (
                "Remote runs use the task image selected by the executor profile or step runspec. "
                "Do not rely on run-time source/package sync for large assets or Python dependencies. "
                "Remote runs should reference assets from persistent storage mounted into the job container."
                if remote_executors
                else "Large assets are still opt-in for local smoke tests; use --include-assets when needed."
            ),
        },
        "runtime_image": {
            "scope": "submitter",
            "build_input": f"<airgap-bundle>/{layout['runtime_dir']}",
            "contains": ["source tree", "uv", "offline Python wheelhouse", "offline.env"],
            "does_not_contain": ["large model weights", "datasets", "customer checkpoints"],
            "note": (
                "This image submits and navigates Nemotron Customizer steps. It is not the remote "
                "training/eval runtime. Lepton/DGX/Slurm profiles should point at mirrored or "
                "derivative task-specific images."
            ),
        },
        "download_assets": download_assets,
        "optional_download_assets": optional_download_assets,
        "runtime_assets": runtime_assets,
        "container_images": container_images,
        "task_containers": [_task_container_plan_item(item) for item in task_containers],
        "standard_mounts": standard_mounts,
        "customer_inputs": customer_inputs,
        "services": service_items,
        "environment": env_items,
    }


def _delivery_stages(*, layout: Mapping[str, str]) -> list[dict[str, str]]:
    """Return the canonical airgap customer journey used by CLI and docs."""

    return [
        {
            "stage": "1. Select and lock",
            "where": "connected prep machine",
            "output": "airgap.lock.yaml",
            "rule": "Lock all known step_id:config targets together.",
        },
        {
            "stage": "2. Fetch submitter runtime",
            "where": "connected prep machine",
            "output": f"<airgap-bundle>/{layout['runtime_dir']}",
            "rule": (
                "Fetch repo Python wheels and small support assets; leave large models and datasets outside the image."
            ),
        },
        {
            "stage": "3. Probe and rebuild task images",
            "where": "connected prep machine with task images",
            "output": "remote-gaps.yaml and derivative task images",
            "rule": "Probe exact executor images, then bake small missing packages and repo checkouts for production.",
        },
        {
            "stage": "4. Stage persistent assets",
            "where": "customer/executor persistent storage",
            "output": "HF cache, datasets, repos, wheelhouse, init scripts",
            "rule": "Remote jobs read models, datasets, checkpoints, and optional repo caches from mounted storage.",
        },
        {
            "stage": "5. Build submitter image",
            "where": "connected or airgapped Docker host",
            "output": "local Docker image",
            "rule": "Bake source, uv, offline.env, and runtime wheels only.",
        },
        {
            "stage": "6. Smoke and verify",
            "where": "inside submitter/runtime image",
            "output": "job logs, checkpoints, and outputs",
            "rule": "Logs should show mounted wheelhouse/cache paths and no public-network fetches.",
        },
    ]


def _asset_plan_item(asset: Mapping[str, Any], *, layout: Mapping[str, str]) -> dict[str, Any]:
    bundle_path = _maybe_str(asset.get("bundle_path"))
    container_path = f"{AIRGAP_CONTAINER_ROOT}/{bundle_path}" if bundle_path else None
    delivery = _maybe_str(asset.get("delivery"))
    action = "stage_external_asset_for_persistent_storage"
    if delivery == "runtime":
        action = "resolve_into_runtime_wheelhouse"
    item = {
        "kind": str(asset.get("kind") or ""),
        "id": str(asset.get("id") or ""),
        "source": _maybe_str(asset.get("source")),
        "revision": _maybe_str(asset.get("revision")),
        "required": _truthy(asset.get("required", True)),
        "delivery": delivery,
        "bundle_path": bundle_path,
        "container_path": container_path,
        "customer_action": action,
    }
    if asset.get("kind") in {"hf_model", "hf_dataset"}:
        item["runtime_lookup"] = layout["container_hf_cache"]
    elif asset.get("kind") == "git_repo":
        item["runtime_lookup"] = layout["container_repos"]
    return item


def _remote_executor_summaries(lock: Mapping[str, Any]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for item in lock.get("executors", []) or []:
        if not isinstance(item, Mapping):
            continue
        executor = str(item.get("executor") or "local").lower()
        if executor not in REMOTE_EXECUTORS:
            continue
        out.append(
            {
                "profile": str(item.get("profile") or ""),
                "executor": executor,
                "env_file": str(item.get("env_file") or ""),
            }
        )
    return out


def _customer_input_plan_item(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "kind": str(item.get("kind") or "local_path"),
        "id": str(item.get("id") or ""),
        "source": _maybe_str(item.get("source")),
        "field": _maybe_str(item.get("field")),
        "note": _maybe_str(item.get("note")),
        "customer_action": "provide_customer_path_or_config_override",
        "known_at_lock_time": False,
    }


def _service_plan_item(service: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "kind": str(service.get("kind") or "service"),
        "id": str(service.get("id") or ""),
        "source": _maybe_str(service.get("source")),
        "field": _maybe_str(service.get("field")),
        "note": _maybe_str(service.get("note")),
        "customer_action": "provide_in_network_endpoint",
    }


def _env_plan_item(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "kind": "env",
        "id": str(item.get("id") or ""),
        "source": _maybe_str(item.get("source")),
        "field": _maybe_str(item.get("field")),
        "note": _maybe_str(item.get("note")),
        "customer_action": "provide_environment_variable",
    }


def _image_plan_item(image: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "kind": "docker_image",
        "id": str(image.get("id") or ""),
        "source": _maybe_str(image.get("source")),
        "scope": "remote_task_image_or_local_base_image",
        "customer_action": "mirror_to_customer_registry_and_pin_by_digest",
    }


def _task_container_plan_item(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "step_id": str(item.get("step_id") or ""),
        "image": str(item.get("image") or ""),
        "candidate_imports": list(item.get("candidate_imports") or []),
        "package_hints": list(item.get("package_hints") or []),
        "repo_mounts": list(item.get("repo_mounts") or []),
        "probe_action": "run task image import probe and write remote-gaps.yaml",
        "production_action": "bake missing packages/repos into a derivative task image and pin it by digest",
    }


def verify_lock(
    lock: Mapping[str, Any], *, strict: bool = False, bundle_dir: str | Path | None = None
) -> list[AirgapIssue]:
    """Static verification for an airgap lock.

    ``strict`` treats warnings as errors. ``bundle_dir`` lets callers check
    that downloaded/bundled assets are present under the expected layout.
    """

    issues = [AirgapIssue(**issue) for issue in lock.get("issues", []) if isinstance(issue, dict)]

    assets = list(lock.get("assets", []) or [])
    services = list(lock.get("services", []) or [])
    manual_inputs = list(lock.get("manual_inputs", []) or [])
    unresolved_env = list(lock.get("unresolved_env", []) or [])
    base_images = list((lock.get("runtime", {}) or {}).get("base_images", []) or [])

    for image in base_images:
        image_id = str(image.get("id", ""))
        if image_id and not DOCKER_DIGEST_RE.search(image_id):
            issues.append(
                AirgapIssue(
                    severity="warning",
                    code="unpinned_image",
                    message=f"Container image {image_id!r} is not pinned by digest.",
                    source=image.get("source"),
                )
            )

    for asset in assets:
        kind = asset.get("kind")
        revision = str(asset.get("revision") or "")
        source = asset.get("source")
        if kind in {"git_repo", "python_git"} and revision and not GIT_SHA_RE.match(revision):
            issues.append(
                AirgapIssue(
                    severity="warning",
                    code="floating_git_ref",
                    message=f"{asset.get('id')} uses non-commit git ref {revision!r}.",
                    source=source,
                )
            )
        if kind in {"hf_model", "hf_dataset"} and revision in PINNED_HF_REVISIONS:
            issues.append(
                AirgapIssue(
                    severity="warning",
                    code="floating_hf_revision",
                    message=f"{asset.get('id')} uses floating HF revision {revision!r}.",
                    source=source,
                )
            )

    if services:
        issues.append(
            AirgapIssue(
                severity="info",
                code="service_closure",
                message=(
                    "Service endpoints are runtime dependencies; package or provide them inside "
                    "the airgapped network."
                ),
            )
        )

    for env_asset in unresolved_env:
        issues.append(
            AirgapIssue(
                severity="warning",
                code="unresolved_env",
                message=f"Required environment variable {env_asset.get('id')!r} has no default.",
                source=env_asset.get("source"),
            )
        )

    for item in manual_inputs:
        item_id = str(item.get("id") or "")
        if item_id.startswith("/path/to/") or item_id.startswith("./path/to/"):
            issues.append(
                AirgapIssue(
                    severity="warning",
                    code="manual_placeholder",
                    message=f"Manual input {item_id!r} is still a placeholder.",
                    source=item.get("source"),
                )
            )

    if bundle_dir is not None:
        issues.extend(_verify_bundle_assets(lock, Path(bundle_dir)))

    if strict:
        strict_issues: list[AirgapIssue] = []
        for issue in issues:
            if issue.severity == "info":
                strict_issues.append(issue)
            else:
                strict_issues.append(
                    AirgapIssue(
                        severity="error",
                        code=issue.code,
                        message=issue.message,
                        source=issue.source,
                    )
                )
        issues = strict_issues

    return _dedupe_issues(issues)


def _load_step_manifest(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _load_step_airgap(step_dir: Path) -> dict[str, Any]:
    """Load optional per-step airgap metadata.

    ``step.toml`` remains the public step contract. ``airgap.yaml`` is a
    narrower deployment contract for disconnected runs: which config fields
    are likely model/data/repo/service references, and which imports/packages
    the remote task image should be probed for before production bake.
    """
    path = step_dir / "airgap.yaml"
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise TypeError(f"{path}: airgap.yaml must be a YAML mapping")
    return data


def _merge_step_airgap(manifest: Mapping[str, Any], hints: Mapping[str, Any]) -> dict[str, Any]:
    """Merge ``airgap.yaml`` hints into the manifest-shaped ``airgap`` block."""
    merged = dict(manifest)
    existing = manifest.get("airgap")
    airgap: dict[str, Any] = dict(existing) if isinstance(existing, Mapping) else {}
    for key, value in hints.items():
        if key == "version":
            continue
        if key in {"assets", "issues", "extras", "config_fields"}:
            airgap[key] = _merge_airgap_list(airgap.get(key), value)
        elif key in {"task_container", "submitter"}:
            current = airgap.get(key)
            combined = dict(current) if isinstance(current, Mapping) else {}
            if isinstance(value, Mapping):
                combined.update(value)
            airgap[key] = combined
        else:
            airgap[key] = value
    if airgap:
        merged["airgap"] = airgap
    return merged


def _merge_airgap_list(left: Any, right: Any) -> list[Any]:
    out: list[Any] = []
    for value in (left, right):
        if value is None:
            continue
        if isinstance(value, list):
            out.extend(value)
        else:
            out.append(value)
    return out


def _resolve_step(step_id: str) -> StepInfo:
    steps = discover_steps()
    by_id = {step.id: step for step in steps}
    if step_id in by_id:
        return by_id[step_id]
    # Tail-name shortcut: only honor it when exactly one step matches AND
    # the user supplied something that cannot collide with a category prefix.
    # Two unrelated steps with the same directory name (``prep/translate`` and
    # ``curate/translate``) used to silently change behavior whenever a third
    # ``*/translate`` step was added; the user-facing ``category/name`` form
    # is the only stable identifier so we surface the ambiguity instead.
    if "/" not in step_id:
        tail_matches = [step for step in steps if step.path.name == step_id]
        if len(tail_matches) == 1:
            return tail_matches[0]
        if len(tail_matches) > 1:
            options = ", ".join(sorted(step.id for step in tail_matches))
            raise ValueError(
                f"Step name {step_id!r} is ambiguous. Use the full id, e.g. one of: {options}"
            )
    available = ", ".join(sorted(by_id))
    raise ValueError(f"Unknown step id {step_id!r}. Available: {available}")


def _runspec_to_dict(spec: Runspec, repo_root: Path) -> dict[str, Any]:
    return {
        "name": spec.name,
        "image": spec.image,
        "launch": spec.run.launch,
        "cmd": spec.run.cmd,
        "workdir": spec.run.workdir,
        "config_dir": _relpath(spec.config_dir, repo_root),
        "default_config": spec.config.default,
        "resources": {
            "nodes": spec.resources.nodes,
            "gpus_per_node": spec.resources.gpus_per_node,
        },
    }


def _add_runtime_assets(
    *,
    spec: Runspec,
    config_data: Mapping[str, Any],
    assets: list[AirgapAsset],
    issues: list[AirgapIssue],
) -> None:
    if spec.image:
        assets.append(
            AirgapAsset(
                kind="docker_image",
                id=spec.image,
                source="runspec.image",
                note="Pin to a digest before customer delivery.",
            )
        )
        if not DOCKER_DIGEST_RE.search(spec.image):
            issues.append(
                AirgapIssue(
                    severity="warning",
                    code="unpinned_image",
                    message=f"Runspec image {spec.image!r} is not pinned by digest.",
                    source="runspec.image",
                )
            )

    env = _select_mapping(config_data, ("run", "env"))
    for key in ("container", "container_image", "image"):
        value = env.get(key)
        if isinstance(value, str) and value:
            assets.append(
                AirgapAsset(
                    kind="docker_image",
                    id=value,
                    source=f"run.env.{key}",
                    note="Pin to a digest before customer delivery.",
                )
            )


def _add_pyproject_assets(repo_root: Path, *, assets: list[AirgapAsset], issues: list[AirgapIssue]) -> None:
    pyproject = repo_root / "pyproject.toml"
    if not pyproject.exists():
        return
    text = pyproject.read_text(encoding="utf-8")
    for match in re.finditer(r"git\+https://[^\"'\s,]+", text):
        raw = "git+" + match.group(0).split("git+", 1)[1]
        git = _parse_git_spec(raw)
        if git is None:
            continue
        url, ref = git
        assets.append(
            AirgapAsset(
                kind="python_git",
                id=_repo_name(url),
                source="pyproject.toml",
                revision=ref,
                note=url,
            )
        )
        _warn_if_git_ref_floating(ref, "pyproject.toml", issues)


def _scan_manifest_airgap(
    manifest: Mapping[str, Any],
    *,
    assets: list[AirgapAsset],
    services: list[AirgapAsset],
    issues: list[AirgapIssue],
) -> None:
    airgap = manifest.get("airgap")
    if not isinstance(airgap, Mapping):
        return
    for entry in airgap.get("assets", []) or []:
        if not isinstance(entry, Mapping):
            continue
        kind = str(entry.get("kind", "")).strip()
        asset_id = str(entry.get("id") or entry.get("url") or "")
        if not kind or not asset_id:
            continue
        target = services if kind == "service" else assets
        target.append(
            AirgapAsset(
                kind=kind,  # type: ignore[arg-type]
                id=asset_id,
                source="step.toml[airgap]",
                revision=_maybe_str(entry.get("revision")),
                repo_type=_asset_repo_type(kind, entry),
                target=_maybe_str(entry.get("target")),
                note=_maybe_str(entry.get("note")),
                required=bool(entry.get("required", True)),
            )
        )
    for entry in airgap.get("issues", []) or []:
        if isinstance(entry, Mapping):
            issues.append(
                AirgapIssue(
                    severity=str(entry.get("severity", "warning")),  # type: ignore[arg-type]
                    code=str(entry.get("code", "manifest_issue")),
                    message=str(entry.get("message", "")),
                    source="step.toml[airgap]",
                )
            )


def _scan_airgap_config_fields(
    manifest: Mapping[str, Any],
    config_data: Mapping[str, Any],
    *,
    assets: list[AirgapAsset],
    services: list[AirgapAsset],
    manual_inputs: list[AirgapAsset],
    issues: list[AirgapIssue],
) -> None:
    airgap = manifest.get("airgap")
    if not isinstance(airgap, Mapping):
        return
    for entry in airgap.get("config_fields", []) or []:
        if not isinstance(entry, Mapping):
            continue
        field_path = str(entry.get("path") or "").strip()
        kind = str(entry.get("kind") or "").strip()
        if not field_path or not kind:
            continue
        value = _get_config_path(config_data, field_path)
        if value is None:
            if entry.get("required", False):
                issues.append(
                    AirgapIssue(
                        severity="warning",
                        code="airgap_field_missing",
                        message=f"airgap.yaml references missing config field {field_path!r}.",
                        source="airgap.yaml",
                    )
                )
            continue
        for item in _flatten_values(value):
            if item in {"", None}:
                continue
            item_id = str(item)
            source = f"config.{field_path}"
            note = _maybe_str(entry.get("note"))
            if kind == "service":
                services.append(
                    AirgapAsset(
                        kind="service",
                        id=item_id,
                        source=source,
                        field=field_path,
                        note=note or "Provide an in-network endpoint for the airgapped environment.",
                    )
                )
            elif kind == "local_path":
                manual_inputs.append(
                    AirgapAsset(
                        kind="local_path",
                        id=item_id,
                        source=source,
                        field=field_path,
                        note=note or "Path must exist on persistent storage visible to the selected executor.",
                    )
                )
            elif kind in {"hf_model", "hf_dataset"} and _looks_like_path(item_id):
                manual_inputs.append(
                    AirgapAsset(
                        kind="local_path",
                        id=item_id,
                        source=source,
                        field=field_path,
                        note=note or "Model or dataset path must be mounted on executor-visible persistent storage.",
                    )
                )
            elif kind in {"hf_model", "hf_dataset", "git_repo", "url"}:
                assets.append(
                    AirgapAsset(
                        kind=kind,  # type: ignore[arg-type]
                        id=item_id.removeprefix("hf://"),
                        source=source,
                        field=field_path,
                        repo_type=_asset_repo_type(kind, entry),
                        revision=_maybe_str(entry.get("revision")),
                        note=note,
                        delivery="external",
                    )
                )
            else:
                issues.append(
                    AirgapIssue(
                        severity="warning",
                        code="airgap_field_kind_unknown",
                        message=f"airgap.yaml field {field_path!r} uses unknown kind {kind!r}.",
                        source="airgap.yaml",
                    )
                )


def _get_config_path(config: Mapping[str, Any], path: str) -> Any:
    current: Any = config
    for part in path.split("."):
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        else:
            return None
    return current


def _asset_repo_type(kind: str, entry: Mapping[str, Any]) -> str | None:
    explicit = _maybe_str(entry.get("repo_type"))
    if explicit:
        return explicit
    if kind == "hf_dataset":
        return "dataset"
    if kind == "hf_model":
        return "model"
    return None


def _flatten_values(value: Any) -> list[Any]:
    if isinstance(value, Mapping):
        return [item for nested in value.values() for item in _flatten_values(nested)]
    if isinstance(value, (list, tuple, set)):
        return [item for nested in value for item in _flatten_values(nested)]
    return [value]


def _task_container_readiness(
    repo_root: Path,
    *,
    step: StepInfo,
    spec: Runspec,
    manifest: Mapping[str, Any],
    config_data: Mapping[str, Any],
    base_images: Iterable[AirgapAsset],
) -> list[dict[str, Any]]:
    """Build generic readiness guidance for remote task images.

    The output is intentionally a plan, not an assertion that one image is
    universally correct. Submitter images run the CLI; task images run training,
    eval, or prep. Each task image should be probed on the connected machine,
    then either baked into a derivative image for production or paired with a
    small offline wheelhouse for smoke testing.
    """
    del config_data  # reserved for future config-driven image selection
    airgap = manifest.get("airgap")
    task_cfg = airgap.get("task_container") if isinstance(airgap, Mapping) else {}
    task_cfg = task_cfg if isinstance(task_cfg, Mapping) else {}

    candidate_imports = set(COMMON_TASK_IMPORTS.get(step.category, []))
    candidate_imports.update(_candidate_imports_from_step(repo_root, step.path / "step.py"))
    candidate_imports.update(_as_str_list(task_cfg.get("required_imports")))
    candidate_imports.update(
        import_name
        for package in _as_str_list(task_cfg.get("required_packages"))
        if (import_name := _package_to_import(package))
    )
    candidate_imports = {name for name in candidate_imports if _is_probe_import(name)}

    repo_mounts = _repo_mount_hints(manifest)
    package_hints = _as_str_list(task_cfg.get("required_packages"))
    if not package_hints:
        package_hints = sorted(_import_to_package_hint(name) for name in candidate_imports)

    images = sorted({asset.id for asset in base_images if asset.kind == "docker_image" and asset.id})
    if not images and spec.image:
        images = [spec.image]

    return [
        {
            "step_id": step.id,
            "image": image,
            "source": "runspec_or_executor_profile",
            "candidate_imports": sorted(candidate_imports),
            "package_hints": package_hints,
            "repo_mounts": repo_mounts,
            "note": _maybe_str(task_cfg.get("note")),
            "probe": {
                "strategy": "run import probe inside this exact image on the connected prep machine",
                "missing_imports_output": "remote-gaps.yaml",
            },
            "smoke_delivery": {
                "mode": "offline_wheelhouse",
                "rule": "stage only missing wheels to executor-visible storage and install with --no-deps",
            },
            "production_delivery": {
                "mode": "derivative_task_image",
                "rule": (
                    "bake missing packages and required repo checkouts into this task image, push to registry "
                    "or supported image path, then use pip_install_mode=preinstalled"
                ),
            },
        }
        for image in images
    ]


def _repo_mount_hints(manifest: Mapping[str, Any]) -> list[dict[str, str]]:
    airgap = manifest.get("airgap")
    if not isinstance(airgap, Mapping):
        return []
    task_cfg = airgap.get("task_container")
    if not isinstance(task_cfg, Mapping):
        return []
    out: list[dict[str, str]] = []
    for item in task_cfg.get("repo_mounts", []) or []:
        if not isinstance(item, Mapping):
            continue
        repo = str(item.get("repo") or item.get("name") or "").strip()
        target = str(item.get("target") or "").strip()
        if repo and target:
            out.append({"repo": repo, "target": target, "delivery": "stage_or_bake_at_locked_sha"})
    return out


def _candidate_imports_from_step(repo_root: Path, entrypoint: Path) -> set[str]:
    """Statically collect imports reachable from a step entrypoint.

    This is deliberately conservative and bounded. It follows local
    ``nemotron`` / ``nemo_runspec`` imports far enough to find helper-library
    imports, but it does not execute code and it does not imply that every
    candidate must be installed. The connected probe decides what is actually
    missing from the selected task image.
    """
    src_root = repo_root / "src"
    pending = [entrypoint]
    visited: set[Path] = set()
    imports: set[str] = set()
    while pending and len(visited) < 48:
        path = pending.pop()
        if path in visited or not path.exists():
            continue
        visited.add(path)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        package = _module_name_for_path(src_root, path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    _record_or_follow_import(alias.name, src_root=src_root, imports=imports, pending=pending)
            elif isinstance(node, ast.ImportFrom):
                module = _absolute_import_from(package, node.module, node.level)
                if module:
                    _record_or_follow_import(module, src_root=src_root, imports=imports, pending=pending)
    return imports


def _module_name_for_path(src_root: Path, path: Path) -> str:
    try:
        rel = path.relative_to(src_root).with_suffix("")
    except ValueError:
        return ""
    parts = list(rel.parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _absolute_import_from(package: str, module: str | None, level: int) -> str | None:
    if level <= 0:
        return module
    package_parts = package.split(".")[:-1]
    if level > 1:
        package_parts = package_parts[: -(level - 1)]
    if module:
        package_parts.extend(module.split("."))
    return ".".join(part for part in package_parts if part)


def _record_or_follow_import(
    module: str,
    *,
    src_root: Path,
    imports: set[str],
    pending: list[Path],
) -> None:
    root = module.split(".", 1)[0]
    if root in LOCAL_IMPORT_ROOTS:
        local = _module_to_path(src_root, module)
        if local is not None:
            pending.append(local)
        return
    if _is_probe_import(root):
        imports.add(root)


def _module_to_path(src_root: Path, module: str) -> Path | None:
    path = src_root.joinpath(*module.split("."))
    py_path = path.with_suffix(".py")
    if py_path.exists():
        return py_path
    init_path = path / "__init__.py"
    if init_path.exists():
        return init_path
    return None


def _is_probe_import(name: str) -> bool:
    if not name or name.startswith("_"):
        return False
    if name in LOCAL_IMPORT_ROOTS:
        return False
    if hasattr(sys, "stdlib_module_names") and name in sys.stdlib_module_names:
        return False
    return True


def _package_to_import(package: str) -> str | None:
    name = package.strip()
    if not name:
        return None
    for marker in ("==", ">=", "<=", "~=", "!=", ">", "<", ";"):
        if marker in name:
            name = name.split(marker, 1)[0]
    name = name.split("[", 1)[0].strip()
    return PACKAGE_IMPORT_ALIASES.get(name, name.replace("-", "_")) if name else None


def _import_to_package_hint(import_name: str) -> str:
    for package, alias in PACKAGE_IMPORT_ALIASES.items():
        if alias == import_name:
            return package
    return import_name.replace("_", "-")


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        return [str(item) for item in value.values()]
    if isinstance(value, Iterable):
        return [str(item) for item in value]
    return [str(value)]


def _python_runtime(
    repo_root: Path,
    *,
    spec: Runspec,
    manifest: Mapping[str, Any],
    issues: list[AirgapIssue] | None = None,
) -> dict[str, Any]:
    files = []
    missing: list[str] = []
    for name in ("pyproject.toml", "uv.lock"):
        path = repo_root / name
        if path.exists():
            files.append(asdict(_fingerprint(path, repo_root)))
        else:
            missing.append(name)
    if missing and issues is not None:
        issues.append(
            AirgapIssue(
                severity="warning",
                code="missing_python_metadata",
                message=(
                    f"repo_root={repo_root} is missing {', '.join(missing)}; the lock cannot describe "
                    "a reproducible offline wheelhouse. Run from a Nemotron checkout or pass --repo-root."
                ),
                source="runtime.python",
            )
        )
    extras = sorted(set(_uv_extras_from_runspec(spec) + _uv_extras_from_manifest(manifest)))
    return {
        "manager": "uv",
        "uv_version": AIRGAP_UV_VERSION,
        "offline_wheelhouse": AIRGAP_CONTAINER_WHEELHOUSE,
        "extras": extras,
        "files": files,
    }


def _bundle_layout() -> dict[str, str]:
    return {
        "runtime_dir": AIRGAP_RUNTIME_DIR,
        "assets_dir": AIRGAP_ASSETS_DIR,
        "container_root": AIRGAP_CONTAINER_ROOT,
        "container_wheelhouse": AIRGAP_CONTAINER_WHEELHOUSE,
        "container_asset_root": AIRGAP_CONTAINER_ASSET_ROOT,
        "container_hf_home": AIRGAP_CONTAINER_HF_HOME,
        "container_hf_cache": AIRGAP_CONTAINER_HF_CACHE,
        "container_repos": AIRGAP_CONTAINER_REPOS,
    }


def _uv_extras_from_runspec(spec: Runspec) -> list[str]:
    if not spec.run.cmd or "uv " not in spec.run.cmd:
        return []
    extras: list[str] = []
    parts = spec.run.cmd.split()
    for idx, part in enumerate(parts):
        if part == "--extra" and idx + 1 < len(parts):
            extras.append(parts[idx + 1])
        elif part.startswith("--extra="):
            extras.append(part.split("=", 1)[1])
    return extras


def _uv_extras_from_manifest(manifest: Mapping[str, Any]) -> list[str]:
    airgap = manifest.get("airgap")
    if not isinstance(airgap, Mapping):
        return []
    extras = airgap.get("extras") or []
    if isinstance(extras, str):
        return [extras]
    if isinstance(extras, list):
        return [str(item) for item in extras]
    return []


def _provenance_files(repo_root: Path, step_dir: Path, config_path: Path) -> list[FileFingerprint]:
    candidates = [
        repo_root / "pyproject.toml",
        repo_root / "uv.lock",
        step_dir / "step.toml",
        step_dir / "step.py",
        config_path,
    ]
    return [_fingerprint(path, repo_root) for path in candidates if path.exists()]


def _fingerprint(path: Path, repo_root: Path) -> FileFingerprint:
    return FileFingerprint(path=_relpath(path, repo_root), sha256=_sha256(path))


def _dedupe_fingerprints(files: Iterable[FileFingerprint]) -> list[FileFingerprint]:
    seen: set[tuple[str, str]] = set()
    out: list[FileFingerprint] = []
    for file in files:
        key = (file.path, file.sha256)
        if key in seen:
            continue
        seen.add(key)
        out.append(file)
    return sorted(out, key=lambda item: item.path)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _select_mapping(data: Mapping[str, Any], path: tuple[str, ...]) -> dict[str, Any]:
    value: Any = data
    for part in path:
        if not isinstance(value, Mapping):
            return {}
        value = value.get(part)
    return dict(value) if isinstance(value, Mapping) else {}


def _values_for_key(value: Any, wanted: str) -> list[Any]:
    out: list[Any] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key == wanted:
                out.append(child)
            out.extend(_values_for_key(child, wanted))
    elif isinstance(value, list):
        for child in value:
            out.extend(_values_for_key(child, wanted))
    return out


def _oc_env_defaults(value: str) -> list[tuple[str, str | None]]:
    out: list[tuple[str, str | None]] = []
    for match in OC_ENV_RE.finditer(value):
        env_name = match.group(1).strip()
        default = match.group(2)
        out.append((env_name, default))
    return out


def _expand_oc_env_defaults(value: str) -> list[str]:
    return [default for _, default in _oc_env_defaults(value) if default]


def _auto_mounts(value: str) -> list[tuple[str, str | None]]:
    return [(match.group(1), match.group(2)) for match in AUTO_MOUNT_RE.finditer(value)]


def _parse_git_spec(value: str) -> tuple[str, str] | None:
    match = GIT_REF_RE.match(value)
    if not match:
        return None
    return match.group("url"), match.group("ref")


def _repo_name(url: str) -> str:
    name = url.rstrip("/").split("/")[-1]
    return name[:-4] if name.endswith(".git") else name


def _warn_if_git_ref_floating(ref: str, source: str, issues: list[AirgapIssue]) -> None:
    if ref and not GIT_SHA_RE.match(ref):
        issues.append(
            AirgapIssue(
                severity="warning",
                code="floating_git_ref",
                message=f"Git ref {ref!r} should be resolved to a commit SHA for airgap delivery.",
                source=source,
            )
        )


def _looks_like_hf_id(value: str) -> bool:
    if not HF_ID_RE.match(value):
        return False
    if value.startswith(("/", "./", "../", "git+", "oci-archive://")):
        return False
    if "://" in value or ":" in value:
        return False
    return True


def _is_hf_context(path: tuple[str, ...]) -> bool:
    if not path:
        return False
    last = path[-1]
    if last in HF_ID_KEYS or last in TOKENIZER_KEYS:
        return True
    joined = ".".join(path)
    return any(marker in joined for marker in ("tokenizer", "model", "dataset", "policy", "reward"))


def _is_dataset_context(path: tuple[str, ...]) -> bool:
    joined = ".".join(path)
    return any(marker in joined for marker in ("dataset", "repo_id", "path_or_dataset_id", "data."))


def _looks_like_path(value: str) -> bool:
    if not value or "\n" in value:
        return False
    if value.startswith(("/", "./", "../", "~")):
        return True
    if any(value.startswith(prefix) for prefix in ("s3://", "gs://", "gcs://", "az://", "abfs://")):
        return True
    if "*" in value and "/" in value:
        return True
    return False


def _resolve_local_path(value: str, *, base_dir: Path, repo_root: Path) -> Path | None:
    if "${" in value or "://" in value:
        return None
    expanded = Path(os.path.expandvars(os.path.expanduser(value)))
    if expanded.is_absolute():
        return expanded
    candidates = [base_dir / expanded, repo_root / expanded, Path.cwd() / expanded]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (repo_root / expanded).resolve()


def _is_local_service_url(value: str) -> bool:
    from urllib.parse import urlparse

    parsed = urlparse(value)
    return parsed.hostname in LOCALHOST_NAMES


def _dedupe_assets(assets: Iterable[AirgapAsset]) -> list[AirgapAsset]:
    seen: set[tuple[Any, ...]] = set()
    out: list[AirgapAsset] = []
    for asset in assets:
        if asset.kind in {"hf_model", "hf_dataset", "git_repo", "docker_image", "python_git", "url"}:
            key = (
                asset.kind,
                asset.id,
                asset.revision,
                asset.repo_type,
                asset.target,
                asset.delivery,
                asset.bundle_path,
            )
        else:
            key = (
                asset.kind,
                asset.id,
                asset.revision,
                asset.repo_type,
                asset.target,
                asset.field,
                asset.delivery,
                asset.bundle_path,
            )
        if key in seen:
            continue
        seen.add(key)
        out.append(asset)
    return sorted(out, key=lambda a: (a.kind, a.id, a.source, a.field or ""))


def _prepare_assets_for_lock(assets: Iterable[AirgapAsset]) -> list[AirgapAsset]:
    return [_asset_with_delivery(asset) for asset in assets]


def _asset_with_delivery(asset: AirgapAsset) -> AirgapAsset:
    delivery = asset.delivery or _default_delivery(asset.kind)
    bundle_path = asset.bundle_path or _default_bundle_path(asset)
    return replace(asset, delivery=delivery, bundle_path=bundle_path)


def _default_delivery(kind: str) -> AssetDelivery:
    if kind == "python_git":
        return "runtime"
    if kind in {"hf_model", "hf_dataset", "git_repo", "url", "docker_image"}:
        return "external"
    return "manual"


def _default_bundle_path(asset: AirgapAsset) -> str | None:
    if asset.kind in {"hf_model", "hf_dataset"}:
        repo_type = asset.repo_type or ("dataset" if asset.kind == "hf_dataset" else "model")
        prefix = "datasets" if repo_type == "dataset" else "models"
        return f"{AIRGAP_ASSETS_DIR}/hf-cache/hub/{prefix}--{asset.id.replace('/', '--')}"
    if asset.kind == "git_repo":
        return f"{AIRGAP_ASSETS_DIR}/repos/{asset.id}"
    if asset.kind == "url":
        digest = hashlib.sha256(asset.id.encode("utf-8")).hexdigest()[:16]
        return f"{AIRGAP_ASSETS_DIR}/urls/{digest}"
    return None


def _dedupe_issues(issues: Iterable[AirgapIssue]) -> list[AirgapIssue]:
    seen: set[tuple[str, str | None]] = set()
    out: list[AirgapIssue] = []
    for issue in issues:
        key = (issue.code, issue.source)
        if key in seen:
            continue
        seen.add(key)
        out.append(issue)
    severity_order = {"error": 0, "warning": 1, "info": 2}
    return sorted(out, key=lambda i: (severity_order.get(i.severity, 3), i.code, i.source or ""))


def _verify_bundle_assets(lock: Mapping[str, Any], bundle_dir: Path) -> list[AirgapIssue]:
    issues: list[AirgapIssue] = []
    python_runtime = (lock.get("runtime", {}) or {}).get("python", {}) or {}
    if python_runtime.get("manager") == "uv":
        runtime_dir = bundle_dir / AIRGAP_RUNTIME_DIR
        wheelhouse = runtime_dir / "wheels"
        requirements = runtime_dir / "requirements-airgap.txt"
        offline_env = runtime_dir / "offline.env"
        if not wheelhouse.exists() or not wheelhouse.is_dir():
            issues.append(
                AirgapIssue(
                    severity="warning",
                    code="bundle_wheelhouse_missing",
                    message=f"Expected Python wheelhouse at {wheelhouse}.",
                    source="runtime.python",
                )
            )
        elif not any(wheelhouse.iterdir()):
            issues.append(
                AirgapIssue(
                    severity="warning",
                    code="bundle_wheelhouse_empty",
                    message=f"Python wheelhouse at {wheelhouse} is empty.",
                    source="runtime.python",
                )
            )
        if not requirements.exists():
            issues.append(
                AirgapIssue(
                    severity="warning",
                    code="bundle_requirements_missing",
                    message=f"Expected offline requirements file at {requirements}.",
                    source="runtime.python",
                )
            )
        if not offline_env.exists():
            issues.append(
                AirgapIssue(
                    severity="warning",
                    code="bundle_offline_env_missing",
                    message=f"Expected offline environment file at {offline_env}.",
                    source="offline_env",
                )
            )

    for asset in lock.get("assets", []) or []:
        if not isinstance(asset, Mapping):
            continue
        if asset.get("delivery") in {"runtime", "manual"}:
            continue
        kind = asset.get("kind")
        asset_id = str(asset.get("id") or "")
        expected: Path | None = None
        bundle_path = _maybe_str(asset.get("bundle_path"))
        if bundle_path:
            expected = bundle_dir / bundle_path
        elif kind in {"hf_model", "hf_dataset"}:
            repo_type = asset.get("repo_type") or ("dataset" if kind == "hf_dataset" else "model")
            prefix = "datasets" if repo_type == "dataset" else "models"
            expected = bundle_dir / AIRGAP_ASSETS_DIR / "hf-cache" / "hub" / f"{prefix}--{asset_id.replace('/', '--')}"
        elif kind in {"git_repo", "python_git"}:
            expected = bundle_dir / AIRGAP_ASSETS_DIR / "repos" / asset_id
        elif kind == "url":
            digest = hashlib.sha256(asset_id.encode("utf-8")).hexdigest()[:16]
            expected = bundle_dir / AIRGAP_ASSETS_DIR / "urls" / digest
        if expected is not None and not expected.exists():
            issues.append(
                AirgapIssue(
                    severity="warning",
                    code="bundle_asset_missing",
                    message=f"Expected bundled asset for {kind} {asset_id!r} at {expected}.",
                    source=asset.get("source"),
                )
            )
            continue
        # Re-verify the recorded sha256 for url-style payloads (the only kind
        # for which we currently capture a content hash). This catches bundles
        # that were corrupted in transit or swapped out after fetch.
        recorded_hash = str(asset.get("sha256") or "").strip().lower()
        if (
            kind == "url"
            and recorded_hash
            and expected is not None
        ):
            if expected.is_file():
                actual = _sha256(expected)
            elif expected.is_dir():
                blobs = [p for p in expected.iterdir() if p.is_file()]
                actual = _sha256(blobs[0]) if len(blobs) == 1 else None
            else:
                actual = None
            if actual is not None:
                if actual != recorded_hash:
                    issues.append(
                        AirgapIssue(
                            severity="error",
                            code="bundle_asset_sha256_mismatch",
                            message=(
                                f"sha256 mismatch for {asset_id!r}: lock recorded {recorded_hash}, "
                                f"bundled file is {actual}."
                            ),
                            source=asset.get("source"),
                        )
                    )
    return issues


def _relpath(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _maybe_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
