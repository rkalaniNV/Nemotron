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

"""``nemotron step airgap`` commands."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from shlex import quote as shlex_quote
from typing import Annotated, Any
from urllib.parse import urlparse

import typer
import yaml
from rich.console import Console
from rich.table import Table

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[no-redef]

from nemotron.steps.airgap import (
    AIRGAP_ASSETS_DIR,
    AIRGAP_RUNTIME_DIR,
    AIRGAP_UV_VERSION,
    LEPTON_INIT_SCRIPT_BUNDLE_PATH,
    OFFLINE_ENV,
    AirgapCompiler,
    AirgapTarget,
    build_delivery_plan,
    lock_to_dict,
    read_lock,
    verify_lock,
    write_lock,
)

console = Console()
DIRECT_REF_PREFIXES = ("git+", "http://", "https://", "file:")
UV_VERSION = AIRGAP_UV_VERSION

airgap_app = typer.Typer(
    name="airgap",
    help="Compile, fetch, build, and verify offline bundles for steps.",
    no_args_is_help=True,
    rich_markup_mode="rich",
    context_settings={"help_option_names": ["-h", "--help"]},
)


def lock_airgap(
    ctx: typer.Context,
    step_id: Annotated[str, typer.Argument(help="Step id, e.g. sft/automodel.")],
    config: Annotated[
        str | None,
        typer.Option("-c", "--config", help="Config name in the step config/ dir, or a YAML path."),
    ] = None,
    output: Annotated[
        Path,
        typer.Option("-o", "--output", help="Where to write the airgap lock YAML."),
    ] = Path("deploy/nemotron-customizer/airgap/airgap.lock.yaml"),
    repo_root: Annotated[
        Path | None,
        typer.Option("--repo-root", help="Repository root. Defaults to the current working directory."),
    ] = None,
    profiles: Annotated[
        list[str] | None,
        typer.Option("--profile", "-p", help="Executor env.toml profile to include in the lock."),
    ] = None,
    env_file: Annotated[
        Path | None,
        typer.Option("--env-file", help="env.toml path used to resolve executor profiles."),
    ] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Print the compiled lock as JSON-like YAML.")] = False,
) -> None:
    """Compile a static airgap lock for a selected step/config.

    Unknown arguments after the command are treated as Hydra/OmegaConf dotlist
    overrides, matching ``nemotron step run``.
    """

    compiler = AirgapCompiler(repo_root=repo_root)
    lock = compiler.compile(
        step_id=step_id,
        config_name=config,
        overrides=ctx.args,
        profiles=profiles or [],
        env_file=env_file,
    )
    write_lock(lock, output)
    lock_data = lock_to_dict(lock)

    if json_output:
        console.print(yaml.safe_dump(lock_data, sort_keys=False), soft_wrap=True)
        return

    _print_lock_summary(output, lock_data)


def lock_workflow_airgap(
    ctx: typer.Context,
    targets: Annotated[
        list[str],
        typer.Argument(
            help=(
                "Step specs as step_id, step_id:config, or "
                "step_id:config+key=value[,key=value...]; e.g. prep/sft_packing:tiny."
            )
        ),
    ],
    output: Annotated[
        Path,
        typer.Option("-o", "--output", help="Where to write the workflow airgap lock YAML."),
    ] = Path("deploy/nemotron-customizer/airgap/airgap.lock.yaml"),
    name: Annotated[str, typer.Option("--name", help="Workflow name recorded in the lockfile.")] = "workflow",
    repo_root: Annotated[
        Path | None,
        typer.Option("--repo-root", help="Repository root. Defaults to the current working directory."),
    ] = None,
    profiles: Annotated[
        list[str] | None,
        typer.Option("--profile", "-p", help="Executor env.toml profile to include in the lock."),
    ] = None,
    env_file: Annotated[
        Path | None,
        typer.Option("--env-file", help="env.toml path used to resolve executor profiles."),
    ] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Print the compiled lock as JSON-like YAML.")] = False,
) -> None:
    """Compile one workflow lock for several selected step/config targets.

    Unknown args after the target list are treated as Hydra/OmegaConf dotlist
    overrides applied to every target. Per-target overrides via ``+key=value``
    take precedence over workflow-wide overrides.
    """

    # Typer slurps every positional into ``targets``; split workflow-wide
    # overrides back out by looking for ``key=value`` shapes that aren't a
    # ``step_id[:config][+overrides]`` spec. Anything after ``--`` lands in
    # ``ctx.args`` and is also treated as a workflow override.
    real_targets: list[str] = []
    inline_overrides: list[str] = []
    for token in targets:
        if _looks_like_override(token):
            inline_overrides.append(token.lstrip("+"))
        else:
            real_targets.append(token)
    workflow_overrides = tuple(inline_overrides + list(ctx.args))
    if not real_targets:
        raise typer.BadParameter("At least one target is required (e.g. prep/sft_packing:tiny).")
    parsed_targets = [
        _merge_target_overrides(_parse_target_spec(target), workflow_overrides) for target in real_targets
    ]
    compiler = AirgapCompiler(repo_root=repo_root)
    lock_data = compiler.compile_many(
        parsed_targets,
        workflow_name=name,
        profiles=profiles or [],
        env_file=env_file,
    )
    write_lock(lock_data, output)

    if json_output:
        console.print(yaml.safe_dump(lock_data, sort_keys=False), soft_wrap=True)
        return

    _print_lock_summary(output, lock_data)


def _looks_like_override(token: str) -> bool:
    """Return True for tokens that look like Hydra dotlist overrides.

    A target is ``step_id[:config][+overrides]`` and always contains a
    ``/`` (the ``category/name`` separator). An override is ``key=value`` (or
    ``+key=value``) and never contains ``/`` before the ``=``. Tokens that
    contain ``=`` but also a ``/`` before it are treated as targets so a
    config name with ``=`` (rare but legal) doesn't get misclassified.
    """
    if "=" not in token:
        return False
    head = token.lstrip("+").split("=", 1)[0]
    return "/" not in head


def _merge_target_overrides(target: AirgapTarget, workflow_overrides: tuple[str, ...]) -> AirgapTarget:
    """Append workflow-wide overrides without losing per-target overrides.

    Per-target overrides come last so they win on key collisions when the
    underlying loader applies them in order (Hydra dotlist semantics).
    """
    if not workflow_overrides:
        return target
    return AirgapTarget(
        step_id=target.step_id,
        config_name=target.config_name,
        overrides=tuple([*workflow_overrides, *target.overrides]),
    )


def fetch_airgap(
    lockfile: Annotated[Path, typer.Argument(help="Path to airgap.lock.yaml.")],
    bundle_dir: Annotated[
        Path,
        typer.Option("-b", "--bundle-dir", help="Directory where fetched assets are written."),
    ] = Path("deploy/nemotron-customizer/airgap/airgap-bundle"),
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", "-d", help="Show planned fetches without downloading."),
    ] = False,
    include_wheels: Annotated[
        bool,
        typer.Option("--include-wheels", help="Build a Python wheelhouse using uv export + pip wheel."),
    ] = False,
    include_repos: Annotated[
        bool,
        typer.Option(
            "--include-repos",
            help="Fetch only git repo assets needed by task-image repo mounts; do not fetch HF models/datasets.",
        ),
    ] = False,
    include_assets: Annotated[
        bool,
        typer.Option(
            "--include-assets/--no-assets",
            help=(
                "Fetch large HF/git/url assets into bundle/assets. Use this for local smoke tests or "
                "transfer bundles; remote execution should normally stage assets directly on persistent storage."
            ),
        ),
    ] = False,
    tighten_lock: Annotated[
        bool,
        typer.Option(
            "--tighten-lock/--no-tighten-lock",
            help=(
                "After fetch, rewrite the lockfile pinning HF/git revisions and URL hashes. "
                "With --no-tighten-lock, fetched content hashes are not saved back."
            ),
        ),
    ] = True,
) -> None:
    """Fetch lockfile assets into a portable bundle directory.

    Run this on a connected machine. The resulting bundle can be copied into
    the disconnected environment and mounted or copied into an image.
    """

    lock = read_lock(lockfile)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir = bundle_dir / AIRGAP_RUNTIME_DIR
    assets_dir = bundle_dir / AIRGAP_ASSETS_DIR
    _write_offline_env(runtime_dir)
    _ensure_asset_skeleton(assets_dir)

    # Fetch helpers mutate the in-memory lock with resolved commits/hashes.
    # ``--no-tighten-lock`` intentionally treats those mutations as temporary
    # probe output and leaves the lockfile unchanged.
    before = yaml.safe_dump(lock, sort_keys=False) if tighten_lock and not dry_run else None
    planned: list[str] = []
    planned.extend(_fetch_support_assets(lock, assets_dir=assets_dir, dry_run=dry_run))
    if include_assets:
        planned.extend(_fetch_assets(lock, assets_dir=assets_dir, dry_run=dry_run))
    else:
        fetched_kinds: set[str] = set()
        if include_repos:
            planned.extend(_fetch_assets(lock, assets_dir=assets_dir, dry_run=dry_run, kinds={"git_repo"}))
            fetched_kinds.add("git_repo")
        skipped = _skipped_external_asset_count(lock, excluding_kinds=fetched_kinds)
        if skipped:
            planned.append(
                f"skipped {skipped} external assets; run `nemotron step airgap plan {lockfile}` "
                "and stage them on the target persistent storage"
            )
    if include_wheels:
        planned.append(
            _fetch_wheels(
                lock,
                runtime_dir=runtime_dir,
                dry_run=dry_run,
                repo_root=_lock_repo_root(lock),
            )
        )

    if before is not None:
        after = yaml.safe_dump(lock, sort_keys=False)
        if after != before:
            write_lock(lock, lockfile)
            console.print(f"[green]Tightened revisions in[/green] {lockfile}")

    _print_fetch_summary(planned, dry_run=dry_run)


def build_airgap(
    lockfile: Annotated[Path, typer.Argument(help="Path to airgap.lock.yaml.")],
    dockerfile: Annotated[
        Path,
        typer.Option("-f", "--dockerfile", help="Checked-in Dockerfile to use for the image build."),
    ] = Path("deploy/nemotron-customizer/airgap/Dockerfile"),
    bundle_dir_name: Annotated[
        str,
        typer.Option(
            "--bundle-dir-name",
            help="Bundle directory name relative to the Docker build context. Only its runtime/ subdir is baked.",
        ),
    ] = "deploy/nemotron-customizer/airgap/airgap-bundle",
    base_image: Annotated[
        str | None,
        typer.Option("--base-image", help="Override BASE_IMAGE instead of using the first lock base image."),
    ] = None,
    python_bin: Annotated[
        str,
        typer.Option("--python-bin", help="Python executable inside BASE_IMAGE used to create the venv."),
    ] = "python",
    image: Annotated[str | None, typer.Option("-t", "--tag", help="Optional docker image tag to build.")] = None,
    execute: Annotated[bool, typer.Option("--execute", help="Run docker build after printing the command.")] = False,
) -> None:
    """Print a docker build command for the checked-in airgap Dockerfile."""

    lock = read_lock(lockfile)
    if not dockerfile.exists():
        raise typer.BadParameter(f"Dockerfile does not exist: {dockerfile}")

    selected_base_image = base_image or _default_base_image(lock)
    _print_base_image_warning(lock, selected_base_image)
    cmd = [
        "docker",
        "build",
        "-f",
        str(dockerfile),
        "--build-arg",
        f"BASE_IMAGE={selected_base_image}",
        "--build-arg",
        f"AIRGAP_BUNDLE={bundle_dir_name}",
        "--build-arg",
        f"PYTHON_BIN={python_bin}",
        "--build-arg",
        f"UV_VERSION={UV_VERSION}",
    ]
    if image:
        cmd.extend(["-t", image])
    cmd.append(".")
    _print_shell_command(cmd)

    if execute:
        raise typer.Exit(subprocess.run(cmd, check=False).returncode)


def verify_airgap(
    lockfile: Annotated[Path, typer.Argument(help="Path to airgap.lock.yaml.")],
    bundle_dir: Annotated[
        Path | None,
        typer.Option("-b", "--bundle-dir", help="Optional fetched bundle directory to inspect."),
    ] = None,
    strict: Annotated[bool, typer.Option("--strict", help="Treat warnings as errors.")] = False,
) -> None:
    """Verify lockfile pinning, manual inputs, and optional bundle contents."""

    lock = read_lock(lockfile)
    issues = verify_lock(lock, strict=strict, bundle_dir=bundle_dir)
    _print_issues(issues)
    if any(issue.severity == "error" for issue in issues):
        raise typer.Exit(1)


def plan_airgap(
    lockfile: Annotated[Path, typer.Argument(help="Path to airgap.lock.yaml.")],
    json_output: Annotated[bool, typer.Option("--json", help="Print the delivery plan as YAML.")] = False,
) -> None:
    """Show what to download, stage remotely, mount, and ask the customer to provide."""

    plan = build_delivery_plan(read_lock(lockfile))
    if json_output:
        console.print(yaml.safe_dump(plan, sort_keys=False), soft_wrap=True)
        return
    _print_delivery_plan(plan)


def probe_task_images(
    lockfile: Annotated[Path, typer.Argument(help="Path to airgap.lock.yaml.")],
    output: Annotated[
        Path | None,
        typer.Option("-o", "--output", help="Optional YAML file for the probe/gap report."),
    ] = None,
    execute: Annotated[
        bool,
        typer.Option("--execute", help="Run docker import probes. Without this, only print the planned probes."),
    ] = False,
    python_bin: Annotated[
        str,
        typer.Option("--python-bin", help="Python executable inside each task image."),
    ] = "python",
    requirements_dir: Annotated[
        Path | None,
        typer.Option(
            "--requirements-dir",
            help="When --execute is set, write one task requirements file per image with missing package hints.",
        ),
    ] = None,
    repo_mounts_output: Annotated[
        Path | None,
        typer.Option(
            "--repo-mounts-output",
            help="Write a JSON repo mount map usable by build-task-image --task-repo-mounts.",
        ),
    ] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Print the probe report as YAML.")] = False,
) -> None:
    """Probe remote task images for the imports needed by selected steps.

    The probe is intentionally import-only. It does not run training; it answers
    the practical airgap question: which small Python packages does this exact
    executor image still need before it can import the Nemotron step wrapper?
    """

    report = _build_task_probe_report(
        read_lock(lockfile),
        execute=execute,
        python_bin=python_bin,
        requirements_dir=requirements_dir,
    )
    if repo_mounts_output is not None:
        mounts = _repo_mounts_from_probe_report(report)
        repo_mounts_output.parent.mkdir(parents=True, exist_ok=True)
        repo_mounts_output.write_text(json.dumps(mounts, indent=2) + "\n", encoding="utf-8")
        console.print(f"[green]Wrote[/green] {repo_mounts_output}")
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(yaml.safe_dump(report, sort_keys=False), encoding="utf-8")
        console.print(f"[green]Wrote[/green] {output}")
    if json_output:
        console.print(yaml.safe_dump(report, sort_keys=False), soft_wrap=True)
        return
    _print_task_probe_report(report, execute=execute)


def build_task_image(
    lockfile: Annotated[Path, typer.Argument(help="Path to airgap.lock.yaml.")],
    dockerfile: Annotated[
        Path,
        typer.Option("-f", "--dockerfile", help="Checked-in task-image Dockerfile to use."),
    ] = Path("deploy/nemotron-customizer/airgap/Dockerfile.task"),
    bundle_dir_name: Annotated[
        str,
        typer.Option(
            "--bundle-dir-name",
            help="Bundle directory name relative to the Docker build context.",
        ),
    ] = "deploy/nemotron-customizer/airgap/airgap-bundle",
    task_requirements: Annotated[
        Path,
        typer.Option(
            "--task-requirements",
            help="Pip requirements file for small missing task-image packages.",
        ),
    ] = Path("deploy/nemotron-customizer/airgap/task-requirements.txt"),
    task_repo_mounts: Annotated[
        Path,
        typer.Option(
            "--task-repo-mounts",
            help="JSON repo mount map baked from assets/repos into target paths.",
        ),
    ] = Path("deploy/nemotron-customizer/airgap/task-repo-mounts.json"),
    step_id: Annotated[
        str | None,
        typer.Option("--step-id", help="Pick the task image for this step when a workflow has several images."),
    ] = None,
    base_image: Annotated[
        str | None,
        typer.Option("--base-image", help="Override BASE_IMAGE instead of using task readiness metadata."),
    ] = None,
    python_bin: Annotated[
        str,
        typer.Option("--python-bin", help="Python executable inside BASE_IMAGE used for pip installs/import checks."),
    ] = "python",
    image: Annotated[str | None, typer.Option("-t", "--tag", help="Optional docker image tag to build.")] = None,
    execute: Annotated[bool, typer.Option("--execute", help="Run docker build after printing the command.")] = False,
) -> None:
    """Print or run a derivative task-image build command.

    Use this for production task images after ``probe-task-images`` identifies
    the tiny package gap. The image remains task-specific; the submitter image
    still only submits jobs.
    """

    lock = read_lock(lockfile)
    if not dockerfile.exists():
        raise typer.BadParameter(f"Dockerfile does not exist: {dockerfile}")
    if not task_requirements.exists():
        raise typer.BadParameter(f"Task requirements file does not exist: {task_requirements}")
    _ensure_task_repo_mounts(task_repo_mounts)

    selected_base_image = base_image or _default_task_image(lock, step_id=step_id)
    _print_task_image_warning(lock, selected_base_image, step_id=step_id)
    cmd = [
        "docker",
        "build",
        "-f",
        str(dockerfile),
        "--build-arg",
        f"BASE_IMAGE={selected_base_image}",
        "--build-arg",
        f"AIRGAP_BUNDLE={bundle_dir_name}",
        "--build-arg",
        f"TASK_REQUIREMENTS={task_requirements.as_posix()}",
        "--build-arg",
        f"TASK_REPO_MOUNTS={task_repo_mounts.as_posix()}",
        "--build-arg",
        f"PYTHON_BIN={python_bin}",
    ]
    if image:
        cmd.extend(["-t", image])
    cmd.append(".")
    _print_shell_command(cmd)

    if execute:
        raise typer.Exit(subprocess.run(cmd, check=False).returncode)


def _ensure_task_repo_mounts(path: Path) -> None:
    """Materialize an empty repo mount map when no repo mounts are needed."""

    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("[]\n", encoding="utf-8")
    console.print(f"[yellow]Task repo mounts file missing; wrote empty mount map:[/yellow] {path}")


def _ensure_asset_skeleton(assets_dir: Path) -> None:
    """Create small optional asset directories used by task-image builds."""

    for rel in ("repos", "lepton", "hf-cache/hub"):
        (assets_dir / rel).mkdir(parents=True, exist_ok=True)


def _print_shell_command(cmd: list[str]) -> None:
    """Print copy-pasteable shell commands without Rich line wrapping."""

    typer.echo("$ " + " ".join(shlex_quote(part) for part in cmd))


def _lock_task_containers(lock: Mapping[str, Any], *, step_id: str | None = None) -> list[dict[str, Any]]:
    runtime = lock.get("runtime", {}) or {}
    out: list[dict[str, Any]] = []
    for item in runtime.get("task_containers", []) or []:
        if not isinstance(item, Mapping):
            continue
        data = dict(item)
        if step_id and data.get("step_id") != step_id:
            continue
        if data.get("image"):
            out.append(data)
    return out


def _default_task_image(lock: Mapping[str, Any], *, step_id: str | None = None) -> str:
    task_containers = _lock_task_containers(lock, step_id=step_id)
    if task_containers:
        return str(task_containers[0]["image"])
    return _default_base_image(dict(lock))


def _print_task_image_warning(lock: Mapping[str, Any], selected_base_image: str, *, step_id: str | None) -> None:
    image_ids = sorted({str(item["image"]) for item in _lock_task_containers(lock, step_id=step_id)})
    if len(image_ids) <= 1:
        return
    scope = f" for step {step_id!r}" if step_id else ""
    console.print(
        "[yellow]Lock references multiple task images"
        f"{scope}. Using BASE_IMAGE={selected_base_image!r}; pass --base-image or --step-id to override.[/yellow]"
    )


def _build_task_probe_report(
    lock: Mapping[str, Any],
    *,
    execute: bool,
    python_bin: str,
    requirements_dir: Path | None,
) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for task in _lock_task_containers(lock):
        imports = sorted({str(item) for item in task.get("candidate_imports", []) or [] if str(item)})
        package_hints = [str(item) for item in task.get("package_hints", []) or [] if str(item)]
        result: dict[str, Any] = {
            "step_id": str(task.get("step_id") or ""),
            "image": str(task.get("image") or ""),
            "candidate_imports": imports,
            "package_hints": package_hints,
            "repo_mounts": list(task.get("repo_mounts", []) or []),
            "status": "planned",
            "missing_imports": [],
            "missing_packages": [],
            "probe_command": _docker_probe_command(str(task.get("image") or ""), imports, python_bin),
        }
        if execute:
            missing = _run_task_import_probe(str(task.get("image") or ""), imports, python_bin)
            result["status"] = "ok" if not missing else "missing_imports"
            result["missing_imports"] = missing
            result["missing_packages"] = _packages_for_missing_imports(missing, package_hints)
            if requirements_dir is not None:
                requirements_path = _write_task_requirements(
                    requirements_dir,
                    step_id=str(task.get("step_id") or "step"),
                    image=str(task.get("image") or "image"),
                    packages=result["missing_packages"],
                )
                result["requirements_file"] = str(requirements_path)
        items.append(result)
    return {
        "schema_version": 1,
        "mode": "executed" if execute else "planned",
        "tasks": items,
        "note": (
            "Use missing_packages to build a derivative task image. For production, set the executor "
            "profile to that derivative image and use pip_install_mode=preinstalled."
        ),
    }


def _docker_probe_command(image: str, imports: list[str], python_bin: str) -> list[str]:
    return ["docker", "run", "--rm", "--pull", "never", image, python_bin, "-c", _task_import_probe_code(imports)]


def _task_import_probe_code(imports: list[str]) -> str:
    return (
        "import importlib.util,json;"
        f"mods={imports!r};"
        "missing=[m for m in mods if importlib.util.find_spec(m) is None];"
        "print(json.dumps({'missing_imports': missing}))"
    )


def _run_task_import_probe(image: str, imports: list[str], python_bin: str) -> list[str]:
    if not imports:
        return []
    cmd = _docker_probe_command(image, imports, python_bin)
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        console.print(result.stderr or result.stdout, style="red")
        raise typer.Exit(result.returncode)
    payload = json.loads(result.stdout.strip() or "{}")
    missing = payload.get("missing_imports") or []
    if not isinstance(missing, list):
        return []
    return [str(item) for item in missing]


def _packages_for_missing_imports(missing_imports: list[str], package_hints: list[str]) -> list[str]:
    by_import: dict[str, str] = {}
    for package in package_hints:
        import_name = _package_to_import(package)
        if import_name and import_name not in by_import:
            by_import[import_name] = package
    return [by_import.get(import_name, import_name.replace("_", "-")) for import_name in missing_imports]


def _package_to_import(package: str) -> str | None:
    name = package.strip()
    if not name:
        return None
    for marker in ("==", ">=", "<=", "~=", "!=", ">", "<", ";"):
        if marker in name:
            name = name.split(marker, 1)[0]
    name = name.split("[", 1)[0].strip()
    aliases = {
        "cosmos-xenna": "cosmos_xenna",
        "pydantic-settings": "pydantic_settings",
        "huggingface-hub": "huggingface_hub",
        "pyyaml": "yaml",
    }
    return aliases.get(name, name.replace("-", "_")) if name else None


def _write_task_requirements(
    requirements_dir: Path,
    *,
    step_id: str,
    image: str,
    packages: list[str],
) -> Path:
    requirements_dir.mkdir(parents=True, exist_ok=True)
    suffix = hashlib.sha256(image.encode("utf-8")).hexdigest()[:8]
    filename = f"task-requirements-{_sanitize_filename(step_id)}-{suffix}.txt"
    path = requirements_dir / filename
    path.write_text("\n".join(packages) + ("\n" if packages else ""), encoding="utf-8")
    return path


def _repo_mounts_from_probe_report(report: Mapping[str, Any]) -> list[dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, str]] = []
    for task in report.get("tasks", []) or []:
        if not isinstance(task, Mapping):
            continue
        for item in task.get("repo_mounts", []) or []:
            if not isinstance(item, Mapping):
                continue
            repo = str(item.get("repo") or "").strip()
            target = str(item.get("target") or "").strip()
            if not repo or not target or (repo, target) in seen:
                continue
            seen.add((repo, target))
            out.append({"repo": repo, "target": target})
    return sorted(out, key=lambda item: (item["repo"], item["target"]))


def _sanitize_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "task"


def _print_task_probe_report(report: dict[str, Any], *, execute: bool) -> None:
    rows = report.get("tasks", []) or []
    if not rows:
        console.print("[yellow]No task containers were recorded in the lock.[/yellow]")
        return
    table = Table(title="Task image probe" if execute else "Task image probe plan")
    table.add_column("Step")
    table.add_column("Image", overflow="fold")
    table.add_column("Status")
    table.add_column("Missing imports", overflow="fold")
    table.add_column("Missing packages", overflow="fold")
    for row in rows:
        table.add_row(
            str(row.get("step_id") or "-"),
            str(row.get("image") or "-"),
            str(row.get("status") or "-"),
            ", ".join(row.get("missing_imports") or []) or "-",
            ", ".join(row.get("missing_packages") or []) or "-",
        )
    console.print(table)


def _print_lock_summary(path: Path, lock: dict[str, Any]) -> None:
    steps = lock.get("steps", []) or []
    executors = lock.get("executors", []) or []
    assets = lock.get("assets", []) or []
    services = lock.get("services", []) or []
    manual = lock.get("manual_inputs", []) or []
    unresolved = lock.get("unresolved_env", []) or []
    issues = lock.get("issues", []) or []

    console.print(f"[green]Wrote[/green] {path}")
    table = Table(title=f"Airgap lock: {lock.get('step', {}).get('id')}")
    table.add_column("Section")
    table.add_column("Count", justify="right")
    if steps:
        table.add_row("steps", str(len(steps)))
    if executors:
        table.add_row("executors", str(len(executors)))
    table.add_row("assets", str(len(assets)))
    table.add_row("services", str(len(services)))
    table.add_row("manual inputs", str(len(manual)))
    table.add_row("unresolved env", str(len(unresolved)))
    table.add_row("issues", str(len(issues)))
    console.print(table)


def _print_delivery_plan(plan: dict[str, Any]) -> None:
    locations = plan.get("asset_locations", {}) or {}
    execution = plan.get("execution", {}) or {}
    console.print("[bold]Airgap Delivery Plan[/bold]")
    console.print(f"Execution mode: {execution.get('mode', '-')}")
    console.print(f"Local runtime image input: {plan.get('runtime_image', {}).get('build_input', '-')}")
    console.print(f"Connected asset staging: {locations.get('connected_staging_dir', '-')}")
    console.print(f"Remote asset root: {locations.get('remote_persistent_root', '-')}")

    _print_plan_table(
        "Stages",
        plan.get("stages", []) or [],
        ("Stage", "Where", "Output", "Rule"),
        ("stage", "where", "output", "rule"),
    )
    _print_plan_table(
        "Download on connected machine",
        plan.get("download_assets", []) or [],
        ("Kind", "ID", "Revision", "Bundle path"),
        ("kind", "id", "revision", "bundle_path"),
    )
    _print_plan_table(
        "Runtime-resolved assets",
        plan.get("runtime_assets", []) or [],
        ("Kind", "ID", "Revision", "Action"),
        ("kind", "id", "revision", "customer_action"),
    )
    _print_plan_table(
        "Container images",
        plan.get("container_images", []) or [],
        ("Kind", "ID", "Action"),
        ("kind", "id", "customer_action"),
    )
    _print_plan_table(
        "Remote task container readiness",
        plan.get("task_containers", []) or [],
        ("Step", "Image", "Probe action", "Production action"),
        ("step_id", "image", "probe_action", "production_action"),
    )
    _print_plan_table(
        "Standard mounts",
        plan.get("standard_mounts", []) or [],
        ("ID", "Scope", "Host path", "Container path", "Mode"),
        ("id", "scope", "host_path", "container_path", "mode"),
    )
    _print_plan_table(
        "Customer-provided paths",
        plan.get("customer_inputs", []) or [],
        ("Kind", "ID", "Source", "Action"),
        ("kind", "id", "source", "customer_action"),
    )
    _print_plan_table(
        "In-network services",
        plan.get("services", []) or [],
        ("Kind", "ID", "Source", "Action"),
        ("kind", "id", "source", "customer_action"),
    )
    _print_plan_table(
        "Environment variables",
        plan.get("environment", []) or [],
        ("Kind", "ID", "Source", "Action"),
        ("kind", "id", "source", "customer_action"),
    )


def _print_plan_table(
    title: str,
    rows: list[dict[str, Any]],
    headers: tuple[str, ...],
    keys: tuple[str, ...],
) -> None:
    if not rows:
        return
    table = Table(title=title)
    for header in headers:
        table.add_column(header, overflow="fold")
    for row in rows:
        table.add_row(*(str(row.get(key) or "-") for key in keys))
    console.print(table)


def _print_fetch_summary(items: list[str], *, dry_run: bool) -> None:
    if not items:
        console.print("[yellow]No fetchable assets found.[/yellow]")
        return
    title = "Fetch plan" if dry_run else "Fetched assets"
    table = Table(title=title)
    table.add_column("Result")
    for item in items:
        table.add_row(item)
    console.print(table)


def _print_issues(issues: list[Any]) -> None:  # list[AirgapIssue]; loose for legacy callers
    if not issues:
        console.print("[green]Airgap lock looks good.[/green]")
        return
    table = Table(title="Airgap verification")
    table.add_column("Severity")
    table.add_column("Code")
    table.add_column("Source")
    table.add_column("Message", overflow="fold")
    for issue in issues:
        table.add_row(issue.severity, issue.code, issue.source or "-", issue.message)
    console.print(table)


def _default_base_image(lock: dict[str, Any]) -> str:
    base_images = list((lock.get("runtime", {}) or {}).get("base_images", []) or [])
    for image in base_images:
        if isinstance(image, dict) and image.get("id"):
            return str(image["id"])
    return "python:3.12-slim"


def _print_base_image_warning(lock: dict[str, Any], selected_base_image: str) -> None:
    image_ids = [
        str(image["id"])
        for image in (lock.get("runtime", {}) or {}).get("base_images", []) or []
        if isinstance(image, dict) and image.get("id")
    ]
    unique_ids = sorted(set(image_ids))
    if len(unique_ids) <= 1:
        return
    console.print(
        "[yellow]Lock references multiple base images. "
        f"Using BASE_IMAGE={selected_base_image!r}; pass --base-image to override.[/yellow]"
    )


def _write_offline_env(runtime_dir: Path) -> None:
    runtime_dir.mkdir(parents=True, exist_ok=True)
    env_path = runtime_dir / "offline.env"
    env_path.write_text("\n".join(f"{key}={value}" for key, value in OFFLINE_ENV.items()) + "\n", encoding="utf-8")


def _fetch_assets(
    lock: dict[str, Any],
    *,
    assets_dir: Path,
    dry_run: bool,
    kinds: set[str] | None = None,
) -> list[str]:
    out: list[str] = []
    for asset in lock.get("assets", []) or []:
        if not isinstance(asset, dict):
            continue
        if _is_support_url_asset(asset):
            continue
        if asset.get("delivery") in {"manual", "runtime"}:
            continue
        kind = asset.get("kind")
        if kinds is not None and kind not in kinds:
            continue
        if kind in {"hf_model", "hf_dataset"}:
            out.append(_fetch_hf(asset, assets_dir=assets_dir, dry_run=dry_run))
        elif kind in {"git_repo", "python_git"}:
            out.append(_fetch_git(asset, assets_dir=assets_dir, dry_run=dry_run))
        elif kind == "url":
            out.append(_fetch_url(asset, assets_dir=assets_dir, dry_run=dry_run))
    return out


def _fetch_support_assets(lock: dict[str, Any], *, assets_dir: Path, dry_run: bool) -> list[str]:
    """Fetch small runtime support assets that should not require all assets."""
    out: list[str] = []
    for asset in lock.get("assets", []) or []:
        if isinstance(asset, dict) and _is_support_url_asset(asset):
            out.append(_fetch_url(asset, assets_dir=assets_dir, dry_run=dry_run))
    return out


def _is_support_url_asset(asset: dict[str, Any]) -> bool:
    return asset.get("kind") == "url" and asset.get("bundle_path") == LEPTON_INIT_SCRIPT_BUNDLE_PATH


def _skipped_external_asset_count(lock: dict[str, Any], *, excluding_kinds: set[str] | None = None) -> int:
    excluding_kinds = excluding_kinds or set()
    return sum(
        1
        for asset in lock.get("assets", []) or []
        if isinstance(asset, dict)
        and asset.get("delivery") == "external"
        and asset.get("kind") not in excluding_kinds
        and not _is_support_url_asset(asset)
    )


def _fetch_hf(asset: dict[str, Any], *, assets_dir: Path, dry_run: bool) -> str:
    repo_id = str(asset["id"])
    repo_type = str(asset.get("repo_type") or ("dataset" if asset.get("kind") == "hf_dataset" else "model"))
    revision = asset.get("revision")
    prefix = "datasets" if repo_type == "dataset" else "models"
    repo_cache = assets_dir / "hf-cache" / "hub" / f"{prefix}--{repo_id.replace('/', '--')}"
    cache_dir = assets_dir / "hf-cache" / "hub"
    if dry_run:
        return f"hf {repo_type}:{repo_id} -> {repo_cache}"

    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for HF asset fetching") from exc

    cache_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        cache_dir=str(cache_dir),
    )
    # Best-effort lock-tighten: rewrite floating refs (None / "main" / branch / tag)
    # to the resolved commit SHA so re-fetches and customer audits are reproducible.
    try:
        info = HfApi().repo_info(repo_id=repo_id, repo_type=repo_type, revision=revision)
        sha = getattr(info, "sha", None)
        if isinstance(sha, str) and sha:
            asset["revision"] = sha
    except Exception:  # noqa: BLE001 - lock-tighten is best-effort.
        pass
    asset["delivery"] = asset.get("delivery") or "external"
    asset["bundle_path"] = f"{AIRGAP_ASSETS_DIR}/hf-cache/hub/{prefix}--{repo_id.replace('/', '--')}"
    return f"hf {repo_type}:{repo_id} -> {repo_cache}"


def _fetch_git(asset: dict[str, Any], *, assets_dir: Path, dry_run: bool) -> str:
    """Fetch a git asset and record the resolved revision in ``asset``.

    The caller only persists that resolved revision when lock tightening is
    enabled. ``--no-tighten-lock`` keeps the fetch useful while preserving the
    input lock exactly as provided.
    """

    url = str(asset.get("note") or asset.get("url") or "")
    if not url:
        raise ValueError(f"Git asset {asset.get('id')} is missing source URL in note")
    ref = str(asset.get("revision") or "HEAD")
    repo_dir = assets_dir / "repos" / str(asset["id"])
    if dry_run:
        return f"git {url}@{ref} -> {repo_dir}"
    # Cheap idempotency check: when the repo already sits at exactly the
    # requested commit SHA, skip the fetch (saves bandwidth on retries).
    if (
        repo_dir.exists()
        and len(ref) == 40
        and all(c in "0123456789abcdefABCDEF" for c in ref)
    ):
        existing = _git_head_sha(repo_dir)
        if existing and existing.lower() == ref.lower():
            asset["bundle_path"] = f"{AIRGAP_ASSETS_DIR}/repos/{asset['id']}"
            return f"git {url}@{ref[:8]} -> {repo_dir} (cached)"

    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    if repo_dir.exists():
        # Customers commonly rewrite ``note`` to a customer-mirrored URL after
        # the bundle is first staged. Without re-pointing ``origin`` we silently
        # keep pulling from the old upstream, which the airgap delivery
        # contract is supposed to forbid.
        subprocess.run(
            ["git", "-C", str(repo_dir), "remote", "set-url", "origin", url],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(repo_dir), "fetch", "--all", "--tags", "--force", "--prune"],
            check=True,
        )
    else:
        subprocess.run(["git", "clone", url, str(repo_dir)], check=True)
    # Plain `git checkout <ref>` does not advance a pre-existing local branch
    # to the just-fetched tip. Resolve to origin/<ref> when ref is a branch,
    # then detach + hard-reset so the working tree always lands on the fetched
    # commit regardless of stale local branches.
    target = _git_resolve_target(repo_dir, ref)
    subprocess.run(["git", "-C", str(repo_dir), "checkout", "--detach", target], check=True)
    subprocess.run(["git", "-C", str(repo_dir), "reset", "--hard", target], check=True)
    resolved = _git_head_sha(repo_dir)
    if resolved:
        asset["revision"] = resolved
    asset["bundle_path"] = f"{AIRGAP_ASSETS_DIR}/repos/{asset['id']}"
    return f"git {url}@{ref} -> {repo_dir}"


def _git_resolve_target(repo_dir: Path, ref: str) -> str:
    """Return ``origin/<ref>`` if it exists (branch refs), else the original ref."""
    res = subprocess.run(
        ["git", "-C", str(repo_dir), "rev-parse", "--verify", "--quiet", f"origin/{ref}"],
        capture_output=True,
    )
    return f"origin/{ref}" if res.returncode == 0 else ref


def _git_head_sha(repo_dir: Path) -> str | None:
    res = subprocess.run(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    sha = res.stdout.strip() if res.returncode == 0 else ""
    return sha or None


_URL_FETCH_TIMEOUT_SECONDS = 60
_URL_FETCH_CHUNK_BYTES = 1024 * 1024


def _fetch_url(asset: dict[str, Any], *, assets_dir: Path, dry_run: bool) -> str:
    url = str(asset["id"])
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"Refusing to fetch non-http(s) url asset: {url!r}")
    filename = Path(parsed.path).name or "download"
    target = _url_asset_target(asset, assets_dir=assets_dir, filename=filename)
    if dry_run:
        return f"url {url} -> {target}"

    target.parent.mkdir(parents=True, exist_ok=True)
    expected = str(asset.get("expected_sha256") or "").strip().lower() or None
    actual_hash = _stream_url_to_file(url, target)
    if expected and actual_hash != expected:
        target.unlink(missing_ok=True)
        raise RuntimeError(
            f"sha256 mismatch for {url}: expected {expected}, got {actual_hash}. "
            "Re-check the lockfile or the upstream source."
        )
    # Always record the resolved sha256 so verify_lock can re-check after the
    # bundle is transferred.
    asset["sha256"] = actual_hash
    if not asset.get("bundle_path"):
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
        asset["bundle_path"] = f"{AIRGAP_ASSETS_DIR}/urls/{digest}"
    return f"url {url} -> {target} (sha256={actual_hash[:12]})"


def _url_asset_target(asset: dict[str, Any], *, assets_dir: Path, filename: str) -> Path:
    """Return the on-disk target for a URL asset.

    ``bundle_path`` is relative to the bundle root. If it names a file, fetch
    exactly there; if it names a directory, place the URL filename inside it.
    The file-vs-directory choice is intentionally suffix-based so the lock can
    stay compact: suffixless paths are directories, paths with suffixes are
    exact file targets.
    """
    bundle_path = str(asset.get("bundle_path") or "").strip()
    if not bundle_path:
        digest = hashlib.sha256(str(asset["id"]).encode("utf-8")).hexdigest()[:16]
        return assets_dir / "urls" / digest / filename

    rel = PurePosixPath(bundle_path)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"Unsafe bundle_path for url asset {asset.get('id')!r}: {bundle_path!r}")
    bundle_root = assets_dir.parent
    target = bundle_root.joinpath(*rel.parts)
    if target.suffix:
        return target
    return target / filename


def _stream_url_to_file(url: str, target: Path) -> str:
    """Download ``url`` into ``target``, returning the sha256 of the contents.

    Uses ``urllib.request`` directly (the deprecated ``urlretrieve`` does not
    expose timeouts and silently follows arbitrary redirects with no integrity
    check). We hash as we stream so we never need a second pass over the file.
    """
    sha = hashlib.sha256()
    request = urllib.request.Request(url, headers={"User-Agent": "nemotron-airgap/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=_URL_FETCH_TIMEOUT_SECONDS) as response:
            with target.open("wb") as handle:
                while True:
                    chunk = response.read(_URL_FETCH_CHUNK_BYTES)
                    if not chunk:
                        break
                    sha.update(chunk)
                    handle.write(chunk)
    except urllib.error.URLError as exc:
        target.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to fetch {url}: {exc}") from exc
    return sha.hexdigest()


def _fetch_wheels(
    lock: dict[str, Any],
    *,
    runtime_dir: Path,
    dry_run: bool,
    repo_root: Path | None = None,
) -> str:
    requirements = runtime_dir / "requirements-airgap.txt"
    source_requirements = runtime_dir / "requirements-airgap.source.txt"
    build_requirements = runtime_dir / "requirements-build-system.txt"
    wheels = runtime_dir / "wheels"
    extras = list(((lock.get("runtime", {}) or {}).get("python", {}) or {}).get("extras", []) or [])
    if dry_run:
        suffix = f" extras={extras}" if extras else ""
        return f"python wheels{suffix} -> {wheels}"
    repo_root = (repo_root or Path.cwd()).resolve()
    uv_cmd = _uv_command()
    export_cmd = [
        *uv_cmd,
        "export",
        "--directory",
        str(repo_root),
        "--format",
        "requirements.txt",
        "--locked",
        "--no-dev",
        "--no-hashes",
        "--no-emit-project",
        "-o",
        str(source_requirements),
    ]
    for extra in extras:
        export_cmd.extend(["--extra", str(extra)])
    runtime_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        export_cmd,
        check=True,
    )
    wheels.mkdir(parents=True, exist_ok=True)
    _write_offline_requirements(source_requirements, requirements, repo_root=repo_root)
    _write_build_requirements(build_requirements, repo_root=repo_root)
    # The offline image consumes wheels only. The connected preparation phase is
    # allowed to build sdist-only packages, such as antlr4-python3-runtime 4.9.3,
    # into wheels before the bundle crosses the airgap.
    with _pip_build_python(uv_cmd) as pip_python:
        subprocess.run(
            [str(pip_python), "-m", "pip", "download", f"uv=={UV_VERSION}", "-d", str(wheels), "--only-binary=:all:"],
            check=True,
        )
        subprocess.run(
            [
                str(pip_python), "-m", "pip", "wheel",
                "-r", str(source_requirements),
                "-w", str(wheels),
            ],
            check=True,
        )
        if build_requirements.exists():
            subprocess.run(
                [
                    str(pip_python), "-m", "pip", "wheel",
                    "-r", str(build_requirements),
                    "-w", str(wheels),
                ],
                check=True,
            )
    suffix = f" extras={extras}" if extras else ""
    return f"python wheels{suffix} -> {wheels}"


@contextmanager
def _pip_build_python(uv_cmd: list[str]) -> Iterator[Path]:
    """Create a temporary pip-capable venv for connected wheelhouse builds."""

    with tempfile.TemporaryDirectory(prefix="nemotron-airgap-pip-") as tmp:
        venv_dir = Path(tmp) / "venv"
        subprocess.run(
            [
                *uv_cmd,
                "venv",
                "--seed",
                "--clear",
                "--python",
                sys.executable,
                str(venv_dir),
            ],
            check=True,
        )
        python = venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        yield python


def _uv_command() -> list[str]:
    candidates = [os.environ.get("UV"), shutil.which("uv")]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            result = subprocess.run(
                [candidate, "--version"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
        except (OSError, subprocess.CalledProcessError):
            continue
        # The wheelhouse the customer eventually consumes ships ``uv==UV_VERSION``;
        # if the producer's uv is far off, the exported requirements may use
        # syntax the airgap-side uv cannot parse. Warn loudly instead of failing
        # so existing CI keeps flowing.
        version_text = (result.stdout or b"").decode("utf-8", "replace").strip()
        version_parts = version_text.split()
        reported_version = version_parts[1] if len(version_parts) > 1 and version_parts[0] == "uv" else version_text
        if reported_version != UV_VERSION:
            console.print(
                f"[yellow][airgap] WARNING: uv {version_text!r} is producing the wheelhouse, "
                f"but the airgap image installs uv=={UV_VERSION}. "
                f"Set UV=/path/to/uv-{UV_VERSION} to suppress this warning.[/yellow]"
            )
        return [candidate]
    raise RuntimeError("uv is required to export locked requirements; run via `uv run ...` or set UV=/path/to/uv")


def _write_offline_requirements(source: Path, output: Path, *, repo_root: Path | None = None) -> None:
    repo_root = (repo_root or Path.cwd()).resolve()
    versions = _locked_package_versions(repo_root / "uv.lock")
    output.write_text(
        _rewrite_direct_refs_to_locked_versions(source.read_text(encoding="utf-8"), versions),
        encoding="utf-8",
    )


def _write_build_requirements(output: Path, *, repo_root: Path | None = None) -> None:
    repo_root = (repo_root or Path.cwd()).resolve()
    pyproject = repo_root / "pyproject.toml"
    if not pyproject.exists():
        return
    with pyproject.open("rb") as handle:
        data = tomllib.load(handle)
    requires = ((data.get("build-system") or {}).get("requires") or [])
    if not requires:
        return
    output.write_text("\n".join(str(requirement) for requirement in requires) + "\n", encoding="utf-8")


def _lock_repo_root(lock: dict[str, Any]) -> Path | None:
    """Return the repo_root recorded in the lock provenance, if usable."""
    provenance = lock.get("provenance") or {}
    if not isinstance(provenance, dict):
        return None
    raw = provenance.get("repo_root")
    if not raw:
        return None
    candidate = Path(str(raw)).expanduser()
    return candidate if candidate.exists() else None


def _locked_package_versions(lock_path: Path) -> dict[str, str]:
    if not lock_path.exists():
        return {}
    with lock_path.open("rb") as handle:
        data = tomllib.load(handle)
    versions: dict[str, str] = {}
    for package in data.get("package", []) or []:
        if not isinstance(package, dict):
            continue
        name = package.get("name")
        version = package.get("version")
        if isinstance(name, str) and isinstance(version, str):
            versions[_normalize_package_name(name)] = version
    return versions


def _rewrite_direct_refs_to_locked_versions(text: str, versions: dict[str, str]) -> str:
    out: list[str] = []
    for line in text.splitlines():
        rewritten = _rewrite_direct_ref_line(line, versions)
        out.append(rewritten)
    return "\n".join(out) + "\n"


def _rewrite_direct_ref_line(line: str, versions: dict[str, str]) -> str:
    stripped = line.strip()
    if " @ " not in stripped or stripped.startswith("#"):
        return line
    requirement, marker = (stripped.split(";", 1) + [""])[:2]
    name, ref = [part.strip() for part in requirement.split(" @ ", 1)]
    if not ref.startswith(DIRECT_REF_PREFIXES):
        return line
    version = versions.get(_normalize_package_name(name))
    if version is None:
        return line
    marker_suffix = f" ; {marker.strip()}" if marker else ""
    return f"{name}=={version}{marker_suffix}"


def _normalize_package_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _parse_target_spec(spec: str) -> AirgapTarget:
    """Parse ``step_id[:config][+key=value[,key=value...]]``.

    Examples:

    - ``sft/automodel`` — step only, default config, no overrides.
    - ``sft/automodel:tiny`` — step + named config.
    - ``sft/automodel:tiny+dataset.repo_id=org/repo,trainer.max_steps=10`` —
      override two config keys for this target.
    """
    spec = spec.strip()
    overrides: tuple[str, ...] = ()
    if "+" in spec:
        spec, _, override_text = spec.partition("+")
        overrides = _split_target_overrides(override_text)
    step_id, sep, config_name = spec.rpartition(":")
    if not sep:
        step_id = spec
        config_name = ""
    step_id = step_id.strip()
    config_name = config_name.strip()
    if not step_id:
        raise typer.BadParameter(f"Invalid target spec {spec!r}; expected step_id or step_id:config")
    return AirgapTarget(step_id=step_id, config_name=config_name or None, overrides=overrides)


def _split_target_overrides(text: str) -> tuple[str, ...]:
    """Split per-target overrides on top-level commas only."""
    items: list[str] = []
    buffer: list[str] = []
    quote_char: str | None = None
    escaped = False
    depth = 0

    for char in text:
        if escaped:
            buffer.append(char)
            escaped = False
            continue
        if quote_char is not None:
            buffer.append(char)
            if char == "\\":
                escaped = True
            elif char == quote_char:
                quote_char = None
            continue
        if char in {"'", '"'}:
            quote_char = char
            buffer.append(char)
            continue
        if char in "[{(":
            depth += 1
            buffer.append(char)
            continue
        if char in "]})":
            if depth > 0:
                depth -= 1
            buffer.append(char)
            continue
        if char == "," and depth == 0:
            item = "".join(buffer).strip()
            if item:
                items.append(item)
            buffer = []
            continue
        buffer.append(char)

    item = "".join(buffer).strip()
    if item:
        items.append(item)
    return tuple(items)


airgap_app.command(
    "lock",
    help="Compile an airgap lockfile for a step/config.",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)(lock_airgap)
airgap_app.command(
    "lock-workflow",
    help="Compile one airgap lockfile for multiple step/config targets.",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)(lock_workflow_airgap)
airgap_app.command("fetch", help="Fetch HF/git/url assets into an airgap bundle.")(fetch_airgap)
airgap_app.command("build", help="Print or run docker build for the checked-in airgap Dockerfile.")(build_airgap)
airgap_app.command(
    "probe-task-images",
    help="Probe task images for imports needed by locked steps.",
)(probe_task_images)
airgap_app.command(
    "build-task-image",
    help="Print or run docker build for a derivative remote task image.",
)(build_task_image)
airgap_app.command("verify", help="Verify an airgap lockfile and optional bundle.")(verify_airgap)
airgap_app.command("plan", help="Show the standardized airgap download, staging, and mount plan.")(plan_airgap)

__all__ = ["airgap_app"]
