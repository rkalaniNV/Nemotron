"""Nemotron BYOB CLI orchestration for BFCL evaluation."""

from __future__ import annotations

import importlib.metadata
from pathlib import Path
from typing import Any, Final, Literal

import yaml  # type: ignore[import-untyped]
from omegaconf import OmegaConf
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictStr,
    ValidationError,
    model_validator,
)

from nemotron.steps.byob.runtime.benchmark_families.base import BenchmarkRunResult
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.config import (
    load_eval_config,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.contamination import (
    evaluate_contamination,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.error_taxonomy import (
    FATAL_EVAL_ERROR_CODES,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.eval_runner import (
    run_declared_eval_sync,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.nemo_native_adapter import (
    NEMO_NATIVE_ADAPTER_VERSION,
    NemoNativeAdapterConfig,
    install_native_framework,
    launcher_task_entry,
    write_native_adapter_config,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.source_verification import (
    verify_eval_source,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.export_contract import (
    export_content_hash,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.nemo_evaluator_export import (
    NEMO_BUNDLE_FILES,
    NEMO_EVALUATOR_ROOT,
)
from nemotron.steps.eval.model_eval.runtime import launch_model_eval_config

BFCL_EVAL_CLI_VERSION: Final = "1.0"


class BfclEvalCliError(RuntimeError):
    """A stable CLI-boundary failure with a process exit status."""

    code = "eval_cli_invalid"
    cli_exit_code = 2

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        cli_exit_code: int | None = None,
    ) -> None:
        super().__init__(message)
        if code is not None:
            self.code = code
        if cli_exit_code is not None:
            self.cli_exit_code = cli_exit_code


class BfclEvalCliRuntimeError(BfclEvalCliError):
    code = "eval_cli_runtime_failed"
    cli_exit_code = 3


class BfclEvalCliArtifactConflictError(BfclEvalCliError):
    code = "eval_cli_artifact_conflict"
    cli_exit_code = 7


class BfclEvalCliFrameworkNotInstalledError(BfclEvalCliError):
    code = "eval_cli_framework_not_installed"
    cli_exit_code = 3


class BfclEvalCliFrameworkVersionMismatchError(BfclEvalCliError):
    code = "eval_cli_framework_version_mismatch"
    cli_exit_code = 3


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class BfclLauncherCliConfig(_Frozen):
    bundle_root: Path | None = None
    native_output_dir: Path
    adapter_config_path: Path
    framework_build_dir: Path
    task_config_path: Path
    launcher_base_config_path: Path
    launcher_config_path: Path
    launcher_output_dir: Path
    dataset_mount_path: StrictStr = "/datasets/bfcl"
    container: StrictStr | None = None
    evaluation_mounts: dict[StrictStr, StrictStr] = Field(default_factory=dict)
    submit: StrictBool = False

    @model_validator(mode="after")
    def _mounts(self) -> BfclLauncherCliConfig:
        for host, container in self.evaluation_mounts.items():
            if not Path(host).is_absolute() or not Path(container).is_absolute():
                raise ValueError("Launcher evaluation mounts must map absolute paths")
        if self.container is not None and not self.evaluation_mounts:
            raise ValueError(
                "a Launcher evaluation container requires explicit evaluation_mounts "
                "for the adapter, eval source, oracle resources, and output trees"
            )
        return self


class BfclEvalCliConfig(_Frozen):
    """Operational envelope kept outside the hash-bearing eval config."""

    schema_version: Literal["1.0"] = BFCL_EVAL_CLI_VERSION
    family: Literal["bfcl"] = "bfcl"
    stage: Literal["eval"] = "eval"
    eval_config_path: Path
    execution_backend: Literal["direct", "nemo_launcher"] = "direct"
    output_format: Literal["human", "json"] = "human"
    probe_oracle: StrictBool = True
    dry_run: StrictBool = False
    launcher: BfclLauncherCliConfig | None = None

    @model_validator(mode="after")
    def _backend_options(self) -> BfclEvalCliConfig:
        if self.execution_backend == "nemo_launcher" and self.launcher is None:
            raise ValueError("nemo_launcher backend requires launcher options")
        if self.execution_backend == "direct" and self.launcher is not None:
            raise ValueError("direct backend may not carry launcher options")
        return self


def _resolved_path(value: Any, *, base: Path, field: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise BfclEvalCliError(f"{field} must be a filesystem path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def load_bfcl_eval_cli_config(path: str | Path) -> BfclEvalCliConfig:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise BfclEvalCliError(f"CLI orchestration config does not exist: {source}")
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise BfclEvalCliError(
            "CLI orchestration config is not valid UTF-8 YAML"
        ) from exc
    if not isinstance(raw, dict):
        raise BfclEvalCliError("CLI orchestration config must contain one object")
    raw["eval_config_path"] = _resolved_path(
        raw.get("eval_config_path"),
        base=source.parent,
        field="eval_config_path",
    )
    launcher = raw.get("launcher")
    if launcher is not None:
        if not isinstance(launcher, dict):
            raise BfclEvalCliError("launcher must contain one object")
        for field in (
            "native_output_dir",
            "adapter_config_path",
            "framework_build_dir",
            "task_config_path",
            "launcher_base_config_path",
            "launcher_config_path",
            "launcher_output_dir",
        ):
            launcher[field] = _resolved_path(
                launcher.get(field),
                base=source.parent,
                field=f"launcher.{field}",
            )
        if launcher.get("bundle_root") is not None:
            launcher["bundle_root"] = _resolved_path(
                launcher["bundle_root"],
                base=source.parent,
                field="launcher.bundle_root",
            )
    try:
        return BfclEvalCliConfig.model_validate(raw)
    except ValidationError as exc:
        raise BfclEvalCliError("CLI orchestration config violates schema 1.0") from exc


def _exit_code(error_code: str) -> int:
    if "contamination" in error_code:
        return 4
    if "candidate" in error_code or "transport" in error_code:
        return 5
    if "oracle" in error_code or "infrastructure" in error_code:
        return 6
    if "artifact" in error_code or "publication" in error_code:
        return 7
    return 3


def _wrap_runtime_failure(exc: Exception) -> BfclEvalCliError:
    exception_code = str(getattr(exc, "code", "eval_cli_runtime_failed"))
    code = (
        exception_code
        if exception_code in FATAL_EVAL_ERROR_CODES
        else "eval_cli_runtime_failed"
    )
    if code == BfclEvalCliRuntimeError.code:
        return BfclEvalCliRuntimeError(str(exc))
    return BfclEvalCliError(str(exc), code=code, cli_exit_code=_exit_code(code))


def _write_immutable_text(path: Path, text: str) -> None:
    payload = text.encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise BfclEvalCliArtifactConflictError(
                f"{path.name} already exists with different orchestration evidence"
            )
        return
    try:
        with path.open("xb") as handle:
            handle.write(payload)
    except FileExistsError:
        if not path.is_file() or path.read_bytes() != payload:
            raise BfclEvalCliArtifactConflictError(
                f"{path.name} appeared concurrently with different evidence"
            )


def _bundle_hash(root: Path) -> str:
    actual = tuple(
        sorted(
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file()
        )
    )
    if actual != tuple(sorted(NEMO_BUNDLE_FILES)):
        raise BfclEvalCliError(
            f"native bundle files differ from the contract: {list(actual)}",
            code="eval_nemo_adapter_invalid",
            cli_exit_code=3,
        )
    try:
        contents = {name: (root / name).read_bytes() for name in NEMO_BUNDLE_FILES}
    except OSError as exc:
        raise BfclEvalCliError(
            "native bundle cannot be read",
            code="eval_nemo_adapter_invalid",
            cli_exit_code=3,
        ) from exc
    return export_content_hash(contents)


def _launcher_document(
    base_path: Path,
    *,
    task: dict[str, Any],
    model_id: str,
    target_url: str,
    api_key_name: str,
    output_dir: Path,
    dry_run: bool,
    evaluation_mounts: dict[str, str],
) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise BfclEvalCliError("Launcher base config is not valid UTF-8 YAML") from exc
    if not isinstance(raw, dict):
        raise BfclEvalCliError("Launcher base config must contain one object")
    raw["dry_run"] = dry_run
    raw["output_dir"] = str(output_dir)
    raw["task_filters"] = [task["name"]]
    execution = raw.setdefault("execution", {})
    execution["output_dir"] = str(output_dir)
    mounts = execution.setdefault("mounts", {})
    mounts.setdefault("evaluation", {}).update(evaluation_mounts)
    raw.setdefault("deployment", {})["type"] = "none"
    endpoint = raw.setdefault("target", {}).setdefault("api_endpoint", {})
    endpoint.update(
        {
            "model_id": model_id,
            "url": target_url,
            "api_key_name": api_key_name,
            "type": "chat",
        }
    )
    raw.setdefault("evaluation", {})["tasks"] = [task]
    return raw


def _run_nemo_launcher(
    cli: BfclEvalCliConfig,
    *,
    config: Any,
) -> BenchmarkRunResult:
    assert cli.launcher is not None
    launcher = cli.launcher
    if len(config.candidates) != 1:
        raise BfclEvalCliError(
            "one NeMo Launcher orchestration config requires exactly one candidate"
        )
    candidate = config.candidates[0]
    bundle_root = (
        launcher.bundle_root
        if launcher.bundle_root is not None
        else (config.source.publication_dir / NEMO_EVALUATOR_ROOT).resolve()
    )
    adapter = NemoNativeAdapterConfig(
        bundle_root=bundle_root,
        bundle_content_hash=_bundle_hash(bundle_root),
        eval_config_path=cli.eval_config_path,
        candidate_alias=candidate.alias,
        native_output_dir=launcher.native_output_dir,
        target_binding="launcher",
    )
    write_native_adapter_config(adapter, launcher.adapter_config_path)
    package = install_native_framework(
        adapter,
        adapter_config_path=launcher.adapter_config_path,
        install_dir=launcher.framework_build_dir,
    )
    task = launcher_task_entry(
        adapter,
        adapter_config_path=launcher.adapter_config_path,
        container=launcher.container,
        dataset_mount_path=launcher.dataset_mount_path,
    )
    _write_immutable_text(
        launcher.task_config_path,
        yaml.safe_dump(task, sort_keys=False, allow_unicode=True),
    )
    launcher_document = _launcher_document(
        launcher.launcher_base_config_path,
        task=task,
        model_id=candidate.model,
        target_url=candidate.api.base_url,
        api_key_name=candidate.api.api_key_env,
        output_dir=launcher.launcher_output_dir,
        dry_run=cli.dry_run,
        evaluation_mounts=dict(launcher.evaluation_mounts),
    )
    _write_immutable_text(
        launcher.launcher_config_path,
        yaml.safe_dump(launcher_document, sort_keys=False, allow_unicode=True),
    )
    payload: dict[str, Any] = {
        "status": "launcher_materialized",
        "candidate": candidate.alias,
        "framework_package": str(package),
        "adapter_config_path": str(launcher.adapter_config_path),
        "task_config_path": str(launcher.task_config_path),
        "launcher_config_path": str(launcher.launcher_config_path),
    }
    if launcher.submit:
        distribution = f"nemo-evaluator-{package.name}"
        try:
            installed = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise BfclEvalCliFrameworkNotInstalledError(
                f"{distribution} is not installed; install the generated package {package}"
            ) from exc
        if installed != NEMO_NATIVE_ADAPTER_VERSION:
            raise BfclEvalCliFrameworkVersionMismatchError(
                f"{distribution} version {installed} differs from "
                f"{NEMO_NATIVE_ADAPTER_VERSION}"
            )
        launch = launch_model_eval_config(
            config_path=launcher.launcher_config_path,
            cfg=OmegaConf.create(launcher_document),
            task_filters=[task["name"]],
        )
        payload.update(
            {
                "status": "launcher_submitted",
                "launcher_invocation_id": launch.invocation_id,
                "submitted_launcher_config_path": str(
                    launch.launcher_config_path
                ),
            }
        )
    return BenchmarkRunResult(
        output_path=launcher.launcher_config_path,
        output_format=cli.output_format,
        payload=payload,
    )


def run_bfcl_eval_cli(config_path: str | Path) -> BenchmarkRunResult:
    """Preflight or execute one declared BFCL evaluation from the Nemotron CLI."""
    cli = load_bfcl_eval_cli_config(config_path)
    try:
        config = load_eval_config(cli.eval_config_path)
        if cli.execution_backend == "nemo_launcher":
            return _run_nemo_launcher(cli, config=config)
        if cli.dry_run:
            source = verify_eval_source(config, probe_oracle=cli.probe_oracle)
            plan = evaluate_contamination(config, source)
            task_counts = {
                alias: len(plan.evaluation_task_ids(alias))
                for alias in plan.candidate_aliases
            }
            payload = {
                "status": "preflight_passed",
                "scope": config.publication_scope,
                "eval_config_hash": config.eval_config_hash,
                "candidates": list(plan.candidate_aliases),
                "task_counts": task_counts,
                "candidate_network_used": False,
                "oracle_probed": cli.probe_oracle,
            }
            return BenchmarkRunResult(
                output_path=config.outputs.output_dir,
                output_format=cli.output_format,
                payload=payload,
            )

        result = run_declared_eval_sync(
            cli.eval_config_path,
            probe_oracle=cli.probe_oracle,
        )
    except BfclEvalCliError:
        raise
    except Exception as exc:
        raise _wrap_runtime_failure(exc) from exc

    payload = {
        "status": "completed",
        "eval_run_id": result.eval_run_id,
        "scope": result.candidate_scores[0].scope,
        "candidates": list(result.plan.candidate_aliases),
        "report_path": str(result.artifacts.report_path),
        "manifest_path": (
            str(result.artifacts.manifest_path)
            if result.artifacts.manifest_path is not None
            else None
        ),
    }
    return BenchmarkRunResult(
        output_path=result.artifacts.report_path,
        output_format=cli.output_format,
        payload=payload,
    )


__all__ = [
    "BFCL_EVAL_CLI_VERSION",
    "BfclEvalCliConfig",
    "BfclEvalCliArtifactConflictError",
    "BfclEvalCliError",
    "BfclEvalCliFrameworkNotInstalledError",
    "BfclEvalCliFrameworkVersionMismatchError",
    "BfclEvalCliRuntimeError",
    "BfclLauncherCliConfig",
    "load_bfcl_eval_cli_config",
    "run_bfcl_eval_cli",
]
