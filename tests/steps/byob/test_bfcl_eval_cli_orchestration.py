"""Regression tests for the Nemotron BFCL evaluation CLI boundary."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from nemotron.steps.byob.runtime.benchmark_families.base import BenchmarkRunResult
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval import (
    cli_orchestration,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.cli_orchestration import (
    BfclEvalCliError,
    load_bfcl_eval_cli_config,
    run_bfcl_eval_cli,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.nemo_evaluator_export import (
    NEMO_BUNDLE_FILES,
)
from nemotron.steps.byob.scripts import run as cli_entry
from nemotron.steps.byob.scripts.runtime import run_byob


def _cli_config(
    tmp_path: Path,
    *,
    dry_run: bool = False,
    output_format: str = "json",
) -> Path:
    path = tmp_path / "eval.cli.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "1.0",
                "family": "bfcl",
                "stage": "eval",
                "eval_config_path": "eval.resolved.yaml",
                "execution_backend": "direct",
                "output_format": output_format,
                "probe_oracle": True,
                "dry_run": dry_run,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def _eval_config(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        publication_scope="trace",
        eval_config_hash="sha256:" + "1" * 64,
        outputs=SimpleNamespace(output_dir=tmp_path / "eval-output"),
    )


def _launcher_cli_config(
    tmp_path: Path,
    *,
    submit: bool = False,
    container: str | None = None,
    evaluation_mounts: dict[str, str] | None = None,
) -> Path:
    bundle = tmp_path / "bundle"
    bundle.mkdir(exist_ok=True)
    for name in NEMO_BUNDLE_FILES:
        (bundle / name).write_text(name, encoding="utf-8")
    base = tmp_path / "launcher.base.yaml"
    base.write_text(
        yaml.safe_dump(
            {
                "execution": {"type": "local", "mode": "sequential"},
                "deployment": {"type": "none"},
                "target": {"api_endpoint": {}},
                "evaluation": {"nemo_evaluator_config": {}, "tasks": []},
            }
        ),
        encoding="utf-8",
    )
    source = tmp_path / "eval.launcher.cli.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "schema_version": "1.0",
                "family": "bfcl",
                "stage": "eval",
                "eval_config_path": "eval.resolved.yaml",
                "execution_backend": "nemo_launcher",
                "output_format": "json",
                "probe_oracle": True,
                "dry_run": False,
                "launcher": {
                    "bundle_root": "bundle",
                    "native_output_dir": "native-output",
                    "adapter_config_path": "native-adapter.yaml",
                    "framework_build_dir": "frameworks",
                    "task_config_path": "launcher-task.yaml",
                    "launcher_base_config_path": "launcher.base.yaml",
                    "launcher_config_path": "launcher.materialized.yaml",
                    "launcher_output_dir": "launcher-output",
                    "dataset_mount_path": "/datasets/bfcl",
                    "container": container,
                    "evaluation_mounts": evaluation_mounts or {},
                    "submit": submit,
                },
            }
        ),
        encoding="utf-8",
    )
    return source


def test_cli_config_resolves_eval_path_without_polluting_eval_identity(
    tmp_path: Path,
) -> None:
    source = _cli_config(tmp_path)

    config = load_bfcl_eval_cli_config(source)

    assert config.eval_config_path == (tmp_path / "eval.resolved.yaml").resolve()
    assert config.execution_backend == "direct"
    assert config.output_format == "json"


def test_cli_dry_run_verifies_source_and_plan_without_candidate_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _cli_config(tmp_path, dry_run=True)
    config = _eval_config(tmp_path)
    plan = SimpleNamespace(
        candidate_aliases=("candidate",),
        evaluation_task_ids=lambda alias: ("t1", "t2"),
    )
    monkeypatch.setattr(cli_orchestration, "load_eval_config", lambda path: config)
    monkeypatch.setattr(
        cli_orchestration,
        "verify_eval_source",
        lambda loaded, probe_oracle: SimpleNamespace(),
    )
    monkeypatch.setattr(
        cli_orchestration,
        "evaluate_contamination",
        lambda loaded, verified: plan,
    )
    monkeypatch.setattr(
        cli_orchestration,
        "run_declared_eval_sync",
        lambda *args, **kwargs: pytest.fail("candidate execution was reached"),
    )

    result = run_bfcl_eval_cli(source)

    assert result.payload["status"] == "preflight_passed"
    assert result.payload["task_counts"] == {"candidate": 2}
    assert result.payload["candidate_network_used"] is False
    assert result.payload["oracle_probed"] is True


def test_cli_direct_backend_returns_machine_readable_artifact_locations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _cli_config(tmp_path)
    config = _eval_config(tmp_path)
    report = tmp_path / "eval-output" / "eval_report.json"
    manifest = tmp_path / "eval-output" / "eval_manifest.json"
    run = SimpleNamespace(
        eval_run_id="eval-run",
        plan=SimpleNamespace(candidate_aliases=("candidate",)),
        candidate_scores=(SimpleNamespace(scope="trace"),),
        artifacts=SimpleNamespace(report_path=report, manifest_path=manifest),
    )
    monkeypatch.setattr(cli_orchestration, "load_eval_config", lambda path: config)
    monkeypatch.setattr(
        cli_orchestration,
        "run_declared_eval_sync",
        lambda path, probe_oracle: run,
    )

    result = run_bfcl_eval_cli(source)

    assert isinstance(result, BenchmarkRunResult)
    assert result.output_path == report
    assert result.payload == {
        "status": "completed",
        "eval_run_id": "eval-run",
        "scope": "trace",
        "candidates": ["candidate"],
        "report_path": str(report),
        "manifest_path": str(manifest),
    }
    assert '"status": "completed"' in result.render()


def test_cli_maps_runtime_taxonomy_to_a_stable_exit_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _cli_config(tmp_path)
    monkeypatch.setattr(
        cli_orchestration,
        "load_eval_config",
        lambda path: _eval_config(tmp_path),
    )

    class CandidateFailureError(RuntimeError):
        code = "eval_candidate_client_invalid"

    monkeypatch.setattr(
        cli_orchestration,
        "run_declared_eval_sync",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            CandidateFailureError("offline")
        ),
    )

    with pytest.raises(BfclEvalCliError) as raised:
        run_bfcl_eval_cli(source)

    assert raised.value.code == "eval_candidate_client_invalid"
    assert raised.value.cli_exit_code == 5


def test_generic_dispatcher_refuses_eval_for_a_family_without_an_eval_hook(
    tmp_path: Path,
) -> None:
    source = _cli_config(tmp_path)

    with pytest.raises(ValueError, match="does not define evaluation"):
        run_byob(config=source, stage="eval", family="mcq")


def test_eval_stage_rejects_generation_resume_controls(tmp_path: Path) -> None:
    source = _cli_config(tmp_path)

    with pytest.raises(ValueError, match="generation-only"):
        run_byob(
            config=source,
            stage="eval",
            family="bfcl",
            skip_until="QUALITY_METRICS",
        )


def test_argparse_entry_maps_dispatch_errors_without_a_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = _cli_config(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "byob-run",
            "--config",
            str(source),
            "--stage",
            "eval",
            "--family",
            "mcq",
        ],
    )

    with pytest.raises(SystemExit) as raised:
        cli_entry.main()

    assert raised.value.code == 2
    assert "byob_stage_unsupported" in capsys.readouterr().err


def test_launcher_backend_materializes_adapter_framework_task_and_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _launcher_cli_config(tmp_path)
    candidate = SimpleNamespace(
        alias="candidate",
        model="model",
        api=SimpleNamespace(
            base_url="https://candidate.example/v1",
            api_key_env="CANDIDATE_API_KEY",
        ),
    )
    config = SimpleNamespace(
        candidates=(candidate,),
        source=SimpleNamespace(publication_dir=tmp_path),
    )
    package = tmp_path / "frameworks" / "bfcl-framework"
    monkeypatch.setattr(cli_orchestration, "load_eval_config", lambda path: config)
    monkeypatch.setattr(
        cli_orchestration,
        "write_native_adapter_config",
        lambda adapter, path: "sha256:" + "1" * 64,
    )
    monkeypatch.setattr(
        cli_orchestration,
        "install_native_framework",
        lambda adapter, adapter_config_path, install_dir: package,
    )
    monkeypatch.setattr(
        cli_orchestration,
        "launcher_task_entry",
        lambda adapter, adapter_config_path, container, dataset_mount_path: {
            "name": "bfcl.task",
            "endpoint_type": "chat",
            "dataset_dir": str(adapter.bundle_root),
            "dataset_mount_path": dataset_mount_path,
        },
    )

    result = run_bfcl_eval_cli(source)
    materialized = yaml.safe_load(
        (tmp_path / "launcher.materialized.yaml").read_text(encoding="utf-8")
    )

    assert result.payload["status"] == "launcher_materialized"
    assert result.payload["framework_package"] == str(package)
    assert materialized["evaluation"]["tasks"][0]["name"] == "bfcl.task"
    assert materialized["target"]["api_endpoint"]["model_id"] == "model"
    assert (
        materialized["target"]["api_endpoint"]["url"]
        == "https://candidate.example/v1"
    )
    assert materialized["task_filters"] == ["bfcl.task"]


def test_launcher_container_requires_and_materializes_explicit_mounts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invalid = _launcher_cli_config(
        tmp_path,
        container="nvcr.io/evaluator@sha256:" + "0" * 64,
    )
    with pytest.raises(BfclEvalCliError, match="violates schema"):
        load_bfcl_eval_cli_config(invalid)

    invalid.unlink()
    source = _launcher_cli_config(
        tmp_path,
        container="nvcr.io/evaluator@sha256:" + "0" * 64,
        evaluation_mounts={"/shared/bfcl": "/shared/bfcl"},
    )
    candidate = SimpleNamespace(
        alias="candidate",
        model="model",
        api=SimpleNamespace(
            base_url="https://candidate.example/v1",
            api_key_env="CANDIDATE_API_KEY",
        ),
    )
    config = SimpleNamespace(
        candidates=(candidate,),
        source=SimpleNamespace(publication_dir=tmp_path),
    )
    monkeypatch.setattr(cli_orchestration, "load_eval_config", lambda path: config)
    monkeypatch.setattr(
        cli_orchestration,
        "write_native_adapter_config",
        lambda adapter, path: "sha256:" + "1" * 64,
    )
    monkeypatch.setattr(
        cli_orchestration,
        "install_native_framework",
        lambda adapter, adapter_config_path, install_dir: (
            tmp_path / "frameworks" / "bfcl-framework"
        ),
    )
    monkeypatch.setattr(
        cli_orchestration,
        "launcher_task_entry",
        lambda adapter, adapter_config_path, container, dataset_mount_path: {
            "name": "bfcl.task",
            "container": container,
            "dataset_dir": str(adapter.bundle_root),
            "dataset_mount_path": dataset_mount_path,
        },
    )

    run_bfcl_eval_cli(source)
    materialized = yaml.safe_load(
        (tmp_path / "launcher.materialized.yaml").read_text(encoding="utf-8")
    )

    assert materialized["execution"]["mounts"]["evaluation"] == {
        "/shared/bfcl": "/shared/bfcl"
    }


def test_launcher_backend_requires_one_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _launcher_cli_config(tmp_path)
    monkeypatch.setattr(
        cli_orchestration,
        "load_eval_config",
        lambda path: SimpleNamespace(candidates=()),
    )

    with pytest.raises(BfclEvalCliError, match="exactly one candidate"):
        run_bfcl_eval_cli(source)


def test_launcher_backend_submits_through_model_eval_api_after_explicit_install(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _launcher_cli_config(tmp_path, submit=True)
    candidate = SimpleNamespace(
        alias="candidate",
        model="model",
        api=SimpleNamespace(
            base_url="https://candidate.example/v1",
            api_key_env="CANDIDATE_API_KEY",
        ),
    )
    config = SimpleNamespace(
        candidates=(candidate,),
        source=SimpleNamespace(publication_dir=tmp_path),
    )
    package = tmp_path / "frameworks" / "bfcl-framework"
    monkeypatch.setattr(cli_orchestration, "load_eval_config", lambda path: config)
    monkeypatch.setattr(
        cli_orchestration,
        "write_native_adapter_config",
        lambda adapter, path: "sha256:" + "1" * 64,
    )
    monkeypatch.setattr(
        cli_orchestration,
        "install_native_framework",
        lambda adapter, adapter_config_path, install_dir: package,
    )
    monkeypatch.setattr(
        cli_orchestration,
        "launcher_task_entry",
        lambda adapter, adapter_config_path, container, dataset_mount_path: {
            "name": "bfcl.task",
            "endpoint_type": "chat",
            "dataset_dir": str(adapter.bundle_root),
            "dataset_mount_path": dataset_mount_path,
        },
    )
    monkeypatch.setattr(
        cli_orchestration.importlib.metadata,
        "version",
        lambda distribution: "1.1",
    )
    monkeypatch.setattr(
        cli_orchestration,
        "launch_model_eval_config",
        lambda config_path, cfg, task_filters: SimpleNamespace(
            invocation_id="invocation",
            launcher_config_path=tmp_path / "submitted.yaml",
        ),
    )

    result = run_bfcl_eval_cli(source)

    assert result.payload["status"] == "launcher_submitted"
    assert result.payload["launcher_invocation_id"] == "invocation"
    assert result.payload["submitted_launcher_config_path"] == str(
        tmp_path / "submitted.yaml"
    )
