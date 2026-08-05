from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from nemotron.steps.byob.runtime.benchmark_families.bfcl.config import BfclConfig
from nemotron.steps.byob.runtime.benchmark_families.registry import list_families

BFCL_CONFIG_DIR = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "nemotron"
    / "steps"
    / "byob"
    / "bfcl"
    / "config"
)
BYOB_DIR = BFCL_CONFIG_DIR.parents[1]


def _copy_tiny_pack(tmp_path: Path) -> Path:
    pack = tmp_path / "pack"
    shutil.copytree(BYOB_DIR / "data" / "tiny_oracle_pack", pack, ignore=shutil.ignore_patterns("__pycache__"))
    return pack


def _write_tiny_config(tmp_path: Path, name: str, **overrides: Any) -> Path:
    config_data = yaml.safe_load((BFCL_CONFIG_DIR / "tiny.yaml").read_text(encoding="utf-8"))
    config_data["output_dir"] = str(tmp_path / "output")
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(config_data.get(key), dict):
            config_data[key].update(value)
        else:
            config_data[key] = value
    path = tmp_path / name
    path.write_text(yaml.safe_dump(config_data), encoding="utf-8")
    return path


def _edit_pack_yaml(path: Path, mutate) -> None:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    mutate(data)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def test_list_families_includes_bfcl_and_mcq() -> None:
    assert list_families() == ["bfcl", "mcq"]


def test_all_bfcl_configs_pin_family() -> None:
    for name in ("tiny.yaml", "default.yaml", "translate.yaml", "banking_vn.yaml"):
        data = yaml.safe_load((BFCL_CONFIG_DIR / name).read_text(encoding="utf-8"))
        assert data["family"] == "bfcl", name


def test_tiny_config_loads_as_smoke(tmp_path: Path) -> None:
    config_data = yaml.safe_load((BFCL_CONFIG_DIR / "tiny.yaml").read_text(encoding="utf-8"))
    config_data["output_dir"] = str(tmp_path / "output")
    temp_config = tmp_path / "tiny.yaml"
    temp_config.write_text(yaml.safe_dump(config_data), encoding="utf-8")

    config = BfclConfig.from_yaml(temp_config)

    assert config.family == "bfcl"
    assert config.stage == "all"
    assert config.lineage.policy == "smoke_no_publication"
    assert config.oracle_runtime.worker == "process"
    assert config.oracle_runtime.tool_timeout_s == 5.0
    assert config.lineage.roles["paraphrase"].enabled is False
    assert config.surface_generation.get("paraphrases_per_template") == 0
    assert config.oracle_pack.manifest_path.name == "manifest.yaml"


def test_default_config_loads() -> None:
    config = BfclConfig.from_yaml(BFCL_CONFIG_DIR / "default.yaml")
    assert config.family == "bfcl"
    assert config.lineage.policy == "strict_separation"


def test_resolved_config_rejects_replacement_tokens(tmp_path: Path) -> None:
    config_data = yaml.safe_load((BFCL_CONFIG_DIR / "default.yaml").read_text(encoding="utf-8"))
    config_data["config_status"] = "resolved"
    config_data["output_dir"] = str(tmp_path / "output")
    path = tmp_path / "resolved.yaml"
    path.write_text(yaml.safe_dump(config_data), encoding="utf-8")

    with pytest.raises(ValueError, match=r"REPLACE_ME_.*lineage\.roles"):
        BfclConfig.from_yaml(path)


def test_rejects_non_bfcl_family(tmp_path: Path) -> None:
    config_data = yaml.safe_load((BFCL_CONFIG_DIR / "tiny.yaml").read_text(encoding="utf-8"))
    config_data["family"] = "mcq"
    config_data["output_dir"] = str(tmp_path / "output")
    bad_path = tmp_path / "bad.yaml"
    bad_path.write_text(yaml.safe_dump(config_data), encoding="utf-8")

    with pytest.raises(ValueError, match="family"):
        BfclConfig.from_yaml(bad_path)


def test_rejects_unknown_config_keys_and_lineage_policies(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="BFCL config has unknown keys: task_generaton"):
        BfclConfig.from_yaml(
            _write_tiny_config(
                tmp_path,
                "unknown-top-level.yaml",
                task_generaton={"tasks_per_category": 999},
            )
        )

    with pytest.raises(ValueError, match="lineage.policy must be one of"):
        BfclConfig.from_yaml(
            _write_tiny_config(
                tmp_path,
                "bad-policy.yaml",
                lineage={"policy": "strict_seperation"},
            )
        )


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"surface_generation": []}, "surface_generation must be a mapping"),
        (
            {"lineage": {"roles": {"profile": {"enabled": "false"}}}},
            "lineage.roles.profile.enabled must be a boolean",
        ),
        (
            {"task_generation": {"tasks_per_category": "4"}},
            "task_generation.tasks_per_category must be an integer",
        ),
    ],
)
def test_config_rejects_values_that_would_be_silently_coerced(
    tmp_path: Path,
    override: dict[str, Any],
    message: str,
) -> None:
    path = _write_tiny_config(
        tmp_path,
        f"bad-type-{len(message)}.yaml",
        **override,
    )

    with pytest.raises(ValueError, match=message):
        BfclConfig.from_yaml(path)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_config_rejects_non_finite_timeouts(tmp_path: Path, value: float) -> None:
    path = _write_tiny_config(
        tmp_path,
        "non-finite-timeout.yaml",
        oracle_runtime={"tool_timeout_s": value},
    )

    with pytest.raises(ValueError, match="must be finite"):
        BfclConfig.from_yaml(path)


def test_config_rejects_output_nested_inside_pack_root(tmp_path: Path) -> None:
    pack = _copy_tiny_pack(tmp_path)
    config = _write_tiny_config(
        tmp_path,
        "nested-output.yaml",
        expt_name="generated",
        output_dir=str(pack),
        oracle_pack={"manifest_path": str(pack / "manifest.yaml")},
        oracle_runtime={"allowed_roots": [str(tmp_path)]},
    )

    with pytest.raises(ValueError, match="outside the oracle pack root"):
        BfclConfig.from_yaml(config)


@pytest.mark.parametrize(
    "expt_name",
    ["../escape", "nested/run", ".", "..", " padded "],
)
def test_config_rejects_an_expt_name_that_is_not_one_directory(
    tmp_path: Path, expt_name: str
) -> None:
    """The run directory names the run, so it may not move the output somewhere else."""
    config = _write_tiny_config(tmp_path, "odd-expt-name.yaml", expt_name=expt_name)

    with pytest.raises(ValueError, match="single directory name"):
        BfclConfig.from_yaml(config)


def test_manifest_reports_replay_apart_from_surface_rejections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A guard-rejected row still replayed, and the counts must let a reader see that."""
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pipeline import (
        generate_bfcl,
        prepare_bfcl,
    )
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages import render

    config = _write_tiny_config(tmp_path, "guard-split.yaml")
    # Validation certifies the pack's own surfaces first; the guard below stands in for
    # an instance whose paraphrase happens to break a guard during generation.
    prepare_bfcl(config)

    real_guards = render.check_surface_guards
    rejected: list[str] = []

    def reject_the_first_surface(template, task, user_texts, tool_names, **kwargs):  # type: ignore[no-untyped-def]
        violations = real_guards(template, task, user_texts, tool_names, **kwargs)
        if not violations and not rejected:
            rejected.append(str(task["task_id"]))
            return [{"guard": "must_preserve", "slot": "stand_in"}]
        return violations

    monkeypatch.setattr(render, "check_surface_guards", reject_the_first_surface)
    benchmark_path = generate_bfcl(config)

    manifest = json.loads(
        (benchmark_path.parent / "run_manifest.json").read_text(encoding="utf-8")
    )
    counts = manifest["stage_counts"]
    assert len(rejected) == 1
    assert counts["replay_passed"] == counts["expanded"]
    assert counts["surface_passed"] == counts["expanded"] - 1
    assert counts["published"] == counts["expanded"] - 1
    assert sum(manifest["surface_guard_rejections"]["by_template"].values()) == 1


def test_tiny_prepare_is_gold_eligible(tmp_path: Path) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pipeline import prepare_bfcl
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.oracle_validation import (
        derive_pack_tier,
    )

    config_data = yaml.safe_load((BFCL_CONFIG_DIR / "tiny.yaml").read_text(encoding="utf-8"))
    config_data["output_dir"] = str(tmp_path / "output")
    temp_config = tmp_path / "tiny.yaml"
    temp_config.write_text(yaml.safe_dump(config_data), encoding="utf-8")

    report_path = prepare_bfcl(temp_config)
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["pack_id"] == "tiny_library"
    assert report["gold_eligible"] is True
    assert report["tier"] == "gold"
    assert derive_pack_tier(report) == (report["gold_eligible"], report["tier"])
    cache = tmp_path / "output" / "bfcl_tiny_library_validation" / "stage_cache"
    assert (cache / "tools_normalized.json").exists()
    assert (cache / "tools_normalized_internal.json").exists()


def test_prepare_rejects_a_template_whose_bound_call_breaks_schema(
    tmp_path: Path,
) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pipeline import prepare_bfcl

    pack = _copy_tiny_pack(tmp_path)
    _edit_pack_yaml(
        pack / "task_templates.yaml",
        lambda templates: templates[0]["assistant_milestones"][0].update(
            {"args": {"unexpected": "value"}}
        ),
    )
    config = _write_tiny_config(
        tmp_path,
        "bad-template-arguments.yaml",
        oracle_pack={"manifest_path": str(pack / "manifest.yaml")},
        oracle_runtime={"allowed_roots": [str(tmp_path)]},
    )

    report = json.loads(prepare_bfcl(config).read_text(encoding="utf-8"))
    contract = next(check for check in report["checks"] if check["id"] == 7)
    assert contract["status"] == "fail"
    assert contract["failures"][0]["reason"] == "representative_trace_schema_mismatch"
    assert report["gold_eligible"] is False


def test_prepare_rejects_a_budget_that_cannot_keep_every_template(tmp_path: Path) -> None:
    """A category budget below its template count fails generation, so gold must see it."""
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pipeline import prepare_bfcl

    pack = _copy_tiny_pack(tmp_path)
    config = _write_tiny_config(
        tmp_path,
        "starved-budget.yaml",
        task_generation={"tasks_per_category": 1},
        oracle_pack={"manifest_path": str(pack / "manifest.yaml")},
        oracle_runtime={"allowed_roots": [str(tmp_path)]},
    )

    report = json.loads(prepare_bfcl(config).read_text(encoding="utf-8"))
    contract = next(check for check in report["checks"] if check["id"] == 7)
    assert contract["status"] == "fail"
    assert contract["failures"][0]["reason"] == "run_contract_failed"
    assert "tasks_per_category" in contract["failures"][0]["detail"]
    assert report["gold_eligible"] is False


def test_prepare_rejects_a_pack_that_cannot_render_its_own_templates(tmp_path: Path) -> None:
    """Render is a hard contract, so a missing text block must not survive to generation."""
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pipeline import prepare_bfcl

    pack = _copy_tiny_pack(tmp_path)
    _edit_pack_yaml(
        pack / "manifest.yaml",
        lambda manifest: manifest["assistant_turn_templates"].pop("final_answer"),
    )
    config = _write_tiny_config(
        tmp_path,
        "unrenderable.yaml",
        oracle_pack={"manifest_path": str(pack / "manifest.yaml")},
        oracle_runtime={"allowed_roots": [str(tmp_path)]},
    )

    report = json.loads(prepare_bfcl(config).read_text(encoding="utf-8"))
    contract = next(check for check in report["checks"] if check["id"] == 7)
    assert contract["status"] == "fail"
    assert contract["failures"][0]["reason"] == "representative_generation_failed"
    assert "final_answer" in contract["failures"][0]["detail"]
    assert report["gold_eligible"] is False


def test_prepare_rejects_a_template_whose_surface_always_breaks_a_guard(tmp_path: Path) -> None:
    """A template that can publish no row is a defect, not an instance-level rejection."""
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pipeline import prepare_bfcl

    pack = _copy_tiny_pack(tmp_path)

    def drop_slot_values(templates: list[dict[str, Any]]) -> None:
        for language in templates[0]["user_turn_templates"]:
            templates[0]["user_turn_templates"][language] = "Please help me with something."

    _edit_pack_yaml(pack / "task_templates.yaml", drop_slot_values)
    config = _write_tiny_config(
        tmp_path,
        "guard-always-fails.yaml",
        oracle_pack={"manifest_path": str(pack / "manifest.yaml")},
        oracle_runtime={"allowed_roots": [str(tmp_path)]},
    )

    report = json.loads(prepare_bfcl(config).read_text(encoding="utf-8"))
    contract = next(check for check in report["checks"] if check["id"] == 7)
    assert contract["status"] == "fail"
    assert contract["failures"][0]["reason"] == "representative_surface_guard_violation"
    assert report["gold_eligible"] is False


def test_prepare_runs_a_representative_templates_assertions(tmp_path: Path) -> None:
    """An importable assertion that always fails must not receive a gold report."""
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pipeline import prepare_bfcl

    pack = _copy_tiny_pack(tmp_path)
    with (pack / "assertions.py").open("a", encoding="utf-8") as handle:
        handle.write(
            "\n\ndef always_fail(*, state, trace, task, ctx):\n"
            "    raise AssertionError('deliberate failure')\n"
            "ASSERTIONS = {name: always_fail for name in ASSERTIONS}\n"
        )
    config = _write_tiny_config(
        tmp_path,
        "failing-assertions.yaml",
        oracle_pack={"manifest_path": str(pack / "manifest.yaml")},
        oracle_runtime={"allowed_roots": [str(tmp_path)]},
    )

    report = json.loads(prepare_bfcl(config).read_text(encoding="utf-8"))
    contract = next(check for check in report["checks"] if check["id"] == 7)
    assert contract["status"] == "fail"
    assert {
        failure["reason"] for failure in contract["failures"]
    } == {"representative_replay_failed"}
    assert report["gold_eligible"] is False


def test_prepare_rejects_a_non_function_tool_envelope(tmp_path: Path) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pipeline import prepare_bfcl

    pack = _copy_tiny_pack(tmp_path)
    tools_path = pack / "tools.json"
    tools = json.loads(tools_path.read_text(encoding="utf-8"))
    tools[0]["type"] = "not-a-function"
    tools_path.write_text(json.dumps(tools), encoding="utf-8")
    config = _write_tiny_config(
        tmp_path,
        "bad-tool-envelope.yaml",
        oracle_pack={"manifest_path": str(pack / "manifest.yaml")},
        oracle_runtime={"allowed_roots": [str(tmp_path)]},
    )

    report = json.loads(prepare_bfcl(config).read_text(encoding="utf-8"))
    alignment = next(check for check in report["checks"] if check["id"] == 3)
    assert alignment["status"] == "fail"
    assert alignment["failures"][0]["reason"] == "tool_type_not_function"


def test_success_coverage_requires_schema_valid_arguments(tmp_path: Path) -> None:
    """A backend accepting hidden arguments must not let those calls prove success."""
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pipeline import prepare_bfcl

    pack = _copy_tiny_pack(tmp_path)

    def add_hidden_argument(cases: list[dict[str, Any]]) -> None:
        success = next(case for case in cases if case["id"] == "success_get_book_status")
        success["arguments"]["backend_only"] = True

    _edit_pack_yaml(pack / "validation_cases.yaml", add_hidden_argument)
    config = _write_tiny_config(
        tmp_path,
        "invalid-success-probe.yaml",
        oracle_pack={"manifest_path": str(pack / "manifest.yaml")},
        oracle_runtime={"allowed_roots": [str(tmp_path)]},
    )

    report = json.loads(prepare_bfcl(config).read_text(encoding="utf-8"))
    coverage = next(check for check in report["checks"] if check["id"] == 5)
    reasons = {failure["reason"] for failure in coverage["failures"]}
    assert "successful_validation_case_schema_mismatch" in reasons
    assert "incomplete_validation_coverage" in reasons


def test_validation_rejects_pack_inputs_that_change_during_import(tmp_path: Path) -> None:
    """The fingerprint must describe one immutable set of inputs throughout validation."""
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pipeline import prepare_bfcl

    pack = _copy_tiny_pack(tmp_path)
    backend = pack / "backend.py"
    source = backend.read_text(encoding="utf-8")
    backend.write_text(
        source.replace(
            "from __future__ import annotations",
            "from __future__ import annotations\n"
            "from pathlib import Path\n"
            "with Path(__file__).with_name('imports.log').open('a') as handle:\n"
            "    handle.write('x')",
            1,
        ),
        encoding="utf-8",
    )
    config = _write_tiny_config(
        tmp_path,
        "mutating-pack.yaml",
        oracle_pack={"manifest_path": str(pack / "manifest.yaml")},
        oracle_runtime={"allowed_roots": [str(tmp_path)]},
    )

    with pytest.raises(RuntimeError, match="changed while it was being validated"):
        prepare_bfcl(config)


def test_failed_rerun_does_not_leave_a_completed_manifest(tmp_path: Path) -> None:
    """A failed invocation must not leave a previous run looking current."""
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pipeline import generate_bfcl

    pack = _copy_tiny_pack(tmp_path)
    config = _write_tiny_config(
        tmp_path,
        "rerun.yaml",
        oracle_pack={"manifest_path": str(pack / "manifest.yaml")},
        oracle_runtime={"allowed_roots": [str(tmp_path)]},
    )
    benchmark = generate_bfcl(config)
    output = benchmark.parent
    assert (output / "run_manifest.json").exists()

    with (pack / "assertions.py").open("a", encoding="utf-8") as handle:
        handle.write(
            "\n\ndef always_fail(*, state, trace, task, ctx):\n"
            "    raise AssertionError('deliberate failure')\n"
            "ASSERTIONS = {name: always_fail for name in ASSERTIONS}\n"
        )
    with pytest.raises(RuntimeError, match="non-gold pack"):
        generate_bfcl(config)

    assert not (output / "run_manifest.json").exists()
    assert not (output / "benchmark.parquet").exists()
    assert not (output / "benchmark_raw.parquet").exists()


def test_prepare_rejects_bad_plans_and_missing_fixture_primary_keys(tmp_path: Path) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pipeline import prepare_bfcl

    plan_root = tmp_path / "bad-plan"
    plan_root.mkdir()
    plan_pack = _copy_tiny_pack(plan_root)
    _edit_pack_yaml(
        plan_pack / "task_templates.yaml",
        lambda templates: templates[0].update(
            {
                "assistant_milestones": [
                    {"type": "ask_confirm"},
                    {"type": "tool_call", "tool": "get_book_status"},
                    {"type": "final_answer"},
                ]
            }
        ),
    )
    plan_config = _write_tiny_config(
        tmp_path,
        "bad-plan.yaml",
        oracle_pack={"manifest_path": str(plan_pack / "manifest.yaml")},
        oracle_runtime={"allowed_roots": [str(tmp_path)]},
    )
    plan_report = json.loads(prepare_bfcl(plan_config).read_text(encoding="utf-8"))
    assert any(
        failure.get("reason") == "invalid_conversation_plan"
        for failure in plan_report["checks"][0]["failures"]
    )

    key_root = tmp_path / "missing-key"
    key_root.mkdir()
    key_pack = _copy_tiny_pack(key_root)
    fixtures_path = key_pack / "fixtures.json"
    fixtures = json.loads(fixtures_path.read_text(encoding="utf-8"))
    fixtures["books"][1].pop("book_id")
    fixtures_path.write_text(json.dumps(fixtures), encoding="utf-8")
    _edit_pack_yaml(
        key_pack / "manifest.yaml",
        lambda manifest: manifest.setdefault("primary_keys", {}).update(
            {"books": "book_id"}
        ),
    )
    _edit_pack_yaml(
        key_pack / "task_templates.yaml",
        lambda templates: templates[0]["slots"]["book_id"].update(
            {"source": "fixture:books.title"}
        ),
    )
    key_config = _write_tiny_config(
        tmp_path,
        "missing-key.yaml",
        oracle_pack={"manifest_path": str(key_pack / "manifest.yaml")},
        oracle_runtime={"allowed_roots": [str(tmp_path)]},
    )
    key_report = json.loads(prepare_bfcl(key_config).read_text(encoding="utf-8"))
    assert any(
        failure.get("reason") == "fixture_row_missing_primary_key"
        for failure in key_report["checks"][1]["failures"]
    )


def test_thread_worker_cannot_claim_gold(tmp_path: Path) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pipeline import prepare_bfcl

    config_data = yaml.safe_load((BFCL_CONFIG_DIR / "tiny.yaml").read_text(encoding="utf-8"))
    config_data["output_dir"] = str(tmp_path / "output")
    config_data["oracle_runtime"]["worker"] = "thread"
    temp_config = tmp_path / "thread.yaml"
    temp_config.write_text(yaml.safe_dump(config_data), encoding="utf-8")

    report = json.loads(prepare_bfcl(temp_config).read_text(encoding="utf-8"))

    assert report["gold_eligible"] is False
    assert report["tier"] == "silver"
    isolation = next(check for check in report["extra_checks"] if check["id"] == "I1")
    assert isolation["status"] == "fail"
    timeout = next(check for check in report["extra_checks"] if check["id"] == "T1")
    assert timeout["status"] == "skipped"


@pytest.mark.parametrize("worker", ["process", "thread"])
def test_validation_checks_expected_result_fields(tmp_path: Path, worker: str) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pipeline import prepare_bfcl

    pack = _copy_tiny_pack(tmp_path)

    def break_expected_status(cases: list[dict[str, Any]]) -> None:
        checkout = next(case for case in cases if case["id"] == "success_checkout_book")
        checkout["expect"]["status"] = "impossible_status"

    _edit_pack_yaml(pack / "validation_cases.yaml", break_expected_status)
    temp_config = _write_tiny_config(
        tmp_path,
        "bad-status.yaml",
        oracle_pack={"manifest_path": str(pack / "manifest.yaml")},
        oracle_runtime={"allowed_roots": [str(tmp_path)], "worker": worker},
    )

    report = json.loads(prepare_bfcl(temp_config).read_text(encoding="utf-8"))
    check = next(check for check in report["checks"] if check["id"] == 5)

    assert check["status"] == "fail"
    assert any(failure["reason"] == "result_field_mismatch" for failure in check["failures"])


@pytest.mark.parametrize("worker", ["process", "thread"])
def test_validation_cases_can_share_state_without_reset(tmp_path: Path, worker: str) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pipeline import prepare_bfcl

    pack = _copy_tiny_pack(tmp_path)

    def chain_after_checkout(cases: list[dict[str, Any]]) -> None:
        index = next(i for i, case in enumerate(cases) if case["id"] == "success_checkout_book")
        cases.insert(
            index + 1,
            {
                "id": "observe_checkout_without_reset",
                "tool": "get_book_status",
                "arguments": {"book_id": "BK-100"},
                "expect": {"result_class": "success", "status": "on_loan"},
                "reset_before": False,
            },
        )

    _edit_pack_yaml(pack / "validation_cases.yaml", chain_after_checkout)
    temp_config = _write_tiny_config(
        tmp_path,
        "shared-state.yaml",
        oracle_pack={"manifest_path": str(pack / "manifest.yaml")},
        oracle_runtime={"allowed_roots": [str(tmp_path)], "worker": worker},
    )

    report = json.loads(prepare_bfcl(temp_config).read_text(encoding="utf-8"))
    check = next(check for check in report["checks"] if check["id"] == 5)

    assert check["status"] == "pass"
    determinism = next(check for check in report["extra_checks"] if check["id"] == "D1")
    assert determinism["status"] == "pass"


@pytest.mark.parametrize(
    "clock",
    ["not-a-timestamp", "2026-03-02T09:00:00"],
)
def test_rejects_unusable_clock(tmp_path: Path, clock: str) -> None:
    temp_config = _write_tiny_config(tmp_path, "clock.yaml", oracle_runtime={"clock": clock})

    with pytest.raises(ValueError, match="oracle_runtime.clock"):
        BfclConfig.from_yaml(temp_config)


def test_rejects_unquoted_yaml_timestamp(tmp_path: Path) -> None:
    text = (BFCL_CONFIG_DIR / "tiny.yaml").read_text(encoding="utf-8")
    text = text.replace('"2026-03-02T09:00:00+07:00"', "2026-03-02T09:00:00+07:00")
    path = tmp_path / "unquoted-clock.yaml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="quoted ISO-8601 string"):
        BfclConfig.from_yaml(path)


def test_rejects_a_schema_version_this_build_cannot_write(tmp_path: Path) -> None:
    """The manifest publishes this value, so it must name a shape the rows really have."""
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.config import (
        DEFAULT_BENCHMARK_SCHEMA_VERSION,
    )

    with pytest.raises(ValueError, match="schema_version"):
        BfclConfig.from_yaml(_write_tiny_config(tmp_path, "future-schema.yaml", schema_version="9.9"))

    config = BfclConfig.from_yaml(
        _write_tiny_config(
            tmp_path, "known-schema.yaml", schema_version=DEFAULT_BENCHMARK_SCHEMA_VERSION
        )
    )
    assert config.schema_version == DEFAULT_BENCHMARK_SCHEMA_VERSION


def test_rejects_non_positive_timeout(tmp_path: Path) -> None:
    temp_config = _write_tiny_config(tmp_path, "timeout.yaml", oracle_runtime={"tool_timeout_s": 0})

    with pytest.raises(ValueError, match="tool_timeout_s"):
        BfclConfig.from_yaml(temp_config)


def test_unevaluable_slot_filter_is_reported_not_raised(tmp_path: Path) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pipeline import prepare_bfcl

    pack = _copy_tiny_pack(tmp_path)

    def break_filter(templates: list[dict[str, Any]]) -> None:
        for template in templates:
            for slot in (template.get("slots") or {}).values():
                if str(slot.get("source", "")).startswith("fixture:"):
                    slot["filter"] = "status IN ['available']"
                    return
        raise AssertionError("tiny pack has no fixture-backed slot to break")

    _edit_pack_yaml(pack / "task_templates.yaml", break_filter)
    temp_config = _write_tiny_config(
        tmp_path,
        "bad-filter.yaml",
        oracle_pack={"manifest_path": str(pack / "manifest.yaml")},
        oracle_runtime={"allowed_roots": [str(tmp_path)]},
    )

    report = json.loads(prepare_bfcl(temp_config).read_text(encoding="utf-8"))
    check = next(check for check in report["checks"] if check["id"] == 2)

    assert report["gold_eligible"] is False
    assert check["status"] == "fail"
    assert any(failure["reason"] == "unevaluable_filter" for failure in check["failures"])


def test_malformed_range_source_fails_prepare_check(tmp_path: Path) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pipeline import prepare_bfcl

    pack = _copy_tiny_pack(tmp_path)

    def break_source(templates: list[dict[str, Any]]) -> None:
        templates[0]["slots"]["book_id"]["source"] = "range:{'min': 1}"

    _edit_pack_yaml(pack / "task_templates.yaml", break_source)
    config = _write_tiny_config(
        tmp_path,
        "bad-range.yaml",
        oracle_pack={"manifest_path": str(pack / "manifest.yaml")},
        oracle_runtime={"allowed_roots": [str(tmp_path)]},
    )
    report = json.loads(prepare_bfcl(config).read_text(encoding="utf-8"))
    check = next(item for item in report["checks"] if item["id"] == 2)
    assert check["status"] == "fail"
    assert any(failure["reason"] == "invalid_source" for failure in check["failures"])


def test_generate_rejects_stage_resume(tmp_path: Path) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pipeline import generate_bfcl

    temp_config = _write_tiny_config(tmp_path, "resume.yaml")

    with pytest.raises(NotImplementedError, match="skip_until"):
        generate_bfcl(temp_config, skip_until="RENDER")


def test_unrunnable_validation_cases_are_skipped_not_passed(tmp_path: Path) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pipeline import prepare_bfcl

    pack = _copy_tiny_pack(tmp_path)
    tools_path = pack / "tools.json"
    tools = json.loads(tools_path.read_text(encoding="utf-8"))
    tools.append(
        {
            "type": "function",
            "function": {
                "name": "tool_the_backend_never_exposes",
                "description": "Declared in tools.json only.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }
    )
    tools_path.write_text(json.dumps(tools, indent=2) + "\n", encoding="utf-8")

    temp_config = _write_tiny_config(
        tmp_path,
        "schema-drift.yaml",
        oracle_pack={"manifest_path": str(pack / "manifest.yaml")},
        oracle_runtime={"allowed_roots": [str(tmp_path)]},
    )

    report = json.loads(prepare_bfcl(temp_config).read_text(encoding="utf-8"))
    alignment = next(check for check in report["checks"] if check["id"] == 3)
    cases = next(check for check in report["checks"] if check["id"] == 5)

    assert alignment["status"] == "fail"
    assert cases["status"] == "skipped"
    assert cases["failures"][0]["reason"] == "not_run"
    assert report["gold_eligible"] is False


def test_gold_requires_a_mutating_tool_to_say_so(tmp_path: Path) -> None:
    """A grader treats a read-only tool differently, so the claim must match reality."""
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pipeline import prepare_bfcl

    pack = _copy_tiny_pack(tmp_path)
    tools_path = pack / "tools.json"
    tools = json.loads(tools_path.read_text(encoding="utf-8"))
    mutating = [tool for tool in tools if tool.pop("x-mutates", False)]
    assert mutating, "the tiny pack must declare a mutating tool for this test to mean anything"
    tools_path.write_text(json.dumps(tools, indent=2) + "\n", encoding="utf-8")

    config = _write_tiny_config(
        tmp_path,
        "undeclared_mutation.yaml",
        oracle_pack={"manifest_path": str(pack / "manifest.yaml")},
        oracle_runtime={"allowed_roots": [str(tmp_path)]},
    )
    report = json.loads(prepare_bfcl(config).read_text(encoding="utf-8"))

    mutation_check = next(check for check in report["extra_checks"] if check["id"] == "M1")
    assert mutation_check["status"] == "fail"
    assert mutation_check["failures"][0]["reason"] == "undeclared_mutation"
    assert report["gold_eligible"] is False


def test_gold_rejects_a_declared_mutation_that_no_success_probe_observes(tmp_path: Path) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pipeline import prepare_bfcl

    pack = _copy_tiny_pack(tmp_path)
    tools_path = pack / "tools.json"
    tools = json.loads(tools_path.read_text(encoding="utf-8"))
    read_only = next(
        tool for tool in tools if tool["function"]["name"] == "get_book_status"
    )
    read_only["x-mutates"] = True
    tools_path.write_text(json.dumps(tools, indent=2) + "\n", encoding="utf-8")

    config = _write_tiny_config(
        tmp_path,
        "false-mutation.yaml",
        oracle_pack={"manifest_path": str(pack / "manifest.yaml")},
        oracle_runtime={"allowed_roots": [str(tmp_path)]},
    )
    report = json.loads(prepare_bfcl(config).read_text(encoding="utf-8"))

    mutation_check = next(check for check in report["extra_checks"] if check["id"] == "M1")
    assert mutation_check["status"] == "fail"
    assert any(
        failure["reason"] == "declared_mutation_not_observed"
        and failure["tool"] == "get_book_status"
        for failure in mutation_check["failures"]
    )


def test_determinism_uses_observed_success_not_only_the_expect_label(tmp_path: Path) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pipeline import prepare_bfcl

    pack = _copy_tiny_pack(tmp_path)

    def omit_success_label(cases: list[dict[str, Any]]) -> None:
        case = next(item for item in cases if item["id"] == "success_get_book_status")
        case["expect"].pop("result_class")

    _edit_pack_yaml(pack / "validation_cases.yaml", omit_success_label)
    config = _write_tiny_config(
        tmp_path,
        "inferred-success.yaml",
        oracle_pack={"manifest_path": str(pack / "manifest.yaml")},
        oracle_runtime={"allowed_roots": [str(tmp_path)]},
    )
    report = json.loads(prepare_bfcl(config).read_text(encoding="utf-8"))

    determinism = next(check for check in report["extra_checks"] if check["id"] == "D1")
    assert determinism["status"] == "pass"
    assert report["gold_eligible"] is True


def test_determinism_compares_state_even_when_result_is_stable(tmp_path: Path) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pipeline import prepare_bfcl

    pack = _copy_tiny_pack(tmp_path)
    backend = pack / "backend.py"
    source = backend.read_text(encoding="utf-8")
    source = source.replace(
        "from __future__ import annotations\n",
        "from __future__ import annotations\n\nimport time\n",
        1,
    ).replace(
        '    if name == "get_book_status":\n        return _get_book_status(arguments)',
        '    if name == "get_book_status":\n'
        '        _STATE["_nondeterministic_nonce"] = time.time_ns()\n'
        "        return _get_book_status(arguments)",
    )
    backend.write_text(source, encoding="utf-8")
    config = _write_tiny_config(
        tmp_path,
        "nondeterministic-state.yaml",
        oracle_pack={"manifest_path": str(pack / "manifest.yaml")},
        oracle_runtime={"allowed_roots": [str(tmp_path)]},
    )

    report = json.loads(prepare_bfcl(config).read_text(encoding="utf-8"))
    determinism = next(
        check for check in report["extra_checks"] if check["id"] == "D1"
    )
    assert determinism["status"] == "fail"
    assert any(
        failure["reason"] == "nondeterministic"
        for failure in determinism["failures"]
    )


def test_pack_load_checks_what_the_guards_depend_on(tmp_path: Path) -> None:
    """A slot with no visibility flag lands in neither the preserve nor the omit set."""
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pack_loader import normalize_templates

    template = {
        "template_id": "tpl",
        "turn_policy": "single_turn",
        "slots": {"thing_id": {"source": "literal:['T-1']", "visible_in_first_turn": True}},
    }

    # The paraphrase block is optional; the run-wide guards still apply without it.
    assert normalize_templates([template])[0]["paraphrase"] == {}

    unflagged = {**template, "slots": {"thing_id": {"source": "literal:['T-1']"}}}
    with pytest.raises(ValueError, match="visible_in_first_turn"):
        normalize_templates([unflagged])

    with pytest.raises(ValueError, match="unknown turn_policy"):
        normalize_templates([{**template, "turn_policy": "single-turn"}])

    missing_policy = {key: value for key, value in template.items() if key != "turn_policy"}
    with pytest.raises(ValueError, match="unknown turn_policy None"):
        normalize_templates([missing_policy])

    duplicate_milestones = {
        **template,
        "assistant_milestones": [
            {"id": "lookup", "type": "tool_call", "tool": "lookup_asset"},
            {"id": "lookup", "type": "tool_call", "tool": "inspect_asset"},
        ],
    }
    with pytest.raises(ValueError, match="duplicate milestone id 'lookup'"):
        normalize_templates([duplicate_milestones])

    dependent_any = {
        **template,
        "turn_policy": "dependent_call",
        "call_order": "any",
    }
    with pytest.raises(ValueError, match="dependent_call requires call_order: strict"):
        normalize_templates([dependent_any])


def test_a_pack_may_name_its_own_confirmation_protocol() -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pack_loader import (
        confirmation_protocol,
    )

    assert confirmation_protocol({})["parameter"] == "confirm"
    renamed = confirmation_protocol({"confirmation": {"parameter": "xac_nhan"}})
    assert renamed["parameter"] == "xac_nhan"
    assert renamed["pending_status"] == "awaiting_confirmation"

    with pytest.raises(ValueError, match="unknown keys"):
        confirmation_protocol({"confirmation": {"parameter_name": "xac_nhan"}})


def test_pack_fingerprint_covers_files_the_backend_reads(tmp_path: Path) -> None:
    """A helper module or data file the backend imports changes what the oracle does."""
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.config import BfclConfig
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pack_loader import (
        pack_fingerprint,
        resolve_pack_paths,
    )

    pack = _copy_tiny_pack(tmp_path)
    config = BfclConfig.from_yaml(
        _write_tiny_config(
            tmp_path,
            "fingerprint.yaml",
            oracle_pack={"manifest_path": str(pack / "manifest.yaml")},
            oracle_runtime={"allowed_roots": [str(tmp_path)]},
        )
    )
    paths = resolve_pack_paths(config)
    before = pack_fingerprint(paths)

    (pack / "policy.json").write_text('{"late_fee": 1}\n', encoding="utf-8")
    with_helper = pack_fingerprint(paths)
    assert with_helper != before

    (pack / "policy.json").write_text('{"late_fee": 2}\n', encoding="utf-8")
    assert pack_fingerprint(paths) != with_helper

    # Bytecode caches are not pack content, so importing the backend must not look
    # like an edit to it.
    edited = pack_fingerprint(paths)
    cache = pack / "__pycache__"
    cache.mkdir(exist_ok=True)
    (cache / "backend.cpython-312.pyc").write_bytes(b"\x00\x01")
    assert pack_fingerprint(paths) == edited


def test_pack_fingerprint_uses_semantic_names_for_external_files(tmp_path: Path) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pack_loader import (
        ResolvedPackPaths,
        pack_fingerprint,
    )

    def paths_for(host: str) -> ResolvedPackPaths:
        root = tmp_path / host / "asset_pack"
        shutil.copytree(
            BYOB_DIR / "data" / "tiny_oracle_pack",
            root,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        external = tmp_path / host / "runtime"
        external.mkdir(parents=True)
        backend = external / "oracle.py"
        backend.write_bytes((root / "backend.py").read_bytes())
        return ResolvedPackPaths(
            pack_root=root,
            manifest_path=root / "manifest.yaml",
            tools_path=root / "tools.json",
            fixtures_path=root / "fixtures.json",
            templates_path=root / "task_templates.yaml",
            assertions_path=root / "assertions.py",
            validation_cases_path=root / "validation_cases.yaml",
            system_prompt_path=None,
            backend_path=backend,
            endpoint_config_path=None,
        )

    assert pack_fingerprint(paths_for("host-a")) == pack_fingerprint(paths_for("host-b"))


def test_gold_requires_every_template_to_state_what_success_means(tmp_path: Path) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pipeline import prepare_bfcl

    pack = _copy_tiny_pack(tmp_path)

    def drop_assertions(templates: list[dict[str, Any]]) -> None:
        for template in templates:
            template["success_assertions"] = []

    _edit_pack_yaml(pack / "task_templates.yaml", drop_assertions)
    temp_config = _write_tiny_config(
        tmp_path,
        "no-assertions.yaml",
        oracle_pack={"manifest_path": str(pack / "manifest.yaml")},
        oracle_runtime={"allowed_roots": [str(tmp_path)]},
    )

    report = json.loads(prepare_bfcl(temp_config).read_text(encoding="utf-8"))
    check = next(item for item in report["checks"] if item["id"] == 4)

    # An importable assertions module is not evidence: a template that names no
    # assertion has no statement of success, so replay could only confirm the trace ran.
    assert report["stats"]["n_assertions"] > 0
    assert check["status"] == "fail"
    assert {failure["reason"] for failure in check["failures"]} == {
        "template_without_success_assertion"
    }
    assert report["gold_eligible"] is False


def test_generate_revalidates_when_worker_changes(tmp_path: Path) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pipeline import generate_bfcl, prepare_bfcl

    process_config = _write_tiny_config(tmp_path, "process.yaml")
    process_report = json.loads(prepare_bfcl(process_config).read_text(encoding="utf-8"))
    assert process_report["gold_eligible"] is True

    thread_config = _write_tiny_config(
        tmp_path,
        "thread.yaml",
        oracle_runtime={"worker": "thread"},
    )
    with pytest.raises(RuntimeError, match="non-gold pack"):
        generate_bfcl(thread_config)

    report_path = (
        tmp_path
        / "output"
        / "bfcl_tiny_library_validation"
        / "stage_cache"
        / "oracle_validation_report.json"
    )
    thread_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert thread_report["validation_config_fingerprint"] != process_report[
        "validation_config_fingerprint"
    ]
    isolation = next(check for check in thread_report["extra_checks"] if check["id"] == "I1")
    assert isolation["status"] == "fail"


def test_duplicate_template_ids_are_rejected(tmp_path: Path) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pipeline import prepare_bfcl

    pack = _copy_tiny_pack(tmp_path)

    def duplicate_first(templates: list[dict[str, Any]]) -> None:
        templates.append(dict(templates[0]))

    _edit_pack_yaml(pack / "task_templates.yaml", duplicate_first)
    config = _write_tiny_config(
        tmp_path,
        "duplicate-template.yaml",
        oracle_pack={"manifest_path": str(pack / "manifest.yaml")},
        oracle_runtime={"allowed_roots": [str(tmp_path)]},
    )
    with pytest.raises(ValueError, match="duplicate template_id"):
        prepare_bfcl(config)


def test_system_prompt_cannot_escape_pack_allowlist(tmp_path: Path) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.isolation import PackTrustError
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pipeline import prepare_bfcl

    pack = _copy_tiny_pack(tmp_path)
    (tmp_path / "secret.txt").write_text("not a pack artifact", encoding="utf-8")

    def point_outside(manifest: dict[str, Any]) -> None:
        manifest["system_prompt_path"] = "../secret.txt"

    _edit_pack_yaml(pack / "manifest.yaml", point_outside)
    config = _write_tiny_config(
        tmp_path,
        "prompt-escape.yaml",
        oracle_pack={"manifest_path": str(pack / "manifest.yaml")},
        oracle_runtime={"allowed_roots": [str(pack)]},
    )
    with pytest.raises(PackTrustError):
        prepare_bfcl(config)


def test_default_config_requests_only_supported_features() -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pipeline import _unsupported_requests

    config = BfclConfig.from_yaml(BFCL_CONFIG_DIR / "default.yaml")

    assert config.stage == "all"
    assert _unsupported_requests(config) == []


def test_the_publication_template_names_no_example_pack() -> None:
    """A template that defaults to a bundled pack publishes it to anyone who forgets."""
    data = yaml.safe_load((BFCL_CONFIG_DIR / "default.yaml").read_text(encoding="utf-8"))

    assert str(data["oracle_pack"]["manifest_path"]).startswith("REPLACE_ME")


def test_generate_refuses_settings_it_would_otherwise_ignore(tmp_path: Path) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pipeline import _unsupported_requests

    baseline = BfclConfig.from_yaml(_write_tiny_config(tmp_path, "baseline.yaml"))
    assert _unsupported_requests(baseline) == []

    asks_for_paraphrases = BfclConfig.from_yaml(
        _write_tiny_config(
            tmp_path,
            "paraphrases.yaml",
            surface_generation={"paraphrases_per_template": 3},
        )
    )
    assert "surface_generation.paraphrases_per_template" in _unsupported_requests(
        asks_for_paraphrases
    )

    with pytest.raises(ValueError, match="surface_generation has unknown keys: langauge"):
        BfclConfig.from_yaml(
            _write_tiny_config(tmp_path, "typo.yaml", surface_generation={"langauge": "en"})
        )

    claims_a_judge = BfclConfig.from_yaml(
        _write_tiny_config(tmp_path, "judge.yaml", lineage={"judge_advisory": True})
    )
    assert "lineage.judge_advisory" in _unsupported_requests(claims_a_judge)

    leftover = BfclConfig.from_yaml(
        _write_tiny_config(tmp_path, "batch.yaml", ndd_batch_size=8)
    )
    assert "ndd_batch_size" in _unsupported_requests(leftover)


def test_generate_revalidates_a_hand_edited_gold_report(tmp_path: Path) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl import pipeline
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pipeline import (
        generate_bfcl,
        prepare_bfcl,
    )

    config = _write_tiny_config(tmp_path, "tampered.yaml")
    prepare_bfcl(config)
    report_path = (
        tmp_path
        / "output"
        / "bfcl_tiny_library_validation"
        / "stage_cache"
        / "oracle_validation_report.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["gold_eligible"] is True
    # Manufacture a passing payload while retaining public fingerprints. Generate
    # must replace it by executing validation, not trust locally editable statuses.
    report["checks"] = [{"id": "invented", "status": "pass", "failures": []}]
    report["extra_checks"] = []
    report_path.write_text(json.dumps(report) + "\n", encoding="utf-8")
    # Drop the in-process verdict so this stands in for a separate `stage=generate` run.
    pipeline._VALIDATED_THIS_PROCESS.clear()

    assert generate_bfcl(config).exists()
    repaired = json.loads(report_path.read_text(encoding="utf-8"))
    assert len(repaired["checks"]) == 7
    assert {check["id"] for check in repaired["extra_checks"]} == {"M1", "D1", "D2", "T1", "I1"}


def test_stage_all_validates_once_but_a_new_run_revalidates(tmp_path: Path) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl import pipeline
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pipeline import (
        generate_bfcl,
        prepare_bfcl,
    )
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages import oracle_validation

    config = _write_tiny_config(tmp_path, "revalidate.yaml")
    calls: list[str] = []
    real_run = oracle_validation.run_oracle_validation

    def counted(cfg, pack):  # type: ignore[no-untyped-def]
        calls.append(str(cfg.output_dir))
        return real_run(cfg, pack)

    oracle_validation.run_oracle_validation = counted
    try:
        prepare_bfcl(config)
        assert generate_bfcl(config).exists()
        assert len(calls) == 1
        pipeline._VALIDATED_THIS_PROCESS.clear()
        assert generate_bfcl(config).exists()
        assert len(calls) == 2
    finally:
        oracle_validation.run_oracle_validation = real_run


def test_resolved_config_hash_input_is_portable_for_external_packs(tmp_path: Path) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.final_output import (
        _resolved_config,
    )

    machine_a = tmp_path / "host-a"
    machine_b = tmp_path / "host-b"
    config_a = BfclConfig.from_yaml(
        _write_tiny_config(
            tmp_path,
            "portable-a.yaml",
            output_dir=str(machine_a / "output"),
            oracle_pack={"manifest_path": str(machine_a / "asset_pack" / "manifest.yaml")},
            oracle_runtime={"allowed_roots": [str(machine_a / "asset_pack")]},
        )
    )
    config_b = BfclConfig.from_yaml(
        _write_tiny_config(
            tmp_path,
            "portable-b.yaml",
            output_dir=str(machine_b / "output"),
            oracle_pack={"manifest_path": str(machine_b / "asset_pack" / "manifest.yaml")},
            oracle_runtime={"allowed_roots": [str(machine_b / "asset_pack")]},
        )
    )

    assert _resolved_config(config_a) == _resolved_config(config_b)


def test_generate_refuses_features_no_stage_applies(tmp_path: Path) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pipeline import generate_bfcl

    temp_config = _write_tiny_config(
        tmp_path,
        "judged.yaml",
        surface_quality_validation={"enabled": True, "drop_authority": True},
    )

    with pytest.raises(NotImplementedError, match="surface_quality_validation.enabled"):
        generate_bfcl(temp_config)


def test_generate_refuses_ignored_task_generation_controls(tmp_path: Path) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pipeline import generate_bfcl

    temp_config = _write_tiny_config(
        tmp_path,
        "balanced.yaml",
        task_generation={"turn_mix": {"single_turn": 1.0}},
    )
    with pytest.raises(ValueError, match="task_generation has unknown keys: turn_mix"):
        generate_bfcl(temp_config)


def test_generate_revalidates_when_pack_changed(tmp_path: Path) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pipeline import generate_bfcl, prepare_bfcl

    pack = _copy_tiny_pack(tmp_path)
    temp_config = _write_tiny_config(
        tmp_path,
        "stale.yaml",
        oracle_pack={"manifest_path": str(pack / "manifest.yaml")},
        oracle_runtime={"allowed_roots": [str(tmp_path)]},
    )
    report_path = prepare_bfcl(temp_config)
    assert json.loads(report_path.read_text(encoding="utf-8"))["gold_eligible"] is True

    def drop_negative_cases(cases: list[dict[str, Any]]) -> None:
        cases[:] = [case for case in cases if case["id"].startswith("success_")]

    _edit_pack_yaml(pack / "validation_cases.yaml", drop_negative_cases)

    with pytest.raises(RuntimeError, match="non-gold pack"):
        generate_bfcl(temp_config)

    assert json.loads(report_path.read_text(encoding="utf-8"))["gold_eligible"] is False
