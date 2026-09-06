# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""curate/flow: one config, six steps, and the things it must refuse.

The value of a flow is not that it saves typing — it is that two agreements
which fail silently by hand (the manifest path the audit compares against, and
the score column subset stratifies on) become impossible to get wrong. Most of
these tests are about refusals, because a flow that runs and produces a
clean-looking result from a broken configuration is worse than six commands.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest
import yaml

from nemotron.steps.curate.runtime import policy as policy_module
from nemotron.steps.curate.runtime import registry as signal_registry
from nemotron.steps.curate.scripts import run_flow

from .._step_helpers import assert_step_static, step_dir

STEP_DIR = step_dir(__file__, "curate", "flow")
LANGPACK_FIXTURES = Path(__file__).parent / "fixtures" / "langpacks"
PACKAGE_PACKS = STEP_DIR.parent / "data" / "langpacks"


def corpus_files(tmp_path: Path, n: int = 40) -> str:
    directory = tmp_path / "raw"
    directory.mkdir(exist_ok=True)
    words = "curation tokens budget corpus stratum nesting document policy".split()
    with (directory / "part_0.jsonl").open("w", encoding="utf-8") as fh:
        for i in range(n):
            text = " ".join(words[(i + j) % len(words)] for j in range(30)) + "."
            fh.write(json.dumps({"id": f"d{i:03d}", "source": "web", "text": text}) + "\n")
    return str(directory / "*.jsonl")


def config(tmp_path: Path, **overrides) -> dict:
    cfg = {
        "corpus": {
            "input": corpus_files(tmp_path),
            "text_field": "text",
            "id_field": "id",
            "source_field": "source",
            "language": "x-test-vi",
            "langpack_dir": str(LANGPACK_FIXTURES),
        },
        "output_root": str(tmp_path / "out"),
        "steps": {
            "profile": {"enabled": True, "signals": ["unicode_alpha_numeric"], "max_total_docs": 40},
            "filter": {"enabled": False},
            "audit": {"enabled": False},
            "subset": {"enabled": False},
            "decontamination": {"enabled": False},
        },
        "approve": None,
    }
    cfg.update(overrides)
    return cfg


# -- static -------------------------------------------------------------------


def test_flow_step_static() -> None:
    assert_step_static(
        STEP_DIR,
        expected_name="steps/curate/flow",
        expected_launch="python",
        expected_default_config="default",
    )


def test_the_flow_declares_no_gpu() -> None:
    """Five of the six steps need none; decontamination asks per-run instead."""
    source = (STEP_DIR / "step.py").read_text(encoding="utf-8")

    assert "gpus_per_node = 0" in source


def test_the_shipped_default_does_not_filter_on_an_unreviewed_policy() -> None:
    cfg = yaml.safe_load((STEP_DIR / "config" / "default.yaml").read_text(encoding="utf-8"))

    assert cfg["approve"] is None
    assert cfg["corpus"]["language"] is None
    assert cfg["corpus"]["langpack_dir"] is None


def test_the_default_config_covers_every_step_in_the_plan() -> None:
    """A step missing from the shipped config is one nobody knows they can enable."""
    cfg = yaml.safe_load((STEP_DIR / "config" / "default.yaml").read_text(encoding="utf-8"))

    assert set(cfg["steps"]) == {plan.key for plan in run_flow.STEP_ORDER}


def test_tiny_runs_ingest_without_gpu_downloads_or_a_language_pack() -> None:
    cfg = yaml.safe_load((STEP_DIR / "config" / "tiny.yaml").read_text(encoding="utf-8"))

    assert cfg["steps"]["ingest"]["enabled"] is True
    assert all(not cfg["steps"][key]["enabled"] for key in ("profile", "filter", "audit", "subset", "decontamination"))
    assert "language" not in cfg["corpus"]


# -- derivation ---------------------------------------------------------------


def test_the_manifest_path_is_derived_for_both_producer_and_consumer(tmp_path) -> None:
    """The agreement that fails silently by hand.

    curate/nemo_curator ships emit_manifest: null and curate/audit ships
    declared_manifest: null. Two nulls in two files: an audit against a producer
    that emitted no manifest claims nothing, which reads as a clean result.
    """
    resolved, paths = run_flow.derive(config(tmp_path))
    by_key = {r.plan.key: r.config for r in resolved}

    assert by_key["filter"]["emit_manifest"] == paths["manifest"]
    assert by_key["audit"]["declared_manifest"] == paths["manifest"]
    assert by_key["filter"]["emit_ledger"] == by_key["audit"]["ledger_glob"]


def test_profile_reads_the_unfiltered_corpus_and_the_rest_read_the_output(tmp_path) -> None:
    """Profiling the filtered output measures gates that have already run."""
    cfg = config(tmp_path)
    resolved, paths = run_flow.derive(cfg)
    by_key = {r.plan.key: r.config for r in resolved}

    assert by_key["profile"]["input_glob"] == cfg["corpus"]["input"]
    assert by_key["profile"]["language"] == cfg["corpus"]["language"]
    assert by_key["profile"]["langpack_dir"] == cfg["corpus"]["langpack_dir"]
    assert by_key["filter"]["input_glob"] == cfg["corpus"]["input"]
    for key in ("subset",):
        assert by_key[key]["input_glob"].startswith(paths["corpus"])
    assert by_key["audit"]["target_glob"].startswith(paths["corpus"])
    assert by_key["decontamination"]["train_glob"].startswith(paths["corpus"])


def test_ingest_hands_canonical_field_names_to_every_consumer(tmp_path) -> None:
    cfg = config(tmp_path)
    cfg["corpus"].update(
        {
            "text_field": "body",
            "id_field": "doc_key",
            "id_field_in_source": True,
            "source_field": "dataset",
            "source_field_in_source": "origin",
            "metadata_fields": ["doc_key", "origin"],
        }
    )
    cfg["steps"]["ingest"] = {"enabled": True}

    resolved, _ = run_flow.derive(cfg)
    by_key = {r.plan.key: r.config for r in resolved}

    assert by_key["ingest"]["text_field"] == "body"
    assert by_key["ingest"]["id_from"] == "doc_key"
    assert by_key["ingest"]["source_from"] == "origin"
    for key in ("profile", "filter", "subset"):
        assert by_key[key]["text_field"] == "text"
        assert by_key[key]["id_field"] == "id"
        assert by_key[key]["source_field"] == "source"
    assert by_key["decontamination"]["text_field"] == "text"
    assert by_key["decontamination"]["id_field"] == "id"
    assert {"id", "source"} <= set(by_key["filter"]["metadata_fields"])


def test_an_authored_key_overrides_the_derivation(tmp_path) -> None:
    """A flow config that cannot be escaped needs a second file, which defeats it."""
    cfg = config(tmp_path)
    cfg["steps"]["subset"]["output_dir"] = "/somewhere/else"

    resolved, _ = run_flow.derive(cfg)
    subset = next(r for r in resolved if r.plan.key == "subset")

    assert subset.config["output_dir"] == "/somewhere/else"


def test_a_corpus_without_an_input_is_refused(tmp_path) -> None:
    cfg = config(tmp_path)
    cfg["corpus"].pop("input")

    with pytest.raises(run_flow.FlowConfigError, match="corpus.input is required"):
        run_flow.derive(cfg)


# -- preflight ----------------------------------------------------------------


def test_a_step_whose_producer_is_disabled_and_absent_is_refused(tmp_path) -> None:
    """Naming the disabled producer is the point: the alternative is a step
    reporting an empty-looking result nobody can trace."""
    cfg = config(tmp_path)
    cfg["steps"]["audit"]["enabled"] = True

    with pytest.raises(run_flow.FlowConfigError) as excinfo:
        resolved, paths = run_flow.derive(cfg)
        run_flow.preflight(cfg, resolved, paths)

    message = str(excinfo.value)
    assert "steps.filter" in message
    assert "disabled" in message


def test_a_previous_runs_artifact_is_reused_when_the_producer_is_disabled(tmp_path) -> None:
    cfg = config(tmp_path)
    cfg["steps"]["audit"]["enabled"] = True
    resolved, paths = run_flow.derive(cfg)

    # Materialise what a previous run would have left behind.
    Path(paths["corpus"]).mkdir(parents=True)
    (Path(paths["corpus"]) / "a.jsonl").write_text('{"id": "x"}\n', encoding="utf-8")
    Path(paths["manifest"]).write_text("{}", encoding="utf-8")
    Path(paths["ledger"]).write_text("{}", encoding="utf-8")

    run_flow.preflight(cfg, resolved, paths)
    audit = next(r for r in resolved if r.plan.key == "audit")

    assert any("reusing" in note for note in audit.notes)


def test_nothing_enabled_is_refused(tmp_path) -> None:
    cfg = config(tmp_path)
    cfg["steps"]["profile"]["enabled"] = False

    resolved, paths = run_flow.derive(cfg)
    with pytest.raises(run_flow.FlowConfigError, match="no steps are enabled"):
        run_flow.preflight(cfg, resolved, paths)


def test_stratifying_on_a_column_the_filter_will_not_write_is_refused(tmp_path) -> None:
    """mode: filter discards scores after use, so the column never exists."""
    cfg = config(tmp_path)
    cfg["steps"]["filter"] = {"enabled": True, "mode": "filter"}
    cfg["steps"]["subset"] = {"enabled": True, "quality_score_field": "__stopword_ratio"}

    resolved, paths = run_flow.derive(cfg)
    with pytest.raises(run_flow.FlowConfigError, match="quality_score_field"):
        run_flow.preflight(cfg, resolved, paths)


def test_mode_alone_does_not_satisfy_the_score_column_requirement(tmp_path) -> None:
    """The score columns are written by the policy's signals, not by the mode.

    With no policy there are no signals, so annotate/both write nothing — and
    that is precisely the documented first run. A check that looked only at
    mode would green-light the one configuration it exists to refuse.
    """
    cfg = config(tmp_path)
    cfg["steps"]["filter"] = {"enabled": True, "mode": "both"}
    cfg["steps"]["subset"] = {"enabled": True, "quality_score_field": "__stopword_ratio"}

    resolved, paths = run_flow.derive(cfg)
    with pytest.raises(run_flow.FlowConfigError, match="no policy is configured"):
        run_flow.preflight(cfg, resolved, paths)


def test_a_policy_plus_annotate_mode_satisfies_it(tmp_path) -> None:
    cfg = config(tmp_path)
    cfg["steps"]["filter"] = {
        "enabled": True,
        "mode": "both",
        "heuristic_filters": {"approved_policy": "./policy.yaml"},
    }
    cfg["steps"]["subset"] = {"enabled": True, "quality_score_field": "__stopword_ratio"}

    resolved, paths = run_flow.derive(cfg)

    run_flow.preflight(cfg, resolved, paths)  # must not raise


def test_an_audit_without_a_ledger_is_warned_not_silently_uninformative(tmp_path) -> None:
    """attribution.available: false means 'nobody recorded why', not 'nothing left'."""
    cfg = config(tmp_path)
    cfg["steps"]["audit"]["enabled"] = True
    resolved, paths = run_flow.derive(cfg)
    Path(paths["corpus"]).mkdir(parents=True)
    (Path(paths["corpus"]) / "a.jsonl").write_text('{"id": "x"}\n', encoding="utf-8")
    Path(paths["manifest"]).write_text("{}", encoding="utf-8")

    warnings = run_flow.preflight(cfg, resolved, paths)

    assert any("attribution.available: false" in w for w in warnings)


def test_decontamination_without_a_holdout_is_refused(tmp_path) -> None:
    cfg = config(tmp_path)
    cfg["steps"]["decontamination"] = {"enabled": True}
    resolved, paths = run_flow.derive(cfg)
    Path(paths["corpus"]).mkdir(parents=True)
    (Path(paths["corpus"]) / "a.jsonl").write_text('{"id": "x"}\n', encoding="utf-8")

    with pytest.raises(run_flow.FlowConfigError, match="holdout"):
        run_flow.preflight(cfg, resolved, paths)


def test_preflight_refuses_before_any_step_runs(tmp_path) -> None:
    """A flow that fails after the filter rewrote the corpus is worse than five commands."""
    cfg = config(tmp_path)
    cfg["steps"]["audit"]["enabled"] = True

    with pytest.raises(run_flow.FlowConfigError):
        run_flow.run(cfg)

    assert not (Path(cfg["output_root"]) / "profile").exists()


def test_profile_preflight_requires_an_external_language_pack(tmp_path) -> None:
    cfg = config(tmp_path)
    cfg["corpus"]["langpack_dir"] = None
    resolved, paths = run_flow.derive(cfg)

    with pytest.raises(run_flow.FlowConfigError, match="explicit langpack_dir"):
        run_flow.preflight(cfg, resolved, paths)


# -- the approval gate --------------------------------------------------------


def swept_threshold(minimum: float = 0.25) -> float:
    return float(next(v for v in signal_registry.SIGNALS["unicode_alpha_numeric"].grid.values() if v >= minimum))


def approve_block(**overrides) -> dict:
    block = {
        "thresholds": [{"signal": "unicode_alpha_numeric", "max": swept_threshold()}],
        "approver": "someone@example.test",
        "date": "2026-08-26",
        "method": "manual",
        "evidence": "retention curve reviewed against the sampled documents",
    }
    block.update(overrides)
    return block


def profiled(tmp_path: Path) -> dict:
    """A config whose profile has already run, so candidates exist."""
    cfg = config(tmp_path)
    run_flow.run(cfg)
    cfg["steps"]["profile"]["enabled"] = False
    cfg["steps"]["filter"]["enabled"] = True
    return cfg


def test_approving_before_profiling_is_refused(tmp_path) -> None:
    """There is nothing to approve until the corpus has been measured."""
    cfg = config(tmp_path)
    cfg["approve"] = approve_block()
    resolved, paths = run_flow.derive(cfg)

    with pytest.raises(run_flow.FlowConfigError, match="does not exist"):
        run_flow.materialise_policy(cfg, resolved, paths)


def test_profile_cannot_replace_candidates_in_the_approval_run(tmp_path) -> None:
    cfg = profiled(tmp_path)
    cfg["steps"]["profile"]["enabled"] = True
    cfg["approve"] = approve_block()
    resolved, paths = run_flow.derive(cfg)

    with pytest.raises(run_flow.FlowConfigError, match="Profiling and approval are two runs"):
        run_flow.preflight(cfg, resolved, paths)


def test_an_approval_produces_an_executable_policy_wired_into_the_filter(tmp_path) -> None:
    cfg = profiled(tmp_path)
    cfg["approve"] = approve_block()
    resolved, paths = run_flow.derive(cfg)

    run_flow.materialise_policy(cfg, resolved, paths)

    document = yaml.safe_load(Path(paths["approved_policy"]).read_text(encoding="utf-8"))
    assert document["approved"] is True
    assert policy_module.validate_approved_policy(document) == []
    filter_cfg = next(r for r in resolved if r.plan.key == "filter").config
    assert filter_cfg["heuristic_filters"]["approved_policy"] == paths["approved_policy"]
    assert filter_cfg["heuristic_filters"]["langpack_dir"] == str(LANGPACK_FIXTURES)


@pytest.mark.parametrize("pack_root", [None, "bundled"])
def test_pack_backed_approval_refuses_without_the_profiled_pack_root(tmp_path, pack_root) -> None:
    cfg = config(tmp_path)
    cfg["steps"]["profile"]["signals"] = ["stopword_ratio"]
    run_flow.run(cfg)
    cfg["steps"]["profile"]["enabled"] = False
    cfg["steps"]["filter"]["enabled"] = True
    cfg["corpus"]["langpack_dir"] = pack_root
    threshold = next(value for value in signal_registry.SIGNALS["stopword_ratio"].grid.values() if value > 0)
    cfg["approve"] = approve_block(thresholds=[{"signal": "stopword_ratio", "min": threshold}])
    resolved, paths = run_flow.derive(cfg)

    with pytest.raises(run_flow.FlowConfigError, match="no langpack_dir"):
        run_flow.materialise_policy(cfg, resolved, paths)


def test_approve_refuses_a_conflicting_per_step_policy_path(tmp_path) -> None:
    cfg = profiled(tmp_path)
    cfg["approve"] = approve_block()
    cfg["steps"]["filter"]["heuristic_filters"] = {"approved_policy": str(tmp_path / "other.yaml")}
    resolved, paths = run_flow.derive(cfg)

    with pytest.raises(run_flow.FlowConfigError, match="cannot truthfully report which policy"):
        run_flow.materialise_policy(cfg, resolved, paths)


def test_an_approval_granted_against_other_data_is_refused(tmp_path) -> None:
    """The failure a shared config file invites: thresholds AND the signature
    that approved them, applied to a corpus nobody approved them for."""
    cfg = profiled(tmp_path)
    cfg["approve"] = approve_block()

    other = tmp_path / "other"
    other.mkdir()
    rows = [{"id": f"o{i}", "source": "web", "text": "different corpus entirely here"} for i in range(20)]
    (other / "b.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    cfg["corpus"]["input"] = str(other / "*.jsonl")
    resolved, paths = run_flow.derive(cfg)

    with pytest.raises(run_flow.FlowConfigError, match="granted against corpus"):
        run_flow.materialise_policy(cfg, resolved, paths)


def test_the_fingerprint_check_can_be_waived_only_explicitly(tmp_path) -> None:
    """Waivable, but never by accident, and the README says not to."""
    cfg = profiled(tmp_path)
    cfg["approve"] = approve_block(verify_corpus=False)
    other = tmp_path / "other2"
    other.mkdir()
    (other / "b.jsonl").write_text(
        json.dumps({"id": "o1", "source": "web", "text": "quite different text here"}) + "\n",
        encoding="utf-8",
    )
    cfg["corpus"]["input"] = str(other / "*.jsonl")
    resolved, paths = run_flow.derive(cfg)

    run_flow.materialise_policy(cfg, resolved, paths)  # must not raise

    assert Path(paths["approved_policy"]).is_file()


def test_an_approval_needs_only_its_thresholds(tmp_path) -> None:
    """approver/date/evidence are recorded when given and never required.

    A name in a YAML file refuses no wrong run. The checks that do — corpus
    fingerprint, profile digest, scorer version, bound direction — are enforced
    regardless of who signed.
    """
    cfg = profiled(tmp_path)
    cfg["approve"] = approve_block(approver=None, date=None, evidence=None)
    resolved, paths = run_flow.derive(cfg)

    run_flow.materialise_policy(cfg, resolved, paths)

    written = yaml.safe_load(Path(paths["approved_policy"]).read_text(encoding="utf-8"))
    assert written["approved"] is True
    assert "approver" not in (written.get("approval") or {}), (
        "a null field would record nothing while looking like provenance"
    )


def test_an_approval_with_no_thresholds_is_refused(tmp_path) -> None:
    cfg = profiled(tmp_path)
    cfg["approve"] = approve_block(thresholds=[])
    resolved, paths = run_flow.derive(cfg)

    with pytest.raises(run_flow.FlowConfigError, match="gates nothing"):
        run_flow.materialise_policy(cfg, resolved, paths)


def test_an_approval_nothing_applies_is_warned(tmp_path) -> None:
    cfg = profiled(tmp_path)
    cfg["steps"]["filter"]["enabled"] = False
    cfg["steps"]["subset"] = {"enabled": True}
    cfg["approve"] = approve_block()
    resolved, paths = run_flow.derive(cfg)
    Path(paths["corpus"]).mkdir(parents=True, exist_ok=True)
    (Path(paths["corpus"]) / "a.jsonl").write_text('{"id": "x"}\n', encoding="utf-8")

    warnings = run_flow.preflight(cfg, resolved, paths)

    assert any("nothing applies it" in w for w in warnings)


def test_the_flow_is_not_a_second_approver() -> None:
    """runtime/policy.py::promote must remain the only one.

    Enforced repo-wide by test_policy.py; asserted here too because a flow is
    exactly where a convenience shortcut would be added.
    """
    import inspect

    source = inspect.getsource(run_flow)

    assert "promote(" in source, "the flow must go through promote()"
    assert '"approved": True' not in source
    assert "'approved': True" not in source


# -- running ------------------------------------------------------------------


def test_a_run_writes_a_plan_and_a_report(tmp_path) -> None:
    cfg = config(tmp_path)

    report = run_flow.run(cfg)

    root = Path(cfg["output_root"])
    assert (root / "flow_plan.json").is_file()
    assert (root / "flow_report.json").is_file()
    assert report["step_id"] == "curate/flow"


def test_a_failed_rerun_does_not_leave_the_previous_success_report(tmp_path) -> None:
    cfg = config(tmp_path)
    run_flow.run(cfg)
    root = Path(cfg["output_root"])
    cfg["steps"]["profile"]["enabled"] = False

    with pytest.raises(run_flow.FlowConfigError, match="no steps are enabled"):
        run_flow.run(cfg)

    assert not (root / "flow_plan.json").exists()
    assert not (root / "flow_report.json").exists()


def test_the_plan_records_every_derived_config_before_anything_runs(tmp_path) -> None:
    """A plan you can only read after the corpus was rewritten cannot reject it."""
    cfg = config(tmp_path)
    run_flow.run(cfg)

    plan = json.loads((Path(cfg["output_root"]) / "flow_plan.json").read_text(encoding="utf-8"))

    assert {s["key"] for s in plan["steps"]} == {p.key for p in run_flow.STEP_ORDER}
    profile = next(s for s in plan["steps"] if s["key"] == "profile")
    assert profile["config"]["input_glob"] == cfg["corpus"]["input"]


def test_disabled_steps_are_reported_as_disabled_not_omitted(tmp_path) -> None:
    """A step missing from the report is indistinguishable from one that failed."""
    report = run_flow.run(config(tmp_path))

    statuses = {s["step_id"]: s["status"] for s in report["steps"]}
    assert statuses["curate/profile"] == "ok"
    assert statuses["curate/nemo_curator"] == "disabled"
    assert len(statuses) == len(run_flow.STEP_ORDER)


def test_the_enabled_step_actually_produced_its_artifacts(tmp_path) -> None:
    cfg = config(tmp_path)

    run_flow.run(cfg)

    profile_dir = Path(cfg["output_root"]) / "profile"
    assert (profile_dir / "profile_report.json").is_file()
    assert (profile_dir / "candidate_policies.yaml").is_file()


def test_custom_raw_fields_run_from_ingest_through_profile(tmp_path) -> None:
    raw_dir = tmp_path / "custom_raw"
    raw_dir.mkdir()
    raw_path = raw_dir / "part.jsonl"
    raw_path.write_text(
        "".join(
            json.dumps({"doc_key": f"k{i}", "origin": "web", "body": f"document {i} has useful text"}) + "\n"
            for i in range(4)
        ),
        encoding="utf-8",
    )
    cfg = config(tmp_path)
    cfg["corpus"] = {
        "input": str(raw_path),
        "text_field": "body",
        "id_field": "doc_key",
        "id_field_in_source": True,
        "source_field": "dataset",
        "source_field_in_source": "origin",
        "language": "x-test-vi",
        "langpack_dir": str(LANGPACK_FIXTURES),
        "metadata_fields": ["doc_key", "origin"],
    }
    cfg["steps"]["ingest"] = {"enabled": True}

    report = run_flow.run(cfg)

    ingested = [
        json.loads(line)
        for line in (Path(cfg["output_root"]) / "ingested" / "part_0.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    profile = json.loads((Path(cfg["output_root"]) / "profile" / "profile_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "ok"
    assert profile["corpus"]["document_count"] == 4
    assert profile["corpus"]["source_count"] == 1
    assert {(row["id"], row["source"], row["text"]) for row in ingested} == {
        (f"k{i}", "web", f"document {i} has useful text") for i in range(4)
    }


def test_flow_cpu_chain_preserves_manifest_and_ledger_identities(tmp_path, monkeypatch) -> None:
    """Run the real audit against a CPU producer stub through the flow seam."""
    from nemotron.steps.curate.runtime import integrity, ledger
    from nemotron.steps.curate.runtime import manifest as manifest_module

    cfg = config(tmp_path)
    cfg["steps"]["profile"]["enabled"] = False
    cfg["steps"]["filter"]["enabled"] = True
    cfg["steps"]["audit"]["enabled"] = True
    real_step_runner = run_flow.step_runner
    filter_config: dict = {}

    def cpu_filter(step_cfg):
        filter_config.update(step_cfg)
        inputs = integrity.expand_inputs(step_cfg["input_glob"])
        output = Path(step_cfg["output_dir"])
        output.mkdir(parents=True, exist_ok=True)
        shard = output / "part_0.jsonl"
        shard.write_text(
            "".join(Path(path).read_text(encoding="utf-8") for path in inputs),
            encoding="utf-8",
        )
        input_counts = manifest_module.count_jsonl(inputs)
        output_counts = manifest_module.count_jsonl([shard])
        document = manifest_module.build_manifest(
            step_id="curate/nemo_curator",
            config=step_cfg,
            started_at="2026-08-25T00:00:00+00:00",
            completed_at="2026-08-25T00:00:01+00:00",
            input_glob=step_cfg["input_glob"],
            input_counts=input_counts,
            output_counts=output_counts,
            id_field=step_cfg["id_field"],
            source_field=step_cfg["source_field"],
        )
        manifest_module.write_manifest(step_cfg["emit_manifest"], document)
        accounting = ledger.StageLedger(stage="curate/nemo_curator")
        accounting.add_input(input_counts["row_count"])
        accounting.add_success(output_counts["row_count"])
        accounting.write(step_cfg["emit_ledger"], require_balanced=True)
        return {"warnings": []}

    def runner(key):
        return cpu_filter if key == "filter" else real_step_runner(key)

    monkeypatch.setattr(run_flow, "step_runner", runner)

    report = run_flow.run(cfg)

    audit_config = next(
        step["config"]
        for step in json.loads((Path(cfg["output_root"]) / "flow_plan.json").read_text(encoding="utf-8"))["steps"]
        if step["key"] == "audit"
    )
    assert audit_config["declared_manifest"] == filter_config["emit_manifest"]
    assert audit_config["ledger_glob"] == filter_config["emit_ledger"]
    assert Path(audit_config["declared_manifest"]).is_file()
    assert Path(audit_config["ledger_glob"]).is_file()
    assert report["audit_passed"] is True


def test_run_does_not_exit_the_process(tmp_path) -> None:
    """The uniform seam: a caller decides what a failure means."""
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(run_flow.run)))
    raises = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Raise)
        and isinstance(getattr(n.exc, "func", n.exc), ast.Name)
        and getattr(n.exc, "func", n.exc).id == "SystemExit"
    ]

    assert not raises


def test_the_step_order_is_a_valid_dependency_order() -> None:
    """Every artifact a step needs is produced by a step earlier in the list."""
    produced: set[str] = set()
    for plan in run_flow.STEP_ORDER:
        missing = set(plan.needs) - produced
        assert not missing, f"{plan.key} needs {sorted(missing)} before anything produces it"
        produced.update(plan.produces)


def test_every_error_the_flow_raises_is_documented() -> None:
    with (STEP_DIR / "step.toml").open("rb") as fh:
        manifest = tomllib.load(fh)

    documented = {e["name"] for e in manifest.get("errors", [])}

    assert {
        "missing_upstream_artifact",
        "score_column_never_written",
        "approval_corpus_mismatch",
        "approve_before_profile",
        "profile_enabled_during_approval",
        "missing_langpack_dir",
        "conflicting_approved_policy",
        "no_steps_enabled",
        "unknown_step",
        "stale_filter_output",
        "ingest_source_field_unmapped",
    } <= documented


# -- the worked examples ------------------------------------------------------
#
# Shipped beside default and tiny, the way sdg/data_designer ships rl_pref and
# customer_support_tools. An example that stops parsing is worse than none: it
# is the first thing a new user copies.

EXAMPLE_CONFIGS = ("en_c4", "vi_c4", "hi_sangraha")


@pytest.mark.parametrize("name", EXAMPLE_CONFIGS)
def test_the_worked_example_derives_six_step_configs(name) -> None:
    cfg = yaml.safe_load((STEP_DIR / "config" / f"{name}.yaml").read_text(encoding="utf-8"))

    resolved, _ = run_flow.derive(cfg)

    assert {r.plan.key for r in resolved} == {p.key for p in run_flow.STEP_ORDER}


@pytest.mark.parametrize("name", EXAMPLE_CONFIGS)
def test_the_worked_example_is_reachable_by_name(name) -> None:
    """``-c vi_c4`` resolves against the step's own config dir, so it must be there."""
    assert (STEP_DIR / "config" / f"{name}.yaml").is_file()


@pytest.mark.parametrize("name", EXAMPLE_CONFIGS)
def test_the_worked_example_ships_unapproved(name) -> None:
    """A shipped config that filters on someone else's thresholds is the trap."""
    cfg = yaml.safe_load((STEP_DIR / "config" / f"{name}.yaml").read_text(encoding="utf-8"))

    assert cfg["approve"] is None


@pytest.mark.parametrize("name", EXAMPLE_CONFIGS)
def test_the_worked_example_names_a_language_and_explicit_pack_root(name) -> None:
    """Examples must not fall back to private test data or an implicit pack."""
    cfg = yaml.safe_load((STEP_DIR / "config" / f"{name}.yaml").read_text(encoding="utf-8"))

    assert cfg["corpus"]["language"]
    assert cfg["corpus"]["langpack_dir"] not in (None, "", "bundled")


def test_the_english_example_opts_into_the_packaged_reference_root() -> None:
    cfg = yaml.safe_load((STEP_DIR / "config" / "en_c4.yaml").read_text(encoding="utf-8"))

    assert cfg["corpus"]["language"] == "en"
    assert cfg["corpus"]["langpack_dir"].endswith("/steps/curate/data/langpacks")
    assert cfg["steps"]["ingest"]["enabled"] is True


@pytest.mark.parametrize("name", ("vi_c4", "hi_sangraha"))
def test_non_english_examples_require_an_external_pack_root(name) -> None:
    cfg = yaml.safe_load((STEP_DIR / "config" / f"{name}.yaml").read_text(encoding="utf-8"))

    assert cfg["corpus"]["langpack_dir"] == "./langpacks"


@pytest.mark.parametrize("name", EXAMPLE_CONFIGS)
def test_the_worked_example_only_names_signals_its_pack_supports(name) -> None:
    """The check the hi example exists to demonstrate.

    The hi pack declares neither diacritic_ratio nor stopword_ratio_folded —
    Devanagari matras are obligatory vowels, so stripping them yields nonsense.
    Naming one is a hard error at runtime, so an example that named one would
    ship broken.
    """
    from nemotron.steps.curate.runtime import langpack

    cfg = yaml.safe_load((STEP_DIR / "config" / f"{name}.yaml").read_text(encoding="utf-8"))
    language = cfg["corpus"]["language"]
    pack = (
        langpack.load("en", PACKAGE_PACKS)
        if language == "en"
        else langpack.load(f"x-test-{language}", LANGPACK_FIXTURES)
    )
    named = cfg["steps"]["profile"].get("signals") or []

    for signal_name in named:
        required = set(signal_registry.SIGNALS[signal_name].requires)
        assert required <= set(pack.capabilities), (
            f"{name}.yaml names {signal_name}, which the {cfg['corpus']['language']} pack "
            f"cannot support (needs {sorted(required)})"
        )


def test_the_hindi_example_does_not_gate_on_sentence_end_ratio() -> None:
    """Measured: at >= 0.8 it keeps 86.8% of web but 15.9% of pdf and 7.7% of speech.

    The corpus figure is a reassuring 78.4%, so a config that gated on it would
    look reasonable while deleting the OCR and ASR sources rather than cleaning
    them. Pinned so nobody adds it back without seeing why it is absent.
    """
    cfg = yaml.safe_load((STEP_DIR / "config" / "hi_sangraha.yaml").read_text(encoding="utf-8"))

    assert cfg["approve"] is None, "an approve block here would need this reasoning revisited"
    assert "sentence_end_ratio" in (cfg["steps"]["profile"]["signals"] or []), (
        "it is profiled on purpose — measuring it is exactly how the pdf/speech "
        "collapse was found; the point is that it is measured and not gated on"
    )


# -- what the adversarial review found ----------------------------------------


def test_an_absent_optional_artifact_is_unset_not_merely_warned(tmp_path) -> None:
    """curate/audit reads declared_manifest unconditionally.

    Leaving the derived path in place turns the case preflight has just called
    legitimate into a FileNotFoundError deep inside the audit. Standalone
    curate/audit ships these as null and degrades cleanly; the flow has to
    reproduce that behaviour, not just describe it in a warning.
    """
    cfg = config(tmp_path)
    cfg["steps"]["audit"]["enabled"] = True
    resolved, paths = run_flow.derive(cfg)
    Path(paths["corpus"]).mkdir(parents=True)
    (Path(paths["corpus"]) / "a.jsonl").write_text('{"id": "x"}\n', encoding="utf-8")

    run_flow.preflight(cfg, resolved, paths)
    audit = next(r for r in resolved if r.plan.key == "audit").config

    assert audit["declared_manifest"] is None
    assert audit["ledger_glob"] is None


def test_a_misspelled_step_name_is_refused(tmp_path) -> None:
    """Otherwise the flow reports success having never attempted the stage."""
    cfg = config(tmp_path)
    cfg["steps"]["decontaminate"] = {"enabled": True}  # the real key is decontamination

    resolved, paths = run_flow.derive(cfg)
    with pytest.raises(run_flow.FlowConfigError, match="unknown step"):
        run_flow.preflight(cfg, resolved, paths)


def test_reusing_a_corpus_from_a_run_that_died_is_warned(tmp_path) -> None:
    """A corpus left by a killed writer parses cleanly and is simply short."""
    cfg = config(tmp_path)
    cfg["steps"]["subset"] = {"enabled": True}
    resolved, paths = run_flow.derive(cfg)
    Path(paths["corpus"]).mkdir(parents=True)
    (Path(paths["corpus"]) / "a.jsonl").write_text('{"id": "x", "text": "y"}\n', encoding="utf-8")
    # A manifest with no completed_at is how an auditor learns the run died.
    Path(paths["manifest"]).write_text(json.dumps({"producer": {"started_at": "t"}}), encoding="utf-8")

    warnings = run_flow.preflight(cfg, resolved, paths)

    assert any("did not reach its write barrier" in w for w in warnings)


def test_reusing_a_corpus_with_no_manifest_at_all_is_warned(tmp_path) -> None:
    cfg = config(tmp_path)
    cfg["steps"]["subset"] = {"enabled": True}
    resolved, paths = run_flow.derive(cfg)
    Path(paths["corpus"]).mkdir(parents=True)
    (Path(paths["corpus"]) / "a.jsonl").write_text('{"id": "x", "text": "y"}\n', encoding="utf-8")

    warnings = run_flow.preflight(cfg, resolved, paths)

    assert any("nothing to say whether the run that wrote it finished" in w for w in warnings)


def test_a_step_failing_still_writes_a_report_saying_how_far_it_got(tmp_path, monkeypatch) -> None:
    """Otherwise the half-written artifacts sit next to a PREVIOUS run's report."""
    cfg = config(tmp_path)

    def boom(_cfg):
        raise RuntimeError("profile exploded")

    monkeypatch.setattr(run_flow, "step_runner", lambda key: boom)

    with pytest.raises(RuntimeError, match="profile exploded"):
        run_flow.run(cfg)

    report = json.loads((Path(cfg["output_root"]) / "flow_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    failed = next(s for s in report["steps"] if s["status"] == "failed")
    assert failed["step_id"] == "curate/profile"
    assert "profile exploded" in failed["error"]


def test_policy_applied_is_false_when_nothing_applied_it(tmp_path) -> None:
    """Promoting a policy is not applying one, and the report must not conflate them."""
    cfg = config(tmp_path)
    cfg["corpus"]["source_field_in_source"] = "source"
    cfg["corpus"]["id_field_in_source"] = True
    cfg["steps"]["ingest"] = {"enabled": True}
    run_flow.run(cfg)  # produce both candidates and the prepared corpus
    cfg["approve"] = approve_block()
    cfg["steps"]["profile"]["enabled"] = False
    cfg["steps"]["filter"]["enabled"] = False

    report = run_flow.run(cfg)

    assert report["policy_promoted"] is True
    assert report["policy_applied"] is False


def test_each_steps_own_warnings_reach_the_flow_report(tmp_path) -> None:
    """step.toml says the report holds every warning; dropping them makes that false."""
    cfg = config(tmp_path)
    cfg["steps"]["subset"] = {"enabled": True, "token_budgets": [10], "tokenizer": None}
    resolved, paths = run_flow.derive(cfg)
    Path(paths["corpus"]).mkdir(parents=True)
    (Path(paths["corpus"]) / "a.jsonl").write_text(
        "".join(json.dumps({"id": f"d{i}", "source": "s", "text": "one two three four"}) + "\n" for i in range(5)),
        encoding="utf-8",
    )

    report = run_flow.run(cfg)
    subset = next(s for s in report["steps"] if s["step_id"] == "curate/subset")

    assert "warnings" in subset


def test_plan_writes_the_plan_file_it_promises(tmp_path) -> None:
    """--help, README and step.toml all say it does; only a real run did."""
    cfg = config(tmp_path)

    run_flow.plan(cfg, dry_run=True)

    written = json.loads((Path(cfg["output_root"]) / "flow_plan.json").read_text(encoding="utf-8"))
    assert written["dry_run"] is True
    assert {s["key"] for s in written["steps"]} == {p.key for p in run_flow.STEP_ORDER}


def test_plan_runs_the_approval_gate_without_leaving_a_policy(tmp_path) -> None:
    """A preview that skipped the gate would miss the refusals it exists to surface."""
    cfg = profiled(tmp_path)
    cfg["approve"] = approve_block()
    _, paths, _ = run_flow.plan(cfg, dry_run=True)

    assert not Path(paths["approved_policy"]).exists(), "a dry run must not approve anything"

    # ...and the refusal still fires on a preview.
    other = tmp_path / "elsewhere"
    other.mkdir()
    (other / "c.jsonl").write_text(
        json.dumps({"id": "z1", "source": "web", "text": "an entirely unrelated document"}) + "\n",
        encoding="utf-8",
    )
    cfg["corpus"]["input"] = str(other / "*.jsonl")
    with pytest.raises(run_flow.FlowConfigError, match="granted against corpus"):
        run_flow.plan(cfg, dry_run=True)


def test_plan_and_run_derive_the_same_thing(tmp_path) -> None:
    """A preview describing a different run from the real one is worse than none."""
    cfg = config(tmp_path)
    preview, _, _ = run_flow.plan(cfg, dry_run=True)
    previewed = {r.plan.key: r.config for r in preview}

    run_flow.run(cfg)
    executed = json.loads((Path(cfg["output_root"]) / "flow_plan.json").read_text(encoding="utf-8"))

    assert {s["key"]: s["config"] for s in executed["steps"]} == previewed


def test_the_fingerprint_covers_content_not_just_identity(tmp_path) -> None:
    """Two corpora sharing an id scheme must not fingerprint identically.

    make_key returns the id alone when id_field is set, and every shipped config
    sets one — so a fingerprint built on it would verify an approval against any
    corpus that numbered its documents the same way, while reporting the corpus
    as checked.
    """
    from nemotron.steps.curate.runtime import integrity

    ids = [f"doc-{i}" for i in range(5)]
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    for path, body in ((a, "one corpus"), (b, "a completely different corpus")):
        path.write_text(
            "".join(json.dumps({"id": i, "text": f"{body} {i}"}) + "\n" for i in ids),
            encoding="utf-8",
        )

    assert integrity.corpus_fingerprint(str(a), "text", "id") != integrity.corpus_fingerprint(str(b), "text", "id")


def test_the_fingerprint_reads_a_directory_a_glob_and_a_list_alike(tmp_path) -> None:
    """Every other curate step accepts all three spellings of corpus.input."""
    from nemotron.steps.curate.runtime import integrity

    directory = tmp_path / "corpus"
    directory.mkdir()
    (directory / "a.jsonl").write_text(json.dumps({"id": "1", "text": "hello there"}) + "\n", encoding="utf-8")

    by_dir = integrity.corpus_fingerprint(str(directory), "text", "id")
    by_glob = integrity.corpus_fingerprint(str(directory / "*.jsonl"), "text", "id")
    by_list = integrity.corpus_fingerprint([str(directory / "a.jsonl")], "text", "id")

    assert by_dir == by_glob == by_list


def test_filtering_twice_into_one_output_root_is_refused(tmp_path) -> None:
    """The two-run approval workflow filters into the same place twice. The step
    does not clear the directory, so the second corpus lands beside the first and
    the union is what every later step reads: the ledger counts more out than in,
    the audit calls the surplus unexplained, and subset refuses on duplicate ids.
    None of those say 'stale directory', so the flow says it before any work."""
    cfg = config(tmp_path)
    cfg["steps"]["filter"]["enabled"] = True
    resolved, paths = run_flow.derive(cfg)
    corpus = Path(paths["corpus"])
    corpus.mkdir(parents=True)
    (corpus / "cfe28204c768.jsonl").write_text(
        json.dumps({"id": "d000", "text": "a corpus an earlier run left here"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(run_flow.FlowConfigError, match="already holds"):
        run_flow.preflight(cfg, resolved, paths)


def test_an_empty_output_root_is_not_mistaken_for_a_stale_one(tmp_path) -> None:
    """The reports the previous run wrote are not shards, and a first run must
    not be refused because the directory exists."""
    cfg = config(tmp_path)
    cfg["steps"]["filter"]["enabled"] = True
    resolved, paths = run_flow.derive(cfg)
    corpus = Path(paths["corpus"])
    corpus.mkdir(parents=True)
    (corpus / "run_manifest.json").write_text("{}", encoding="utf-8")

    run_flow.preflight(cfg, resolved, paths)  # must not raise


def test_a_line_that_parses_to_something_other_than_a_document_is_skipped(tmp_path) -> None:
    """Binary input can yield a line that is valid JSON but not an object, and
    calling .get on it raised AttributeError instead of being skipped."""
    from nemotron.steps.curate.runtime import integrity

    clean, noisy = tmp_path / "clean.jsonl", tmp_path / "noisy.jsonl"
    record = json.dumps({"id": "1", "text": "hello there"}) + "\n"
    clean.write_text(record, encoding="utf-8")
    noisy.write_text("5\n[1, 2]\n" + record + '"a string"\n', encoding="utf-8")

    assert integrity.corpus_fingerprint(str(noisy), "text", "id") == integrity.corpus_fingerprint(
        str(clean), "text", "id"
    )


def test_the_fingerprint_refuses_a_corpus_it_cannot_read(tmp_path) -> None:
    """expand_inputs resolves parquet because ingest reads it, but this reader is
    JSONL-only. Returning the digest of nothing would be worse than refusing."""
    from nemotron.steps.curate.runtime import integrity

    binary = tmp_path / "part_0.parquet"
    binary.write_bytes(b"PAR1\x00\x01\x02\x03rows and columns, not lines\xff\xfePAR1")

    with pytest.raises(integrity.UnreadableCorpusError, match="JSONL-only"):
        integrity.corpus_fingerprint(str(binary), "text", "id")


def test_two_unreadable_corpora_cannot_verify_as_each_other(tmp_path) -> None:
    """The substitution the fingerprint exists to catch: every unreadable corpus
    used to digest to one constant, equal to the digest of no input at all, so an
    approval granted against one verified cleanly against any other."""
    from nemotron.steps.curate.runtime import integrity

    hindi, english = tmp_path / "hi.parquet", tmp_path / "en.parquet"
    hindi.write_bytes(b"PAR1\x00\xff\x01 hindi rows PAR1")
    english.write_bytes(b"PAR1\x00\xff\x02 english rows PAR1")

    for path in (hindi, english):
        with pytest.raises(integrity.UnreadableCorpusError):
            integrity.corpus_fingerprint(str(path), "text", "id")


def test_an_absent_corpus_is_refused_rather_than_fingerprinted(tmp_path) -> None:
    """This used to assert the opposite, on a premise that does not hold.

    The reasoning given was that "resolving to no file at all is how preflight
    asks whether an artifact exists". It is not: preflight uses _artifact_exists,
    which goes through expand_inputs. integrity.corpus_fingerprint has exactly
    one caller — the approval check at run_flow.materialise_policy — and there,
    returning a digest for an absent corpus meant every absent corpus shared one
    value. An approval granted against nothing would then verify cleanly against
    anything else that was also nothing.
    """
    from nemotron.steps.curate.runtime import integrity

    with pytest.raises(integrity.UnreadableCorpusError, match="matched no files"):
        integrity.corpus_fingerprint(str(tmp_path / "nothing" / "*.jsonl"), "text", "id")


def test_the_approval_is_verified_against_the_corpus_the_filter_reads(tmp_path) -> None:
    """With ingest enabled, the corpus the thresholds get applied to is the
    ingested JSONL — which is also what the profile measured. Verifying
    corpus.input instead compared a corpus nobody profiled: ingest mints ids, so
    the two differ even over identical text, and every approval was refused."""
    from nemotron.steps.curate.runtime import integrity

    # Its own corpus: the shared one repeats text, and a content-derived id
    # cannot tell two identical documents apart, so ingest refuses it.
    raw = tmp_path / "unique"
    raw.mkdir()
    with (raw / "part_0.jsonl").open("w", encoding="utf-8") as fh:
        for i in range(40):
            fh.write(json.dumps({"id": f"d{i:03d}", "source": "web", "text": f"document {i} " + "body " * 20}) + "\n")

    cfg = config(tmp_path)
    cfg["corpus"]["input"] = str(raw / "*.jsonl")
    # The raw records above carry a `source` column, so ingest is told to map it.
    # Without this the flow refuses: corpus.source_field names what every step
    # AFTER ingest reads, and ingest would have nothing to write into it.
    cfg["corpus"]["source_field_in_source"] = "source"
    cfg["steps"]["ingest"] = {"enabled": True}
    run_flow.run(cfg)
    cfg["steps"]["profile"]["enabled"] = False
    cfg["steps"]["filter"]["enabled"] = True
    cfg["approve"] = approve_block()
    resolved, paths = run_flow.derive(cfg)

    warnings = run_flow.materialise_policy(cfg, resolved, paths)

    ingested = next(r for r in resolved if r.plan.key == "filter").config["input_glob"]
    assert ingested != cfg["corpus"]["input"]
    verified = integrity.corpus_fingerprint(ingested, "text", "id")
    assert verified != integrity.corpus_fingerprint(cfg["corpus"]["input"], "text", "id")
    assert any(verified in warning for warning in warnings)


def test_profile_and_the_flow_compute_the_same_fingerprint(tmp_path) -> None:
    """The gate is meaningless unless producer and verifier measure one quantity."""
    from nemotron.steps.curate.runtime import integrity
    from nemotron.steps.curate.scripts import run_profile

    path = tmp_path / "c.jsonl"
    path.write_text(
        "".join(json.dumps({"id": f"d{i}", "source": "s", "text": f"document {i} body"}) + "\n" for i in range(6)),
        encoding="utf-8",
    )

    _, stats = run_profile.count_sources([str(path)], "source", "text", "id")

    assert stats["fingerprint"] == integrity.corpus_fingerprint(str(path), "text", "id")


# -- the source column has two halves and only one is obvious ------------------
#
# corpus.source_field names the column every step AFTER ingest reads. What ingest
# WRITES comes from source_field_in_source or source_value. Naming only the first
# is the natural thing to write, produces no source column at all, and every
# downstream step then falls back to "each shard is its own source" — reporting
# per-shard figures that read exactly like per-corpus ones.


def _ingest_cfg(tmp_path, **corpus_over):
    # A real file, because preflight now refuses a corpus.input that matches
    # nothing — the check that tells a user their relative path resolved against
    # the working directory rather than against the config.
    raw = tmp_path / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "part.parquet").write_bytes(b"")

    corpus = {
        "input": str(tmp_path / "raw" / "*.parquet"),
        "text_field": "text",
        "id_field": "id",
        "language": "x-test-vi",
        "langpack_dir": str(LANGPACK_FIXTURES),
    }
    corpus.update(corpus_over)
    return {
        "corpus": corpus,
        "output_root": str(tmp_path / "out"),
        "steps": {"ingest": {"enabled": True}, "profile": {"enabled": True}},
    }


def test_a_source_field_ingest_cannot_write_is_refused(tmp_path) -> None:
    cfg = _ingest_cfg(tmp_path, source_field="type")

    with pytest.raises(run_flow.FlowConfigError, match="nothing to write into it"):
        run_flow.plan(cfg, dry_run=True)


def test_a_constant_source_value_satisfies_it(tmp_path) -> None:
    cfg = _ingest_cfg(tmp_path, source_field="source", source_value="c4_vi")

    resolved, _, _ = run_flow.plan(cfg, dry_run=True)

    by_key = {r.plan.key: r.config for r in resolved}
    assert by_key["ingest"]["source"] == "c4_vi"


def test_a_source_column_in_the_raw_data_satisfies_it(tmp_path) -> None:
    cfg = _ingest_cfg(tmp_path, source_field="source", source_field_in_source="type")

    resolved, _, _ = run_flow.plan(cfg, dry_run=True)

    by_key = {r.plan.key: r.config for r in resolved}
    assert by_key["ingest"]["source_from"] == "type"


def test_the_check_does_not_fire_when_ingest_is_disabled(tmp_path) -> None:
    """Without ingest the corpus already carries whatever column it carries."""
    cfg = _ingest_cfg(tmp_path, source_field="type")
    cfg["steps"]["ingest"]["enabled"] = False

    resolved, _, _ = run_flow.plan(cfg, dry_run=True)

    assert {r.plan.key for r in resolved} == {p.key for p in run_flow.STEP_ORDER}


# -- ingest and approve must be able to coexist --------------------------------
#
# materialise_policy verifies the approval against the corpus the FILTER reads,
# which is right: ingest mints ids, so the raw input and the ingested corpus are
# different data and the profile measured the latter. But plan() runs it before
# any step, so with steps.ingest enabled that corpus does not exist yet and the
# check compared a real fingerprint against the digest of nothing. Two of the six
# steps could not be enabled together.


def test_a_policy_can_be_approved_with_ingest_enabled(tmp_path) -> None:
    """The corpus ingest will produce is not there at plan time; that is expected."""
    # Every document distinct: a content-derived id cannot tell two identical
    # texts apart, and ingest refuses rather than choosing for you.
    # NOT tmp_path/"raw": corpus_files() writes its own part_0.jsonl there and
    # config() is called below, so a corpus written first would be overwritten.
    raw = tmp_path / "unique_raw"
    raw.mkdir()
    with (raw / "part_0.jsonl").open("w", encoding="utf-8") as fh:
        for i in range(40):
            body = " ".join(f"w{i}x{j}" for j in range(30))
            fh.write(json.dumps({"id": f"d{i:03d}", "source": "web", "text": f"document {i} {body}"}) + "\n")

    cfg = config(tmp_path)
    cfg["corpus"]["input"] = str(raw / "*.jsonl")
    cfg["corpus"]["source_field_in_source"] = "source"
    cfg["steps"]["ingest"] = {"enabled": True}
    run_flow.run(cfg)

    cfg["steps"]["profile"]["enabled"] = False
    cfg["steps"]["filter"]["enabled"] = True
    cfg["approve"] = approve_block()
    resolved, paths = run_flow.derive(cfg)

    # Must not raise: the ingested corpus exists now, and this is the same shape
    # a second run of the flow takes.
    warnings = run_flow.materialise_policy(cfg, resolved, paths)

    assert any("fingerprint verified" in w for w in warnings)


def test_approval_is_reverified_after_ingest_before_filtering(tmp_path, monkeypatch) -> None:
    raw = tmp_path / "changing_raw"
    raw.mkdir()
    path = raw / "part_0.jsonl"

    def write_raw(changed: bool = False) -> None:
        with path.open("w", encoding="utf-8") as fh:
            for i in range(40):
                text = f"document {i} " + " ".join(f"word{i}_{j}" for j in range(20))
                if changed and i == 0:
                    text += " changed after approval profiling"
                fh.write(json.dumps({"source": "web", "text": text}) + "\n")

    write_raw()
    cfg = config(tmp_path)
    cfg["corpus"]["input"] = str(path)
    cfg["corpus"]["source_field_in_source"] = "source"
    cfg["steps"]["ingest"] = {"enabled": True}
    run_flow.run(cfg)

    write_raw(changed=True)
    cfg["steps"]["profile"]["enabled"] = False
    cfg["steps"]["filter"]["enabled"] = True
    cfg["approve"] = approve_block()
    filter_called = False
    real_step_runner = run_flow.step_runner

    def runner(key):
        if key != "filter":
            return real_step_runner(key)

        def fail_if_called(_cfg):
            nonlocal filter_called
            filter_called = True
            return {}

        return fail_if_called

    monkeypatch.setattr(run_flow, "step_runner", runner)

    with pytest.raises(run_flow.FlowConfigError, match="granted against corpus"):
        run_flow.run(cfg)

    assert filter_called is False


def test_a_policy_is_refused_when_the_corpus_it_names_is_absent(tmp_path) -> None:
    """And the refusal must name the emptiness, not compare a digest of nothing."""
    from nemotron.steps.curate.runtime import integrity

    cfg = config(tmp_path)
    cfg["steps"]["ingest"] = {"enabled": True}
    cfg["steps"]["profile"]["enabled"] = False
    cfg["approve"] = approve_block()
    resolved, paths = run_flow.derive(cfg)

    with pytest.raises((integrity.UnreadableCorpusError, run_flow.FlowConfigError)) as caught:
        run_flow.materialise_policy(cfg, resolved, paths)

    assert "matched no files" in str(caught.value) or "does not exist" in str(caught.value)


def test_an_unmatched_corpus_names_the_directory_it_looked_in(tmp_path, monkeypatch) -> None:
    """A relative path in a config resolves against the working directory, not
    against the config's own directory, so the same file names two different
    places depending on where the command was run. The refusal has to say which
    one applied, or the reader is left guessing.
    """
    monkeypatch.chdir(tmp_path)
    cfg = {
        "corpus": {"input": "./raw_jsonl/*.jsonl", "text_field": "text", "language": "x-test-vi"},
        "output_root": str(tmp_path / "out"),
        "steps": {"filter": {"enabled": True}},
    }

    with pytest.raises(run_flow.FlowConfigError) as caught:
        run_flow.plan(cfg, dry_run=True)

    message = str(caught.value)
    assert "corpus.input matched no files" in message
    assert str(tmp_path) in message, "the working directory it resolved against must be named"
    assert "not against the directory holding the config" in message


def test_a_corpus_the_run_will_download_is_not_refused_for_being_absent(tmp_path) -> None:
    """curate/nemo_curator calls snapshot_download before it resolves its input
    glob, so a corpus materialised from a Hugging Face snapshot legitimately does
    not exist when preflight runs. Refusing it would block a working config."""
    cfg = {
        "corpus": {"input": str(tmp_path / "never" / "*.jsonl"), "text_field": "text"},
        "output_root": str(tmp_path / "out"),
        "steps": {"filter": {"enabled": True, "dataset": {"repo_id": "x/y", "repo_type": "dataset"}}},
    }

    resolved, _, _ = run_flow.plan(cfg, dry_run=True)

    assert any(r.plan.key == "filter" and r.enabled for r in resolved)
