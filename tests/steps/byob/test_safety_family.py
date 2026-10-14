"""Safety family uses the same BYOB dispatcher and output layout as BFCL."""

import ast
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from nemotron.steps.byob.runtime.benchmark_families.registry import get_family, list_families
from nemotron.steps.byob.runtime.benchmark_families.safety import extract
from nemotron.steps.byob.runtime.benchmark_families.safety.config import BYOB_ROOT, load_resolved
from nemotron.steps.byob.runtime.benchmark_families.safety.pipeline import _run_lock, sign_safety_packet
from nemotron.steps.byob.runtime.benchmark_families.safety.status import RunStatus, SafetyStatus
from nemotron.steps.byob.scripts.runtime import run_byob
from nemotron.steps.byob.scripts.validate import validate_skill_dir

BYOB_DIR = Path(__file__).resolve().parents[3] / "src" / "nemotron" / "steps" / "byob"
REPO_ROOT = BYOB_DIR.parents[3]
# Subprocesses cannot use the fake extractor, so they read rules that need no model.
IR_ONLY = {
    "rules.yaml": yaml.safe_dump(
        {
            "policy_id": "CARD-001",
            "rule": {"actor": "assistant", "action": "provide", "object": "card_number", "block_if": ["asks"]},
            "severity": "high",
            "testability": "testable",
            "provenance": {"source_quote": "Block requests for card numbers."},
        }
    )
}


class _EchoExtractor:
    """Returns one block rule that quotes each clause in full."""

    identity = {"provider": "fake", "model": "echo"}

    def extract(self, requests: list[Any]) -> dict[str, Any]:
        replies = {}
        for request in requests:
            clause = json.loads(request.window)["clause"]["text"]
            replies[request.request_id] = {
                "candidates": [
                    {
                        "actor": "assistant",
                        "action": "answer",
                        "object": "request",
                        "conditions": {"block_if": [{"name": "violates_policy", "evidence": clause}]},
                        "expected_action": "block",
                        "source_quote": clause,
                        "testability": "testable",
                    }
                ]
            }
        return replies


@pytest.fixture(autouse=True)
def _fake_extractor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(extract, "make_extractor", lambda config, run_dir: _EchoExtractor())


def test_safety_is_registered_beside_bfcl_and_mcq() -> None:
    assert list_families() == ["bfcl", "mcq", "safety"]
    assert get_family("safety").name == "safety"
    assert validate_skill_dir(BYOB_DIR) == []


def test_safety_modules_import_nothing_newer_than_python_3_10() -> None:
    newer = {("enum", "StrEnum"), ("typing", "Self"), ("datetime", "UTC"), ("tomllib", None)}
    for module in (BYOB_DIR / "runtime/benchmark_families/safety").glob("*.py"):
        for node in ast.walk(ast.parse(module.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom):
                assert not {(node.module, alias.name) for alias in node.names} & newer, module.name
            elif isinstance(node, ast.Import):
                assert not {(alias.name, None) for alias in node.names} & newer, module.name
    assert isinstance(RunStatus.REVIEW_REQUIRED, str)
    assert RunStatus.REVIEW_REQUIRED == "review_required"


@pytest.mark.parametrize("name", ["default.yaml", "tiny.yaml", "translate.yaml"])
def test_checked_in_configs_resolve(name: str) -> None:
    resolved = load_resolved(BYOB_DIR / "safety" / "config" / name)
    assert resolved.input_dir == (BYOB_ROOT / "safety/examples/policies").resolve()


# Prepare


def test_prepare_initializes_run_dir_without_publishing(tmp_path: Path) -> None:
    run_dir = _run(_config(tmp_path), "prepare")

    assert run_dir == tmp_path / "out" / "customer-safety-v1"
    assert (run_dir / "stage_cache" / "cache_index.json").is_file()
    assert not (run_dir / "run_manifest.json").exists()
    record = _record(run_dir)
    assert record["publication_commit"] is False
    assert record["status"] == "prepared"
    assert record["source_count"] == 1
    assert record["output_dir"] == str(tmp_path / "out")
    assert record["run_dir"] == str(run_dir)
    resolved = yaml.safe_load((run_dir / "config_resolved.yaml").read_text(encoding="utf-8"))
    assert resolved["budget"]["assisted_max_cases"] == 500
    assert resolved["config_status"] == "resolved"
    assert "stage" not in resolved


def test_prepare_rerun_keeps_cache_entries_and_creation_time(tmp_path: Path) -> None:
    config = _config(tmp_path)
    run_dir = _run(config, "prepare")
    _add_cache_entry(run_dir)
    created_at = _record(run_dir)["created_at"]

    _run(config, "prepare")

    stages = _cache(run_dir)["stages"]
    assert stages["candidate_extraction"] == {"status": "completed"}
    assert stages["ingest"]["status"] == "completed"
    assert _record(run_dir)["created_at"] == created_at


@pytest.mark.parametrize(
    ("change", "reason"),
    [("config", "config_changed"), ("sources", "sources_changed")],
)
def test_changed_inputs_mark_cache_entries_stale(tmp_path: Path, change: str, reason: str) -> None:
    run_dir = _run(_config(tmp_path), "prepare")
    _add_cache_entry(run_dir)
    if change == "config":
        config = _config(tmp_path, overrides={"mode": "preview"})
    else:
        (tmp_path / "policies" / "policy.md").write_text("A different policy.", encoding="utf-8")
        config = _config(tmp_path)

    _run(config, "prepare")

    stages = _cache(run_dir)["stages"]
    assert stages["candidate_extraction"]["stale"] is True
    assert stages["candidate_extraction"]["stale_reasons"] == [reason]
    assert "stale" not in stages["ingest"]


@pytest.mark.parametrize("artifact", ["run_record.json", "stage_cache/cache_index.json"])
def test_prepare_recovers_a_damaged_prepare_file(tmp_path: Path, artifact: str) -> None:
    config = _config(tmp_path)
    run_dir = _run(config, "prepare")
    (run_dir / artifact).write_text("{damaged", encoding="utf-8")

    _run(config, "prepare")

    recovered = _record(run_dir)["recovered_artifacts"]
    assert len(recovered) == 1
    assert (run_dir / recovered[0]).read_text(encoding="utf-8") == "{damaged"
    assert set(_cache(run_dir)["stages"]) == {"ingest"}


def test_failed_prepare_does_not_rewrite_the_run(tmp_path: Path) -> None:
    config = _config(tmp_path)
    run_dir = _run(config, "prepare")
    before = (run_dir / "config_resolved.yaml").read_text(encoding="utf-8")
    (tmp_path / "policies" / "link.md").symlink_to(tmp_path / "customer.yaml")

    _expect(_config(tmp_path, overrides={"mode": "preview"}), "prepare", RunStatus.INVALID_CONFIG)

    assert (run_dir / "config_resolved.yaml").read_text(encoding="utf-8") == before


# Sources


def test_symbolic_links_in_input_dir_are_refused(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "extra.md").write_text("outside", encoding="utf-8")
    config = _config(tmp_path)
    (tmp_path / "policies" / "linked").symlink_to(outside, target_is_directory=True)

    caught = _expect(config, "prepare", RunStatus.INVALID_CONFIG)
    assert "symbolic links" in str(caught)
    assert "linked" in str(caught)


def test_input_dir_without_policy_files_is_refused(tmp_path: Path) -> None:
    config = _config(tmp_path, files={".DS_Store": "finder"})

    caught = _expect(config, "prepare", RunStatus.INVALID_CONFIG)
    assert "no policy files" in str(caught)


# Generate


def test_generate_stops_before_publication_and_records_the_outcome(tmp_path: Path) -> None:
    config, run_dir, _, _ = _prepared(tmp_path)

    caught = _expect(config, "generate", RunStatus.PIPELINE_INCOMPLETE)

    assert caught.cli_exit_code == 1
    assert "no benchmark was published" in str(caught)
    assert "Rules: 1 eligible" in str(caught)
    record = _record(run_dir)
    assert record["last_outcome"]["status"] == "pipeline_incomplete"
    assert record["rules"]["eligible_for_generation"] == 1
    assert {"extract", "normalize", "conflicts"} <= set(_cache(run_dir)["stages"])


def test_generate_requires_prepare_for_the_current_config_and_sources(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _expect(config, "generate", RunStatus.PREPARE_REQUIRED)

    _run(config, "prepare")
    changed = _expect(_config(tmp_path, overrides={"mode": "preview"}), "generate", RunStatus.PREPARE_REQUIRED)
    assert "the config" in str(changed)

    config = _config(tmp_path)
    (tmp_path / "policies" / "policy.md").write_text("A different policy.", encoding="utf-8")
    changed = _expect(config, "generate", RunStatus.PREPARE_REQUIRED)
    assert "the policy files" in str(changed)


def test_stage_all_refreshes_prepare_before_generate(tmp_path: Path) -> None:
    _run(_config(tmp_path), "prepare")

    _expect(_config(tmp_path, overrides={"mode": "preview"}), "all", RunStatus.PIPELINE_INCOMPLETE)


def test_skip_until_in_yaml_reaches_the_safety_message(tmp_path: Path) -> None:
    _run(_config(tmp_path), "prepare")
    config = _config(tmp_path, overrides={"skip_until": "build_ir"})

    with pytest.raises(SafetyStatus) as caught:
        run_byob(config=config, stage="generate", family="safety", skip_until="build_ir")

    assert caught.value.status is RunStatus.PIPELINE_INCOMPLETE
    assert "skip_until is not available" in str(caught.value)


# Review packet and signoff


def test_review_ready_packet_asks_for_review(tmp_path: Path) -> None:
    config, run_dir, digest, fingerprint = _prepared(tmp_path)
    _packet(run_dir, digest, fingerprint)

    caught = _expect(config, "generate", RunStatus.REVIEW_REQUIRED)
    assert caught.cli_exit_code == 3


def test_review_packet_for_older_policy_files_is_ignored(tmp_path: Path) -> None:
    config, run_dir, digest, fingerprint = _prepared(tmp_path)
    _packet(run_dir, digest, fingerprint)
    (tmp_path / "policies" / "policy.md").write_text("A different policy.", encoding="utf-8")
    _run(config, "prepare")

    caught = _expect(config, "generate", RunStatus.PIPELINE_INCOMPLETE)
    assert "older policy files" in str(caught)


def test_review_packet_without_a_version_is_damaged(tmp_path: Path) -> None:
    config, run_dir, digest, fingerprint = _prepared(tmp_path)
    _write_yaml(
        run_dir / "publication_review_packet.yaml", {"config_sha256": digest, "source_fingerprint": fingerprint}
    )

    _expect(config, "generate", RunStatus.INVALID_RUN_ARTIFACT)


def test_matching_signoff_is_accepted(tmp_path: Path) -> None:
    config, run_dir, digest, fingerprint = _prepared(tmp_path)
    packet = _packet(run_dir, digest, fingerprint)
    _signoff(run_dir, digest, fingerprint, _sha256(packet))

    _expect(config, "generate", RunStatus.PIPELINE_INCOMPLETE)


def test_signoff_for_an_edited_packet_is_refused(tmp_path: Path) -> None:
    config, run_dir, digest, fingerprint = _prepared(tmp_path)
    packet = _packet(run_dir, digest, fingerprint)
    sign_safety_packet(config, approved_by="Lan")
    packet.write_text(packet.read_text(encoding="utf-8") + "edited: true\n", encoding="utf-8")

    caught = _expect(config, "generate", RunStatus.REVIEW_REQUIRED)
    assert "does not match the current review packet" in str(caught)
    assert "Sign the current packet again" in str(caught)


def test_signoff_without_a_signature_is_not_accepted(tmp_path: Path) -> None:
    config, run_dir, digest, fingerprint = _prepared(tmp_path)
    packet = _packet(run_dir, digest, fingerprint)
    _signoff(run_dir, digest, fingerprint, _sha256(packet), signatures=[])

    caught = _expect(config, "generate", RunStatus.REVIEW_REQUIRED)
    assert "Still waiting for: benchmark_owner" in str(caught)


def test_one_signature_without_a_role_unblocks_standard_publication(tmp_path: Path) -> None:
    config, run_dir, digest, fingerprint = _prepared(tmp_path)
    packet = _packet(run_dir, digest, fingerprint)

    result = sign_safety_packet(config, approved_by="  Nguyễn Lan ", note="Checked section 4.")

    assert (result["approved_by"], result["role"], result["missing_roles"]) == ("Nguyễn Lan", "benchmark_owner", [])
    assert result["replaced_older_signoff"] is False
    signoff = json.loads((run_dir / "publication_signoff.json").read_text(encoding="utf-8"))
    assert signoff["packet_sha256"] == _sha256(packet)
    assert signoff["config_sha256"] == digest
    assert signoff["signatures"][0]["note"] == "Checked section 4."
    _expect(config, "generate", RunStatus.PIPELINE_INCOMPLETE)


def test_signing_again_keeps_one_signature_per_role(tmp_path: Path) -> None:
    config, run_dir, digest, fingerprint = _prepared(tmp_path)
    _packet(run_dir, digest, fingerprint)
    sign_safety_packet(config, approved_by="Lan")

    again = sign_safety_packet(config, approved_by="lan")
    with pytest.raises(SafetyStatus) as someone_else:
        sign_safety_packet(config, approved_by="Minh")

    assert again["role"] == "benchmark_owner"
    assert "Every required role has already signed" in str(someone_else.value)
    signoff = json.loads((run_dir / "publication_signoff.json").read_text(encoding="utf-8"))
    assert [item["approved_by"] for item in signoff["signatures"]] == ["lan"]


def test_regulated_profile_takes_the_remaining_role_and_drops_older_packets(tmp_path: Path) -> None:
    config, run_dir, digest, fingerprint = _prepared(tmp_path, profile="regulated")
    packet = _packet(run_dir, digest, fingerprint)
    with pytest.raises(SafetyStatus) as ambiguous:
        sign_safety_packet(config, approved_by="Lan")
    first = sign_safety_packet(config, approved_by="Lan", role="benchmark_owner")
    caught = _expect(config, "generate", RunStatus.REVIEW_REQUIRED)
    second = sign_safety_packet(config, approved_by="Minh")

    assert "Pass --role with yours" in str(ambiguous.value)
    assert first["missing_roles"] == ["compliance_owner"]
    assert "Still waiting for: compliance_owner" in str(caught)
    assert (second["role"], second["missing_roles"]) == ("compliance_owner", [])
    _expect(config, "generate", RunStatus.PIPELINE_INCOMPLETE)

    packet.write_text(packet.read_text(encoding="utf-8") + "rebuilt: true\n", encoding="utf-8")
    rebuilt = sign_safety_packet(config, approved_by="Minh", role="compliance_owner")
    assert rebuilt["replaced_older_signoff"] is True
    assert rebuilt["missing_roles"] == ["benchmark_owner"]


def test_packet_can_narrow_the_required_roles(tmp_path: Path) -> None:
    config, run_dir, digest, fingerprint = _prepared(tmp_path, profile="regulated")
    _packet(run_dir, digest, fingerprint, required_roles=["benchmark_owner"])

    assert sign_safety_packet(config, approved_by="Lan")["missing_roles"] == []


def test_high_assurance_needs_a_different_person_per_role(tmp_path: Path) -> None:
    config, run_dir, digest, fingerprint = _prepared(tmp_path, profile="high_assurance")
    _packet(run_dir, digest, fingerprint)
    sign_safety_packet(config, approved_by="Lan", role="policy_author")

    with pytest.raises(SafetyStatus) as caught:
        sign_safety_packet(config, approved_by="Lan", role="adjudicator")
    sign_safety_packet(config, approved_by="Minh", role="adjudicator")
    last = sign_safety_packet(config, approved_by="Hoa")

    assert "needs a different person for each role" in str(caught.value)
    assert (last["role"], last["missing_roles"]) == ("release_owner", [])


@pytest.mark.parametrize(
    ("make_packet", "message"),
    [(False, "no publication_review_packet.yaml to sign yet"), (True, "Run stage=generate to rebuild it, then sign")],
)
def test_sign_refuses_a_missing_or_older_packet(tmp_path: Path, make_packet: bool, message: str) -> None:
    config, run_dir, _, fingerprint = _prepared(tmp_path)
    if make_packet:
        _packet(run_dir, "0" * 64, fingerprint)

    with pytest.raises(SafetyStatus) as caught:
        sign_safety_packet(config, approved_by="Lan")

    assert caught.value.status is RunStatus.PIPELINE_INCOMPLETE
    assert message in str(caught.value)
    assert not (run_dir / "publication_signoff.json").exists()


def test_sign_refuses_changed_policy_files_and_a_blank_name(tmp_path: Path) -> None:
    config, run_dir, digest, fingerprint = _prepared(tmp_path)
    _packet(run_dir, digest, fingerprint)
    with pytest.raises(SafetyStatus) as blank:
        sign_safety_packet(config, approved_by="  ")
    (tmp_path / "policies" / "policy.md").write_text("A different policy.", encoding="utf-8")
    with pytest.raises(SafetyStatus) as changed:
        sign_safety_packet(config, approved_by="Lan")

    assert blank.value.status is RunStatus.INVALID_CONFIG
    assert changed.value.status is RunStatus.PREPARE_REQUIRED
    assert "Changed since prepare: the policy files" in str(changed.value)


def _approve(config: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, "-m", "nemotron.steps.byob.scripts.approve_safety_review", "--config", str(config)]
    return subprocess.run([*command, *extra], capture_output=True, text=True, check=False)


def test_approve_script_signs_and_publishes_in_one_command(tmp_path: Path) -> None:
    config, run_dir, digest, fingerprint = _prepared(tmp_path, files=IR_ONLY)
    _packet(run_dir, digest, fingerprint)

    result = _approve(config)

    assert "Signed by " in result.stdout
    assert "as benchmark_owner." in result.stdout
    assert result.returncode == 1  # publication is not connected yet, so generate stops after the signature
    assert result.stderr.startswith("pipeline_incomplete:")
    assert "Traceback" not in result.stderr


def test_approve_script_waits_for_other_roles_and_can_sign_only(tmp_path: Path) -> None:
    config, run_dir, digest, fingerprint = _prepared(tmp_path, profile="regulated")
    _packet(run_dir, digest, fingerprint)

    first = _approve(config, "--approved-by", "Lan", "--role", "benchmark_owner")
    second = _approve(config, "--approved-by", "Minh", "--sign-only")
    wrong_role = _approve(config, "--role", "owner")

    assert first.returncode == 3, first.stderr
    assert "Still waiting for: compliance_owner" in first.stdout
    assert second.returncode == 0, second.stderr
    assert "Run stage=generate to publish" in second.stdout
    assert wrong_role.returncode == 2
    assert "invalid choice" in wrong_role.stderr


# Privacy


def test_every_source_quarantined_blocks_the_run(tmp_path: Path) -> None:
    files = {"secret.md": "secret", "sub/card.md": "card", ".DS_Store": "finder"}
    config, run_dir, _, _ = _prepared(tmp_path, files=files)
    _quarantine(tmp_path, run_dir, {"secret.md": "quarantined", "sub/card.md": "quarantined"})

    caught = _expect(config, "generate", RunStatus.PRIVACY_BLOCKED)
    assert caught.cli_exit_code == 4
    assert _record(run_dir)["last_outcome"]["status"] == "privacy_blocked"


def test_one_safe_source_prevents_the_privacy_block(tmp_path: Path) -> None:
    config, run_dir, _, _ = _prepared(tmp_path, files={"secret.md": "secret", "sub/clean.md": "public"})
    _quarantine(tmp_path, run_dir, {"secret.md": "quarantined", "sub/clean.md": "safe"})

    _expect(config, "generate", RunStatus.PIPELINE_INCOMPLETE)


def test_incomplete_quarantine_report_is_not_treated_as_safe(tmp_path: Path) -> None:
    config, run_dir, _, _ = _prepared(tmp_path, files={"secret.md": "secret", "other.md": "other"})
    _quarantine(tmp_path, run_dir, {"secret.md": "quarantined"})

    caught = _expect(config, "generate", RunStatus.INVALID_RUN_ARTIFACT)
    assert "Missing: ['other.md']" in str(caught)


def test_quarantine_report_for_older_file_contents_is_stale(tmp_path: Path) -> None:
    config, run_dir, _, _ = _prepared(tmp_path, files={"a.md": "secret"})
    old_row = json.dumps({"path": "a.md", "status": "quarantined", "sha256": _sha256(tmp_path / "policies" / "a.md")})
    (tmp_path / "policies" / "a.md").write_text("now safe", encoding="utf-8")
    _run(config, "prepare")
    (run_dir / "quarantine" / "files.jsonl").write_text(old_row + "\n", encoding="utf-8")

    caught = _expect(config, "generate", RunStatus.INVALID_RUN_ARTIFACT)
    assert "Changed: ['a.md']" in str(caught)


@pytest.mark.parametrize("path", ["./a.md", "sub\\a.md", "/a.md", "sub/../a.md", ""])
def test_non_canonical_quarantine_paths_are_refused(tmp_path: Path, path: str) -> None:
    config, run_dir, _, _ = _prepared(tmp_path, files={"a.md": "secret"})
    row = {"path": path, "status": "quarantined", "sha256": _sha256(tmp_path / "policies" / "a.md")}
    (run_dir / "quarantine" / "files.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    caught = _expect(config, "generate", RunStatus.INVALID_RUN_ARTIFACT)
    assert "relative POSIX path" in str(caught)


def test_run_blocked_row_needs_the_current_inputs(tmp_path: Path) -> None:
    config, run_dir, digest, fingerprint = _prepared(tmp_path)
    report = run_dir / "quarantine" / "files.jsonl"
    file_row = {"path": "policy.md", "status": "safe", "sha256": _sha256(tmp_path / "policies" / "policy.md")}

    stale = {"run_blocked": True, "config_sha256": digest, "source_fingerprint": "old"}
    report.write_text(json.dumps(stale) + "\n" + json.dumps(file_row) + "\n", encoding="utf-8")
    _expect(config, "generate", RunStatus.PIPELINE_INCOMPLETE)

    current = {"run_blocked": True, "config_sha256": digest, "source_fingerprint": fingerprint}
    report.write_text(json.dumps(current) + "\n", encoding="utf-8")
    _expect(config, "generate", RunStatus.PRIVACY_BLOCKED)


def test_damaged_quarantine_report_is_not_called_a_config_error(tmp_path: Path) -> None:
    config, run_dir, _, _ = _prepared(tmp_path)
    (run_dir / "quarantine" / "files.jsonl").write_text("{bad\n", encoding="utf-8")

    caught = _expect(config, "generate", RunStatus.INVALID_RUN_ARTIFACT)
    assert "config itself is valid" in str(caught)


# Config validation


def test_config_status_may_be_omitted(tmp_path: Path) -> None:
    run_dir = _run(_config(tmp_path, overrides={"config_status": None}), "prepare")
    assert (
        yaml.safe_load((run_dir / "config_resolved.yaml").read_text(encoding="utf-8"))["config_status"] == "resolved"
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"config_status": "template"}, "not runnable"),
        ({"config_status": "draft"}, "or omitted"),
        ({"config_status": []}, "or omitted"),
        ({"stage": []}, "stage must be one of"),
        ({"mode": ["assisted"]}, "mode must be one of"),
        ({"assurance": {"profile": []}}, "assurance.profile must be one of"),
        ({"generation": {"modules": {"adversarial": []}}}, "generation.modules.adversarial must be true"),
        ({"generation": {"modules": {"adversarial": 1}}}, "generation.modules.adversarial must be true"),
        ({"expt_name": ".."}, "expt_name"),
        ({"budget": {"assisted_max_case": 5}}, "Unknown budget keys: assisted_max_case"),
        ({"budget": {"assisted_max_cases": 1.5}}, "budget.assisted_max_cases must be a whole number"),
        ({"generation": {"turn_mixx": {}}}, "Unknown generation keys: turn_mixx"),
        ({"generation": {"turn_mix": {"multi_turn": 0.5}}}, "shares must add up to 1"),
        ({"proposals": {"jurisdicton": "VN"}}, "Unknown proposals keys: jurisdicton"),
        ({"proposals": {"target_languages": "en"}}, "proposals.target_languages must be a list"),
        ({"proposals": {"decision_actions": ["allow", "ban"]}}, "unsupported entries ban"),
        ({"assurance": {"review_sample_rate": "all"}}, "review_sample_rate must be a share"),
        ({"assurance": {"second_judge_on": ["never"]}}, "unsupported entries never"),
        (
            {"generation_model_config": {"inference_parameters": {"temperature": "hot"}}},
            "temperature must be a number",
        ),
        ({"generation_model_config": {"api_key": "sk-test"}}, "generation_model_config.api_key looks like a secret"),
        ({"judge_model_configs": [{"provider_api_key": "x"}]}, "judge_model_configs[0].provider_api_key looks like"),
        ({"evaluation": {"endpoint": {"Authorization": "Bearer x"}}}, "evaluation.endpoint.Authorization looks like"),
        ({"proposals": {"domain": "REPLACE_ME_DOMAIN"}}, "REPLACE_ME placeholder"),
        ({"proposals": {"moderation_targets": ["user_input", "tool_log"]}}, "unsupported entries tool_log"),
        (
            {"proposals": {"decision_actions": ["allow", "block"]}},
            "decision_mix gives a share to caution, escalate",
        ),
        (
            {
                "generation": {
                    "decision_mix": {"allow": 0.4, "block": 0.4, "caution": 0, "escalate": 0, "rewrite": 0.2}
                }
            },
            "decision_mix gives a share to rewrite",
        ),
        ({"generation": {"decision_mix": {"ban": 0.1}}}, "Unknown generation.decision_mix keys: ban"),
    ],
)
def test_invalid_config_is_reported_without_writing(tmp_path: Path, overrides: dict[str, Any], message: str) -> None:
    caught = _expect(_config(tmp_path, overrides=overrides), "prepare", RunStatus.INVALID_CONFIG)

    assert message in str(caught)
    assert not (tmp_path / "out").exists()


def test_config_accepts_every_target_and_action_policy_rule_ir_accepts(tmp_path: Path) -> None:
    overrides = {
        "proposals": {
            "moderation_targets": ["user_input", "retrieved_context", "tool_input", "tool_output", "tool_call"],
            "decision_actions": ["allow", "block", "require_confirmation", "rewrite"],
        },
        "generation": {
            "decision_mix": {
                "allow": 0.4,
                "block": 0.3,
                "caution": 0,
                "escalate": 0,
                "require_confirmation": 0.2,
                "rewrite": 0.1,
            }
        },
    }
    run_dir = _run(_config(tmp_path, overrides=overrides), "prepare")

    resolved = yaml.safe_load((run_dir / "config_resolved.yaml").read_text(encoding="utf-8"))
    assert resolved["proposals"]["moderation_targets"][-1] == "tool_call"
    assert resolved["generation"]["decision_mix"]["require_confirmation"] == 0.2


def test_secret_never_reaches_the_run_directory(tmp_path: Path) -> None:
    _expect(
        _config(tmp_path, overrides={"generation_model_config": {"api_key": "sk-test"}}),
        "prepare",
        RunStatus.INVALID_CONFIG,
    )
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("family: safety\nfamily: safety\n", "duplicate key"),
        ("family: safety\n1: x\n", "non-string key"),
        ("family: safety\nproposals:\n  domain: 2026-01-01\n", "unsupported YAML type date"),
    ],
)
def test_ambiguous_yaml_is_refused(tmp_path: Path, text: str, message: str) -> None:
    config = tmp_path / "customer.yaml"
    config.write_text(text, encoding="utf-8")

    caught = _expect(config, "prepare", RunStatus.INVALID_CONFIG)
    assert message in str(caught)


@pytest.mark.parametrize("output", ["policies", "policies/runs"])
def test_output_inside_input_is_rejected(tmp_path: Path, output: str) -> None:
    caught = _expect(
        _config(tmp_path, overrides={"output_dir": str(tmp_path / output)}), "prepare", RunStatus.INVALID_CONFIG
    )
    assert "must not be the same directory or inside each other" in str(caught)
    assert not (tmp_path / "policies" / "customer-safety-v1").exists()


def test_second_writer_is_refused_while_the_run_is_locked(tmp_path: Path) -> None:
    config = _config(tmp_path)
    with _run_lock(tmp_path / "out" / "customer-safety-v1"):
        _expect(config, "prepare", RunStatus.RUN_LOCKED)


# Real process exit codes


@pytest.mark.parametrize(
    ("stage", "exit_code", "prefix"),
    [
        ("prepare", 0, ""),
        ("generate", 3, "review_required:"),
        ("eval", 1, "stage_unavailable:"),
        ("translate", 1, "stage_unavailable:"),
    ],
)
def test_cli_process_exit_codes(tmp_path: Path, stage: str, exit_code: int, prefix: str) -> None:
    config, run_dir, digest, fingerprint = _prepared(tmp_path, files=IR_ONLY)
    _packet(run_dir, digest, fingerprint)

    result = _cli(config, stage)

    assert result.returncode == exit_code, result.stderr
    assert "Traceback" not in result.stderr
    if prefix:
        assert result.stderr.startswith(prefix)


def test_cli_reports_malformed_yaml_without_a_traceback(tmp_path: Path) -> None:
    config = tmp_path / "broken.yaml"
    config.write_text("family: [\n", encoding="utf-8")

    result = _cli(config, "prepare")

    assert result.returncode == 1
    assert result.stderr.startswith("byob_config_unreadable:")
    assert "Traceback" not in result.stderr


# Helpers


def _run(config: Path, stage: str) -> Path:
    return run_byob(config=config, stage=stage, family="safety")


def _expect(config: Path, stage: str, status: RunStatus) -> SafetyStatus:
    with pytest.raises(SafetyStatus) as caught:
        _run(config, stage)
    assert caught.value.status is status, str(caught.value)
    return caught.value


def _cli(config: Path, stage: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "nemotron.steps.byob.scripts.run",
            "--config",
            str(config),
            "--family",
            "safety",
            "--stage",
            stage,
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
        check=False,
        timeout=120,
    )


def _prepared(
    tmp_path: Path, *, files: dict[str, str] | None = None, profile: str | None = None
) -> tuple[Path, Path, str, str]:
    overrides = {"assurance": {"profile": profile}} if profile else None
    config = _config(tmp_path, files=files, overrides=overrides)
    run_dir = _run(config, "prepare")
    record = _record(run_dir)
    return config, run_dir, record["config_sha256"], record["source_fingerprint"]


def _record(run_dir: Path) -> dict:
    return json.loads((run_dir / "run_record.json").read_text(encoding="utf-8"))


def _cache(run_dir: Path) -> dict:
    return json.loads((run_dir / "stage_cache" / "cache_index.json").read_text(encoding="utf-8"))


def _add_cache_entry(run_dir: Path) -> None:
    index = _cache(run_dir)
    index["stages"]["candidate_extraction"] = {"status": "completed"}
    (run_dir / "stage_cache" / "cache_index.json").write_text(json.dumps(index), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _quarantine(tmp_path: Path, run_dir: Path, statuses: dict[str, str]) -> None:
    rows = [
        json.dumps({"path": path, "status": status, "sha256": _sha256(tmp_path / "policies" / path)})
        for path, status in statuses.items()
    ]
    (run_dir / "quarantine" / "files.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _packet(run_dir: Path, digest: str, fingerprint: str, **extra: Any) -> Path:
    path = run_dir / "publication_review_packet.yaml"
    _write_yaml(path, {"packet_version": 1, "config_sha256": digest, "source_fingerprint": fingerprint, **extra})
    return path


def _signoff(
    run_dir: Path, digest: str, fingerprint: str, packet_sha256: str, *, signatures: list | None = None
) -> None:
    signoff = {
        "signoff_version": 1,
        "config_sha256": digest,
        "source_fingerprint": fingerprint,
        "packet_sha256": packet_sha256,
        "signatures": [{"approved_by": "Lan", "role": "benchmark_owner"}] if signatures is None else signatures,
    }
    (run_dir / "publication_signoff.json").write_text(json.dumps(signoff), encoding="utf-8")


def _write_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def _config(
    tmp_path: Path,
    *,
    files: dict[str, str] | None = None,
    overrides: dict[str, Any] | None = None,
) -> Path:
    policies = tmp_path / "policies"
    policies.mkdir(exist_ok=True)
    if not any(policies.iterdir()):
        for relative, text in (files or {"policy.md": "Block requests for card numbers."}).items():
            (policies / relative).parent.mkdir(parents=True, exist_ok=True)
            (policies / relative).write_text(text, encoding="utf-8")
    data: dict[str, Any] = {
        "family": "safety",
        "expt_name": "customer-safety-v1",
        "stage": "prepare",
        "config_status": "resolved",
        "mode": "assisted",
        "input_dir": str(policies),
        "output_dir": str(tmp_path / "out"),
    }
    for key, value in (overrides or {}).items():
        if value is None:
            data.pop(key, None)
        else:
            data[key] = value
    config = tmp_path / "customer.yaml"
    config.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return config
