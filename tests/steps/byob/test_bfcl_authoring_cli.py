from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from nemotron.steps.byob.runtime.authoring_workflow.resume import (
    AuthoringResumeError,
    ResumedAuthoringSession,
    ResumeVerdict,
    bind_artifact,
)
from nemotron.steps.byob.runtime.authoring_workflow.workspace_keys import workspace_key_pair
from nemotron.steps.byob.runtime.authoring_workflow.workspace_lock import WorkspaceLock
from nemotron.steps.byob.runtime.pack_authoring.artifacts import write_canonical_json
from nemotron.steps.byob.runtime.pack_authoring.authorization import ExposureSubject
from nemotron.steps.byob.runtime.pack_authoring.questions import (
    AnswerSubmission,
    build_answer_set,
    build_open_questions,
    write_answer_set,
    write_open_questions,
)
from nemotron.steps.byob.runtime.source_adapters.evidence import load_source_evidence
from nemotron.steps.byob.scripts import bfcl_author
from tests.steps.byob.test_bfcl_authoring_questions import _candidate
from tests.steps.byob.test_bfcl_authoring_revisions import _committed_session

DIGEST_A = "sha256:" + "a" * 64
DIGEST_B = "sha256:" + "b" * 64


class _NoopResumed:
    lease = None

    def __enter__(self) -> _NoopResumed:
        return self

    def __exit__(self, *_args: object) -> None:
        return None


def _write_head(workspace: Path, session_digest: str, phase: str) -> None:
    write_canonical_json(
        {
            "schema_version": "bfcl-authoring-head-v1",
            "tenant_id": "tenant-a",
            "run_id": "run-a",
            "phase": phase,
            "session_digest": session_digest,
        },
        workspace / "authoring_head.json",
    )


def test_help_lists_commands_and_separate_approval_boundaries() -> None:
    help_text = bfcl_author._parser().format_help()

    for command in (
        "prepare",
        "author",
        "resume",
        "answer",
        "authorize",
        "approve",
        "draft",
        "review",
        "freeze",
        "publish",
        "purge-cache",
    ):
        assert command in help_text
    assert "Pre-model authorization" in help_text
    assert "final release approval" in help_text
    assert "CI mode never prompts" in help_text


def test_prepare_is_a_four_input_dev_entry_and_stops_for_source_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    tools = tmp_path / "tools.json"
    tools.write_text("[]\n", encoding="utf-8")
    brief = tmp_path / "brief.txt"
    brief.write_text("Evaluate hotel booking.\n", encoding="utf-8")
    observed: dict[str, Any] = {}
    monkeypatch.setattr(
        bfcl_author,
        "_delegate",
        lambda module, arguments: observed.update(module=module, arguments=arguments),
    )
    workspace = tmp_path / "workspace"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bfcl_author.py",
            "prepare",
            "--workspace",
            str(workspace),
            "--tools",
            str(tools),
            "--brief",
            str(brief),
            "--language",
            "en",
        ],
    )

    bfcl_author.main()

    output = json.loads(capsys.readouterr().out)
    profile = json.loads((workspace / bfcl_author.AUTHORING_PROFILE_FILE).read_text(encoding="utf-8"))
    assert output["status"] == "source_review_required"
    assert output["trust_mode"] == "dev"
    assert profile["release_status"] == "development_unsealed"
    assert profile["official_publishable"] is False
    assert profile["language"] == "en"
    assert observed["module"] == "nemotron.steps.byob.scripts.scaffold_source_package"
    assert "--dependency-lock" in observed["arguments"]


@pytest.mark.parametrize(
    ("marker", "expected_adapter", "expected_module"),
    [
        (
            "backend.py",
            "local_python",
            "nemotron.steps.byob.scripts.build_source_intake",
        ),
        (
            "endpoint_config.yaml",
            "http_package",
            "nemotron.steps.byob.scripts.build_source_intake",
        ),
    ],
)
def test_author_detects_conventional_adapter_and_delegates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    marker: str,
    expected_adapter: str,
    expected_module: str,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / marker).write_text("# reviewed\n", encoding="utf-8")
    brief = tmp_path / "brief.txt"
    brief.write_text("Evaluate inventory lookup.", encoding="utf-8")
    observed: dict[str, Any] = {}

    def delegate(module: str, arguments: list[str]) -> None:
        observed.update(module=module, arguments=arguments)

    monkeypatch.setattr(bfcl_author, "_delegate", delegate)
    monkeypatch.setattr(bfcl_author, "_commit_intake_session", lambda **_kwargs: None)
    monkeypatch.setattr(bfcl_author, "_advance_developer_consent", lambda *_args, **_kwargs: None)
    monkeypatch.setenv(
        (
            "BFCL_ENABLE_LOCAL_PYTHON"
            if expected_adapter == "local_python"
            else "BFCL_ENABLE_HTTP_PACKAGE"
        ),
        "1",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bfcl_author.py",
            "--ci",
            "author",
            "--workspace",
            str(tmp_path / "workspace"),
            "--source",
            source.as_uri(),
            "--brief",
            str(brief),
            "--pack-id",
            "inventory",
            "--pack-version",
            "1.0.0",
            "--held-out-not-applicable-reason",
            "The catalog is public.",
            "--held-out-reviewed-by",
            "reviewer@example.test",
        ],
    )
    bfcl_author.main()

    assert observed["module"] == expected_module
    arguments = observed["arguments"]
    assert arguments[arguments.index("--adapter") + 1] == expected_adapter
    assert arguments[arguments.index("--pack-id") + 1] == "inventory"
    assert (
        arguments[arguments.index("--held-out-not-applicable-reason") + 1]
        == "The catalog is public."
    )
    assert (
        arguments[arguments.index("--held-out-reviewed-by") + 1]
        == "reviewer@example.test"
    )
    assert "--resolved-authoring-config" in arguments
    assert (tmp_path / "workspace" / "resolved_authoring_config.json").is_file()
    assert not (tmp_path / "workspace" / ".locks" / "default" / "authoring.lock").read_text(
        encoding="utf-8"
    )


def test_prepared_dev_authoring_hides_certification_and_held_out_plumbing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    source = workspace / "source"
    source.mkdir(parents=True)
    (source / "backend.py").write_text("# reviewed\n", encoding="utf-8")
    brief = tmp_path / "brief.txt"
    brief.write_text("Evaluate inventory lookup.\n", encoding="utf-8")
    write_canonical_json(
        {
            "schema_version": "bfcl-authoring-profile-v1",
            "trust_mode": "dev",
            "language": "vi",
            "source": str(source),
            "domain_brief": str(brief),
            "model_exposure_consented": False,
        },
        workspace / bfcl_author.AUTHORING_PROFILE_FILE,
    )
    observed: dict[str, Any] = {}
    monkeypatch.setattr(
        bfcl_author,
        "_delegate",
        lambda module, arguments: observed.update(module=module, arguments=arguments),
    )
    monkeypatch.setattr(bfcl_author, "_commit_intake_session", lambda **_kwargs: None)
    monkeypatch.setattr(
        bfcl_author,
        "_advance_developer_consent",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setenv("BFCL_ENABLE_LOCAL_PYTHON", "1")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bfcl_author.py",
            "--ci",
            "author",
            "--workspace",
            str(workspace),
            "--pack-id",
            "inventory",
            "--pack-version",
            "1.0.0",
            "--allow-model-exposure",
            "--reviewed-by",
            "developer@example.test",
        ],
    )

    bfcl_author.main()

    arguments = observed["arguments"]
    assert "--certification-private-key" in arguments
    assert arguments[arguments.index("--certification-key-id") + 1] == "workspace-development"
    assert arguments[arguments.index("--held-out-reviewed-by") + 1] == "developer@example.test"
    assert "Development benchmark" in arguments[
        arguments.index("--held-out-not-applicable-reason") + 1
    ]
    assert arguments[arguments.index("--domain-brief-language") + 1] == "vi"
    profile = json.loads(
        (workspace / bfcl_author.AUTHORING_PROFILE_FILE).read_text(encoding="utf-8")
    )
    assert profile["model_exposure_consented"] is True


def test_dev_draft_needs_only_model_and_provider_after_consent(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    write_canonical_json(
        {
            "schema_version": "bfcl-authoring-profile-v1",
            "trust_mode": "dev",
            "model_exposure_consented": True,
        },
        workspace / bfcl_author.AUTHORING_PROFILE_FILE,
    )
    args = type("Args", (), {"workspace": workspace, "trust_mode": None})()

    arguments = bfcl_author._developer_draft_arguments(
        args,
        ["--model", "mistral-small-24b", "--model-provider", "openai"],
    )

    assert arguments[arguments.index("--model-alias") + 1] == "author"
    assert arguments[arguments.index("--model-canonical-id") + 1] == "mistral-small-24b"
    assert arguments[arguments.index("--bundle") + 1].endswith(
        "intake/evidence_bundle.json"
    )
    assert arguments[arguments.index("--output") + 1].endswith("workspace/drafting")
    assert arguments[arguments.index("--approval") + 1].endswith(
        "workspace/development_evidence_consent.json"
    )


def test_dev_publish_fills_release_seal_plumbing(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    write_canonical_json(
        {"schema_version": "bfcl-authoring-profile-v1", "trust_mode": "dev"},
        workspace / bfcl_author.AUTHORING_PROFILE_FILE,
    )
    args = type("Args", (), {"workspace": workspace, "trust_mode": None})()

    arguments = bfcl_author._developer_publish_arguments(
        args,
        ["--config", "publication.yaml"],
    )

    assert arguments[arguments.index("--release") + 1].endswith("workspace/release")
    assert arguments[arguments.index("--seal-public-key") + 1].endswith(
        "workspace/.trust/release-seal-public.pem"
    )
    assert arguments[arguments.index("--seal-key-id") + 1] == "workspace-development-release"
    # An old dev release without bound trust tags must not regain Gold eligibility
    # through the new developer defaults. Signature verification remains downstream.
    manifest = workspace / "release/freeze_manifest.json"
    write_canonical_json({"schema_version": bfcl_author.FREEZE_MANIFEST_VERSION_V3}, manifest)
    with pytest.raises(bfcl_author.GuidedCliError, match="publication_trust_mismatch"):
        bfcl_author._developer_publish_arguments(args, ["--config", "publication.yaml"])
    write_canonical_json(bfcl_author.trust_fields("dev"), manifest)
    assert bfcl_author._developer_publish_arguments(args, ["--config", "publication.yaml"]) == arguments


def test_workspace_keys_are_private_from_creation_and_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nemotron.steps.byob.runtime.authoring_workflow import workspace_keys

    original_open = os.open
    creation_modes: list[int] = []

    def observed_open(path, flags, mode=0o777, **kwargs):
        descriptor = original_open(path, flags, mode, **kwargs)
        if flags & os.O_CREAT:
            creation_modes.append(os.fstat(descriptor).st_mode & 0o777)
        return descriptor

    monkeypatch.setattr(workspace_keys.os, "open", observed_open)
    original_umask = os.umask(0)
    try:
        private, public = workspace_key_pair(tmp_path, "source-certification")
    finally:
        os.umask(original_umask)
    contents = private.read_bytes(), public.read_bytes()
    assert creation_modes == [0o600, 0o600]
    assert workspace_key_pair(tmp_path, "source-certification") == (private, public)
    assert (private.read_bytes(), public.read_bytes()) == contents
    seal_private, _ = workspace_key_pair(tmp_path, "release-seal")
    assert seal_private.read_bytes() != contents[0]


@pytest.mark.parametrize("damage", ["missing", "mismatch", "permissions", "symlink", "directory_symlink"])
def test_workspace_key_damage_never_rotates_existing_private_key(tmp_path: Path, damage: str) -> None:
    private, public = workspace_key_pair(tmp_path / "workspace", "release-seal")
    original = private.read_bytes()
    if damage == "missing":
        public.unlink()
    elif damage == "mismatch":
        _, other_public = workspace_key_pair(tmp_path / "other", "release-seal")
        public.write_bytes(other_public.read_bytes())
    elif damage == "permissions":
        private.chmod(0o644)
    elif damage == "symlink":
        target = tmp_path / "original-private.pem"
        private.rename(target)
        private.symlink_to(target)
    else:
        target = tmp_path / "original-trust"
        private.parent.rename(target)
        private.parent.symlink_to(target, target_is_directory=True)
    with pytest.raises((OSError, ValueError)):
        workspace_key_pair(tmp_path / "workspace", "release-seal")
    assert private.read_bytes() == original


def test_missing_developer_approval_arguments_never_acquire_a_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_canonical_json({"trust_mode": "dev"}, tmp_path / bfcl_author.AUTHORING_PROFILE_FILE)
    calls: list[str] = []
    monkeypatch.setattr(bfcl_author, "_current_session", lambda _args, command: calls.append(command))
    with pytest.raises(SystemExit):
        _run_cli(monkeypatch, [
            "approve", "--workspace", str(tmp_path), "--accept-reviewed-release",
        ])
    assert calls == []
    assert not (tmp_path / ".trust").exists()


@pytest.fixture
def interrupted_dev_approval(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Exercise the real CLI and leases with deterministic release delegates."""
    write_canonical_json({"trust_mode": "dev"}, tmp_path / bfcl_author.AUTHORING_PROFILE_FILE)
    packet_path = write_canonical_json({"record_digest": DIGEST_A}, tmp_path / "review_packet.json")
    state = SimpleNamespace(phase="review_ready", bindings=SimpleNamespace(
        review_packet=bind_artifact(tmp_path, packet_path, digest_kind="canonical_json"),
        release_approval=None,
    ))
    lock = WorkspaceLock(tmp_path, tenant_id="tenant-a", run_id="run-a")
    gate = SimpleNamespace(load_state=lambda _digest: state)
    calls: list[str] = []

    def current(_args, command):
        lease = lock.acquire()
        permitted = {"review_ready": "approve_release", "release_approved": "freeze"}
        if permitted.get(state.phase) != command:
            lease.release()
            raise AuthoringResumeError("resume_command_not_permitted", "wrong phase", recovery="freeze")
        verdict = ResumeVerdict(
            session_digest=DIGEST_A, phase=state.phase, command=command, permitted_commands=(command,),
        )
        return gate, ResumedAuthoringSession(verdict, lease)

    def commit(_args, _gate, _resumed, *, phase, updates):
        state.phase = phase
        for name, value in updates.items():
            setattr(state.bindings, name, value)

    def delegate(module, arguments):
        calls.append(module)
        output = bfcl_author._required_path_argument(arguments, "--output")
        if module.endswith("approve_authoring_review"):
            assert output == tmp_path / "release_approval.json"
            assert all(
                f"--accept-{name.replace('_', '-')}" in arguments
                for name in bfcl_author.REQUIRED_CHECKLIST_V2
            )
            write_canonical_json({
                "approved_by": "reviewer@example.test", "reviewed_at": "2026-09-09T12:00:00Z",
                "approval_digest": DIGEST_B, "review_packet_digest": DIGEST_A,
            }, output)
        elif calls.count(module) == 1:
            raise OSError("injected freeze failure")
        else:
            write_canonical_json({}, output / "freeze_manifest.json")

    monkeypatch.setattr(bfcl_author, "_current_session", current)
    monkeypatch.setattr(bfcl_author, "_commit_transition", commit)
    monkeypatch.setattr(bfcl_author, "_delegate", delegate)
    command = [
        "approve", "--workspace", str(tmp_path), "--approved-by", " reviewer@example.test ",
        "--reviewed-at", "2026-09-09T12:00:00Z", "--accept-reviewed-release",
        "--output", str(tmp_path / "unused.json"), f"--output={tmp_path / 'release_approval.json'}",
    ]
    with pytest.raises(SystemExit):
        _run_cli(monkeypatch, command)
    assert state.phase == "release_approved"
    # Acquiring a new real lease proves failure released the previous one.
    lock.acquire().release()
    return state, lock, command, calls


def test_approve_retries_failed_freeze_without_reapproving(
    interrupted_dev_approval, monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, lock, command, calls = interrupted_dev_approval
    _run_cli(monkeypatch, command)
    assert state.phase == "frozen"
    assert sum(module.endswith("approve_authoring_review") for module in calls) == 1
    assert sum(module.endswith("freeze_authoring_pack") for module in calls) == 2
    lock.acquire().release()


@pytest.mark.parametrize("release_status", ["exact", "different_approval", "invalid_seal", "unsigned_legacy"])
def test_approve_recovers_a_completed_freeze_only_after_seal_and_approval_verification(
    interrupted_dev_approval, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, release_status: str,
) -> None:
    state, lock, command, calls = interrupted_dev_approval
    write_canonical_json({}, tmp_path / "release" / "freeze_manifest.json")
    verified: list[Path] = []

    def load_release(path, *, trusted_seal_keys, expected_seal_issuer):
        assert "workspace-development-release" in trusted_seal_keys
        assert expected_seal_issuer == "workspace-development"
        verified.append(path)
        if release_status == "invalid_seal":
            raise ValueError("invalid signed release")
        return SimpleNamespace(manifest={
            "schema_version": (
                "legacy" if release_status == "unsigned_legacy" else bfcl_author.FREEZE_MANIFEST_VERSION_V4
            ),
            "review_approval_digest": DIGEST_A if release_status == "different_approval" else DIGEST_B,
            "review_packet_digest": DIGEST_A,
        })

    monkeypatch.setattr(bfcl_author, "load_frozen_release", load_release)
    if release_status == "exact":
        _run_cli(monkeypatch, command)
        assert state.phase == "frozen"
    else:
        with pytest.raises(SystemExit):
            _run_cli(monkeypatch, command)
        assert state.phase == "release_approved"
    assert verified == [tmp_path / "release"]
    assert len(calls) == 2
    lock.acquire().release()


@pytest.mark.parametrize("changed", ["packet", "approval", "reviewer"])
def test_freeze_retry_rejects_changed_approval_and_releases_lock(
    interrupted_dev_approval, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed: str,
) -> None:
    state, lock, command, calls = interrupted_dev_approval
    if changed == "reviewer":
        command[command.index("--approved-by") + 1] = "someone-else@example.test"
    else:
        path = tmp_path / ("review_packet.json" if changed == "packet" else "release_approval.json")
        write_canonical_json({"changed": True}, path)
    with pytest.raises(SystemExit):
        _run_cli(monkeypatch, command)
    assert state.phase == "release_approved"
    assert len(calls) == 2
    lock.acquire().release()


def test_developer_review_cannot_override_profile_trust(tmp_path: Path) -> None:
    write_canonical_json({"trust_mode": "dev"}, tmp_path / bfcl_author.AUTHORING_PROFILE_FILE)
    args = SimpleNamespace(workspace=tmp_path)
    with pytest.raises(bfcl_author.GuidedCliError, match="authoring_trust_override_forbidden"):
        bfcl_author._developer_review_arguments(args, ["--trust-mode=compliance"])
    assert bfcl_author._argument_value(bfcl_author._developer_review_arguments(args, []), "--trust-mode") == "dev"


def test_compliance_approval_keeps_explicit_boundary_requirement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exited:
        _run_cli(
            monkeypatch,
            [
                "approve",
                "--workspace",
                str(tmp_path / "workspace"),
                "--approved-by",
                "reviewer@example.test",
            ],
        )

    assert exited.value.code == 1
    assert json.loads(capsys.readouterr().err)["code"] == "approval_boundary_required"


def test_workspace_preflight_reports_shared_filesystem_failure_before_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unsupported(*_args: object, **_kwargs: object) -> Path:
        raise OSError("operation not supported")

    monkeypatch.setattr(bfcl_author, "write_text_atomic", unsupported)
    with pytest.raises(
        bfcl_author.GuidedCliError,
        match="workspace_atomic_writes_unsupported",
    ) as refused:
        bfcl_author._preflight_workspace(tmp_path, run_id="test")
    assert "local writable workspace" in refused.value.recovery


def test_author_detects_mcp_and_delegates_existing_intake(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "mcp_intake.yaml"
    source.write_text("intake_version: test\n", encoding="utf-8")
    brief = tmp_path / "brief.txt"
    brief.write_text("Evaluate MCP tools.", encoding="utf-8")
    observed: dict[str, Any] = {}

    monkeypatch.setattr(
        bfcl_author,
        "_delegate",
        lambda module, arguments: observed.update(
            module=module,
            arguments=arguments,
        ),
    )
    monkeypatch.setattr(bfcl_author, "_commit_intake_session", lambda **_kwargs: None)
    monkeypatch.setenv("BFCL_ENABLE_MCP_MODE_A", "1")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bfcl_author.py",
            "author",
            "--workspace",
            str(tmp_path / "workspace"),
            "--source",
            str(source),
            "--brief",
            str(brief),
            "--pack-id",
            "mcp-tools",
            "--pack-version",
            "1.0.0",
            "--held-out-not-applicable-reason",
            "No reserved content in this catalog.",
            "--held-out-reviewed-by",
            "reviewer@example.test",
        ],
    )
    bfcl_author.main()

    assert observed["module"] == "nemotron.steps.byob.scripts.build_mcp_intake"
    assert "--intake" in observed["arguments"]
    assert "--domain-brief" in observed["arguments"]
    assert "--held-out-not-applicable-reason" in observed["arguments"]
    assert "--held-out-reviewed-by" in observed["arguments"]


def _author_argv(tmp_path: Path, source: Path, brief: Path) -> list[str]:
    return [
        "--ci",
        "author",
        "--workspace",
        str(tmp_path / "workspace"),
        "--source",
        str(source),
        "--brief",
        str(brief),
        "--pack-id",
        "inventory",
        "--pack-version",
        "1.0.0",
    ]


def test_author_refuses_to_start_without_a_held_out_decision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Held-out status is settled before intake, so it cannot be an unnamed passthrough."""
    source = tmp_path / "source"
    source.mkdir()
    (source / "backend.py").write_text("# reviewed\n", encoding="utf-8")
    brief = tmp_path / "brief.txt"
    brief.write_text("Evaluate tools.", encoding="utf-8")
    monkeypatch.setattr(
        bfcl_author,
        "_delegate",
        lambda *_args: pytest.fail("intake ran without a held-out decision"),
    )

    with pytest.raises(SystemExit) as exited:
        _run_cli(monkeypatch, _author_argv(tmp_path, source, brief))
    assert exited.value.code == 2

    with pytest.raises(SystemExit) as reviewer_missing:
        _run_cli(
            monkeypatch,
            [
                *_author_argv(tmp_path, source, brief),
                "--held-out-not-applicable-reason",
                "The catalog is public.",
            ],
        )
    assert reviewer_missing.value.code == 2


def test_author_refuses_held_out_content_without_its_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "backend.py").write_text("# reviewed\n", encoding="utf-8")
    brief = tmp_path / "brief.txt"
    brief.write_text("Evaluate tools.", encoding="utf-8")
    content = tmp_path / "reserved.yaml"
    content.write_text("terms: [secret]\n", encoding="utf-8")
    monkeypatch.setattr(
        bfcl_author,
        "_delegate",
        lambda *_args: pytest.fail("intake ran with unbound held-out content"),
    )

    with pytest.raises(SystemExit) as exited:
        _run_cli(
            monkeypatch,
            [
                *_author_argv(tmp_path, source, brief),
                "--held-out-not-applicable-reason",
                "The catalog is public.",
                "--held-out-reviewed-by",
                "reviewer@example.test",
                "--held-out-content",
                str(content),
            ],
        )

    assert exited.value.code == 1
    assert json.loads(capsys.readouterr().err)["code"] == "held_out_content_unbound"


def test_evidence_approval_is_distinct_and_digest_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    workspace, _, session_digest, paths = _committed_session(
        tmp_path,
        phase="intake_complete",
    )
    _write_head(workspace, session_digest, "intake_complete")
    evidence_digest = load_source_evidence(paths["evidence"]).bundle_digest
    subject = write_canonical_json(
        ExposureSubject(
            evidence_digest=evidence_digest,
            domain_brief_content_digest=DIGEST_A,
            domain_brief_source_digest=DIGEST_A,
            domain_brief_redaction_report_digest=DIGEST_A,
            held_out_decision_digest=DIGEST_A,
            held_out_policy_digest=None,
            held_out_redaction_report_digest=DIGEST_A,
        ).model_dump(mode="json"),
        workspace / "subject.json",
    )
    output = workspace / "evidence_approval.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bfcl_author.py",
            "authorize",
            "--workspace",
            str(workspace),
            "--tenant-id",
            "tenant-a",
            "--run-id",
            "run-a",
            "--subject",
            str(subject),
            "--authorized-by",
            "reviewer@example.test",
        ],
    )
    bfcl_author.main()
    capsys.readouterr()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bfcl_author.py",
            "approve",
            "--workspace",
            str(workspace),
            "--tenant-id",
            "tenant-a",
            "--run-id",
            "run-a",
            "--boundary",
            "evidence",
            "--approved-by",
            "reviewer@example.test",
            "--source-bundle-digest",
            DIGEST_A,
            "--normalized-bundle-digest",
            evidence_digest,
            "--output",
            str(output),
        ],
    )
    bfcl_author.main()

    approval = json.loads(output.read_text(encoding="utf-8"))
    result = json.loads(capsys.readouterr().out)
    assert approval["source_bundle_digest"] == DIGEST_A
    assert approval["normalized_bundle_digest"] == evidence_digest
    assert result["status"] == "evidence_approved"
    assert "cannot substitute for final release approval" in result["note"]


def _run_cli(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["bfcl_author.py", *argv])
    bfcl_author.main()


def test_answer_commits_a_revision_that_must_be_reauthorized_and_reapproved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """One correction must travel the guided CLI as a new content-addressed revision."""
    workspace, gate, parent_digest, paths = _committed_session(
        tmp_path,
        phase="intake_complete",
        with_approval=False,
    )
    _write_head(workspace, parent_digest, "intake_complete")
    parent_evidence = load_source_evidence(paths["evidence"])
    questions = build_open_questions(
        evidence_digest=parent_evidence.bundle_digest,
        candidates=(_candidate(),),
    )
    answers = build_answer_set(
        evidence_digest=parent_evidence.bundle_digest,
        question_artifact_digest=questions.artifact_digest,
        answers=(
            AnswerSubmission(
                question_id=questions.questions[0].question_id,
                value=5,
            ),
        ),
    )
    questions_path = write_open_questions(questions, workspace / "open_questions.json")
    answers_path = write_answer_set(answers, workspace / "answer_set.json")
    workspace_argv = [
        "--workspace",
        str(workspace),
        "--tenant-id",
        "tenant-a",
        "--run-id",
        "run-a",
    ]

    _run_cli(
        monkeypatch,
        [
            "answer",
            *workspace_argv,
            "--evidence",
            str(paths["evidence"]),
            "--questions",
            str(questions_path),
            "--answers",
            str(answers_path),
        ],
    )
    revised = json.loads(capsys.readouterr().out)

    assert revised["status"] == "evidence_revised"
    revised_digest = revised["evidence_digest"]
    assert revised_digest != parent_evidence.bundle_digest
    revision_root = Path(revised["revision"])
    assert revision_root.parent == (workspace / "revisions").resolve()
    assert revision_root.name == revised_digest.removeprefix("sha256:")
    assert (revision_root / "revision_record.json").is_file()

    head = json.loads((workspace / "authoring_head.json").read_text(encoding="utf-8"))
    assert head["phase"] == "evidence_revised"
    revised_session = head["session_digest"]
    assert revised_session != parent_digest

    # The correction withdraws the right to draft until the boundaries are redone.
    with pytest.raises(AuthoringResumeError) as blocked:
        gate.open(revised_session, command="draft")
    assert blocked.value.code == "resume_command_not_permitted"

    subject = write_canonical_json(
        ExposureSubject(
            evidence_digest=revised_digest,
            domain_brief_content_digest=DIGEST_A,
            domain_brief_source_digest=DIGEST_A,
            domain_brief_redaction_report_digest=DIGEST_A,
            held_out_decision_digest=DIGEST_A,
            held_out_policy_digest=None,
            held_out_redaction_report_digest=DIGEST_A,
        ).model_dump(mode="json"),
        workspace / "revised_subject.json",
    )
    _run_cli(
        monkeypatch,
        [
            "authorize",
            *workspace_argv,
            "--subject",
            str(subject),
            "--authorized-by",
            "reviewer@example.test",
        ],
    )
    capsys.readouterr()
    _run_cli(
        monkeypatch,
        [
            "approve",
            *workspace_argv,
            "--boundary",
            "evidence",
            "--approved-by",
            "reviewer@example.test",
            "--source-bundle-digest",
            DIGEST_A,
            "--normalized-bundle-digest",
            revised_digest,
            "--output",
            str(workspace / "revised_approval.json"),
        ],
    )
    capsys.readouterr()

    head = json.loads((workspace / "authoring_head.json").read_text(encoding="utf-8"))
    assert head["phase"] == "evidence_approved"
    resumed = gate.open(head["session_digest"], command="draft")
    try:
        assert resumed.verdict.permitted_commands == ("draft",)
    finally:
        resumed.lease.release()


def test_pre_model_authorization_is_a_separate_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    workspace, _, session_digest, _ = _committed_session(
        tmp_path,
        phase="intake_complete",
    )
    _write_head(workspace, session_digest, "intake_complete")
    subject_path = write_canonical_json(
        ExposureSubject(
            evidence_digest=DIGEST_A,
            domain_brief_content_digest=DIGEST_A,
            domain_brief_source_digest=DIGEST_A,
            domain_brief_redaction_report_digest=DIGEST_A,
            held_out_decision_digest=DIGEST_A,
            held_out_policy_digest=None,
            held_out_redaction_report_digest=DIGEST_A,
        ).model_dump(mode="json"),
        workspace / "subject.json",
    )
    output = workspace / "authorization.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bfcl_author.py",
            "authorize",
            "--workspace",
            str(workspace),
            "--tenant-id",
            "tenant-a",
            "--run-id",
            "run-a",
            "--subject",
            str(subject_path),
            "--authorized-by",
            "reviewer@example.test",
            "--output",
            str(output),
        ],
    )
    bfcl_author.main()

    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "model_exposure_authorized"
    assert result["note"] == "This is not final release approval."
    assert output.is_file()


def test_freeze_delegates_to_adapter_neutral_kernel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, Any] = {}
    output = tmp_path / "workspace" / "release"
    output.mkdir(parents=True)
    write_canonical_json({}, output / "freeze_manifest.json")
    monkeypatch.setattr(
        bfcl_author,
        "_delegate",
        lambda module, arguments: observed.update(
            module=module,
            arguments=arguments,
        ),
    )
    monkeypatch.setattr(
        bfcl_author,
        "_current_session",
        lambda *_args: (object(), _NoopResumed()),
    )
    monkeypatch.setattr(bfcl_author, "_commit_transition", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bfcl_author.py",
            "freeze",
            "--workspace",
            str(tmp_path / "workspace"),
            "--freeze-inputs",
            "freeze_inputs.json",
            "--output",
            str(output),
        ],
    )
    bfcl_author.main()

    assert observed["module"] == "nemotron.steps.byob.scripts.freeze_authoring_pack"
    assert observed["arguments"] == [
        "--freeze-inputs",
        "freeze_inputs.json",
        "--output",
        str(output),
    ]


def test_review_dispatches_local_adapter_to_generalized_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, Any] = {}
    output = tmp_path / "workspace" / "review.json"
    output.parent.mkdir(parents=True)
    write_canonical_json({}, output)
    monkeypatch.setattr(
        bfcl_author,
        "_delegate",
        lambda module, arguments: observed.update(
            module=module,
            arguments=arguments,
        ),
    )
    monkeypatch.setattr(
        bfcl_author,
        "_current_session",
        lambda *_args: (object(), _NoopResumed()),
    )
    monkeypatch.setattr(bfcl_author, "_commit_transition", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bfcl_author.py",
            "review",
            "--workspace",
            str(tmp_path / "workspace"),
            "--adapter-kind",
            "local_python",
            "--pack",
            "candidate",
            "--output",
            str(output),
        ],
    )
    bfcl_author.main()

    assert observed["module"] == "nemotron.steps.byob.scripts.build_authoring_review"
    assert observed["arguments"] == [
        "--adapter-kind",
        "local_python",
        "--pack",
        "candidate",
        "--output",
        str(output),
    ]


def test_ci_mode_never_prompts_for_missing_required_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "builtins.input",
        lambda *_args, **_kwargs: pytest.fail("CI mode attempted to prompt"),
    )
    monkeypatch.setattr(sys, "argv", ["bfcl_author.py", "--ci", "author"])

    with pytest.raises(SystemExit) as exited:
        bfcl_author.main()
    assert exited.value.code == 2


def test_author_fails_before_adapter_when_rollout_is_omitted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    for variable in (
        "BFCL_ENABLE_LOCAL_PYTHON",
        "BFCL_ENABLE_HTTP_PACKAGE",
        "BFCL_ENABLE_MCP_MODE_A",
        "BFCL_ENABLE_EXPERIMENTAL_MCP",
    ):
        monkeypatch.delenv(variable, raising=False)
    source = tmp_path / "source"
    source.mkdir()
    (source / "backend.py").write_text("# reviewed\n", encoding="utf-8")
    brief = tmp_path / "brief.txt"
    brief.write_text("Evaluate tools.", encoding="utf-8")
    monkeypatch.setattr(
        bfcl_author,
        "_delegate",
        lambda *_args: pytest.fail("disabled adapter was invoked"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bfcl_author.py",
            "--ci",
            "author",
            "--workspace",
            str(tmp_path / "workspace"),
            "--source",
            str(source),
            "--brief",
            str(brief),
            "--pack-id",
            "tools",
            "--pack-version",
            "1.0.0",
            "--held-out-not-applicable-reason",
            "No reserved content.",
            "--held-out-reviewed-by",
            "reviewer@example.test",
        ],
    )

    with pytest.raises(SystemExit) as exited:
        bfcl_author.main()

    result = json.loads(capsys.readouterr().err)
    assert exited.value.code == 1
    assert result["code"] == "adapter_rollout_disabled"
