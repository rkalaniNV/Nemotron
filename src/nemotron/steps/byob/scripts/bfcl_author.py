#!/usr/bin/env python3
"""Guided BFCL authoring dispatcher.

Pre-model authorization (``authorize`` and ``approve --boundary evidence``) is
separate from final release approval (``approve --boundary release``). Review and
freeze are adapter-neutral; publication remains adapter-scoped.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import urllib.parse
from pathlib import Path
from typing import Any, Literal, cast

from nemotron.steps.byob.runtime.authoring_workflow.cache_retention import (
    CACHE_PURGE_AUDIT_FILE_NAME,
    infer_authoring_cache_path,
    purge_authoring_cache,
)
from nemotron.steps.byob.runtime.authoring_workflow.events import (
    EVENT_FILE_NAME,
    AdapterIdentityPayload,
    CertificationPayload,
    FileAuthoringEventSink,
    RefusalPayload,
    ReleaseFrozenPayload,
    ValidationVerdictPayload,
    emit_authoring_event,
)
from nemotron.steps.byob.runtime.authoring_workflow.resolved_config import (
    RESOLVED_AUTHORING_CONFIG_FILE,
    AdapterKind,
    load_resolved_authoring_config,
    resolve_authoring_config,
    write_resolved_authoring_config,
)
from nemotron.steps.byob.runtime.authoring_workflow.resume import (
    ApprovalBinding,
    AuthoringCommand,
    AuthoringPhase,
    AuthoringResumeGate,
    ResumedAuthoringSession,
    SessionBindings,
    bind_artifact,
    build_session_state,
)
from nemotron.steps.byob.runtime.authoring_workflow.workspace_lock import (
    WorkspaceLease,
    WorkspaceLock,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.config import BfclConfig
from nemotron.steps.byob.runtime.pack_authoring.artifacts import (
    sha256_json,
    write_canonical_json,
)
from nemotron.steps.byob.runtime.pack_authoring.authorization import (
    ExposureSubject,
    authorize_model_exposure_by_human,
    authorize_model_exposure_by_policy,
    write_exposure_authorization,
)
from nemotron.steps.byob.runtime.pack_authoring.questions import (
    apply_answers,
    load_answer_set,
    load_open_questions,
    write_evidence_revision,
)
from nemotron.steps.byob.runtime.source_adapters.certification import AdapterTier
from nemotron.steps.byob.runtime.source_adapters.evidence import load_source_evidence
from nemotron.steps.byob.runtime.source_adapters.migration import (
    MIGRATION_APPROVAL_VERSION,
    NormalizedEvidenceApproval,
)

_DELEGATES = {
    "draft": "nemotron.steps.byob.scripts.draft_mcp_pack",
    "review": "nemotron.steps.byob.scripts.build_authoring_review",
    "freeze": "nemotron.steps.byob.scripts.freeze_authoring_pack",
    "publish": "nemotron.steps.byob.scripts.publish_authoring_release",
}


class GuidedCliError(ValueError):
    def __init__(self, code: str, detail: str, *, recovery: str) -> None:
        self.code = code
        self.detail = detail
        self.recovery = recovery
        super().__init__(f"{code}: {detail}; recovery: {recovery}")


def _add_workspace(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--tenant-id", default="default")
    parser.add_argument("--run-id", default="authoring")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "CI mode never prompts. Supply all adapter-specific flags after the known "
            "guided flags; they are delegated to the existing runtime command."
        ),
    )
    parser.add_argument("--ci", action="store_true", help="Never request interactive input")
    subparsers = parser.add_subparsers(dest="command", required=True)

    author = subparsers.add_parser(
        "author",
        help="Run source intake from the normal source + brief inputs",
    )
    _add_workspace(author)
    author.add_argument("--source", required=True, metavar="PATH_OR_FILE_URI")
    author.add_argument("--brief", type=Path, required=True)
    author.add_argument(
        "--adapter",
        choices=("auto", "local_python", "http_package", "mcp_mode_a"),
        default="auto",
    )
    author.add_argument("--policy", type=Path)
    author.add_argument("--pack-id")
    author.add_argument("--pack-version")
    author.add_argument("--confirm-pack-id", action="store_true")
    author.add_argument("--confirm-pack-version", action="store_true")
    author.add_argument("--required-tier", choices=("A0", "A1", "A2"))

    resume = subparsers.add_parser("resume", help="Verify a session and permitted next step")
    _add_workspace(resume)
    resume.add_argument("--session-digest")
    resume.add_argument(
        "--next",
        dest="next_command",
        required=True,
        choices=(
            "intake",
            "answer",
            "authorize_exposure",
            "approve_evidence",
            "draft",
            "review",
            "approve_release",
            "freeze",
            "publish",
        ),
    )

    purge_cache = subparsers.add_parser(
        "purge-cache",
        help="Plan or execute reference-aware authoring cache retention",
    )
    _add_workspace(purge_cache)
    purge_cache.add_argument("--cache", type=Path)
    purge_cache.add_argument("--actor", required=True)
    purge_cache.add_argument("--reason-code", required=True)
    purge_cache.add_argument("--execute", action="store_true")
    purge_cache.add_argument("--expected-plan-digest")

    answer = subparsers.add_parser("answer", help="Apply reviewed answers as a new revision")
    _add_workspace(answer)
    answer.add_argument("--evidence", type=Path, required=True)
    answer.add_argument("--questions", type=Path, required=True)
    answer.add_argument("--answers", type=Path, required=True)

    authorize = subparsers.add_parser(
        "authorize",
        help="Grant the distinct pre-model exposure authorization",
    )
    _add_workspace(authorize)
    authorize.add_argument("--subject", type=Path, required=True)
    mode = authorize.add_mutually_exclusive_group(required=True)
    mode.add_argument("--authorized-by")
    mode.add_argument("--organizational-policy-digest")
    authorize.add_argument("--output", type=Path)

    approve = subparsers.add_parser(
        "approve",
        help="Record evidence approval or distinct final release approval",
    )
    _add_workspace(approve)
    approve.add_argument("--boundary", choices=("evidence", "release"), required=True)
    approve.add_argument("--approved-by")
    approve.add_argument("--source-bundle-digest")
    approve.add_argument("--normalized-bundle-digest")
    approve.add_argument("--migration-record-digest")
    approve.add_argument("--acknowledge-warning", action="append", default=[])
    approve.add_argument("--acknowledge-finding", action="append", default=[])
    approve.add_argument("--note")
    approve.add_argument("--output", type=Path)

    for command, help_text in (
        ("draft", "Delegate approved evidence to shared drafting"),
        ("review", "Build an adapter-neutral review packet"),
        ("freeze", "Freeze an approved adapter-neutral release"),
        ("publish", "Publish a freshly validated authoring release"),
    ):
        child = subparsers.add_parser(command, help=help_text)
        _add_workspace(child)
        if command in {"review", "publish"}:
            child.add_argument("--adapter-kind", default="mcp_mode_a")
    return parser


def _print(document: dict[str, Any]) -> None:
    print(json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True))


def _event_sink(workspace: Path) -> FileAuthoringEventSink:
    return FileAuthoringEventSink(workspace.resolve() / ".events" / EVENT_FILE_NAME)


def _emit_cli_refusal(args: argparse.Namespace, code: str) -> None:
    try:
        emit_authoring_event(
            _event_sink(args.workspace),
            "refusal_recorded",
            RefusalPayload(
                primary_classification="command_refused",
                reason_codes=(code,) if code and code[0].islower() else (),
            ),
            tenant_id=args.tenant_id,
            run_id=args.run_id,
            session_digest=None,
        )
    except (OSError, ValueError):
        pass


def _delegate(module_name: str, arguments: list[str]) -> None:
    module = importlib.import_module(module_name)
    previous = sys.argv
    try:
        sys.argv = [module_name, *arguments]
        module.main()
    finally:
        sys.argv = previous


def _required_path_argument(arguments: list[str], name: str) -> Path:
    try:
        index = arguments.index(name)
        raw = arguments[index + 1]
    except (ValueError, IndexError) as exc:
        raise GuidedCliError(
            "guided_output_missing",
            f"delegated command requires {name} for session binding",
            recovery=f"provide {name} inside the guided workspace",
        ) from exc
    return Path(raw).resolve()


def _current_session(
    args: argparse.Namespace,
    command: AuthoringCommand,
) -> tuple[AuthoringResumeGate, ResumedAuthoringSession]:
    try:
        head = json.loads(
            (args.workspace.resolve() / "authoring_head.json").read_text(
                encoding="utf-8"
            )
        )
        session_digest = head["session_digest"]
    except (OSError, KeyError, json.JSONDecodeError, TypeError) as exc:
        raise GuidedCliError(
            "authoring_session_required",
            f"guided {command} requires a verified authoring head: {exc}",
            recovery="start with bfcl_author.py author or use the lower-level artifact script",
        ) from exc
    gate = AuthoringResumeGate(
        args.workspace,
        tenant_id=args.tenant_id,
        run_id=args.run_id,
    )
    return gate, gate.open(session_digest, command=command)


def _commit_transition(
    args: argparse.Namespace,
    gate: AuthoringResumeGate,
    resumed: ResumedAuthoringSession,
    *,
    phase: AuthoringPhase,
    updates: dict[str, Any],
) -> str:
    parent = gate.load_state(resumed.verdict.session_digest)
    bindings = parent.bindings.model_copy(update=updates)
    state = build_session_state(
        tenant_id=args.tenant_id,
        run_id=args.run_id,
        phase=phase,
        bindings=bindings,
        parent_session_digest=parent.session_digest,
    )
    gate.commit_state(state, lease=resumed.lease)
    sink = _event_sink(args.workspace)
    if phase == "review_ready":
        binding = state.bindings.review_packet
        assert binding is not None
        packet = json.loads(
            (args.workspace.resolve() / binding.path).read_text(encoding="utf-8")
        )
        validation = packet["adapter_review"]["validation"]
        fingerprint = str(validation["pack_fingerprint"])
        if not fingerprint.startswith("sha256:"):
            fingerprint = f"sha256:{fingerprint}"
        emit_authoring_event(
            sink,
            "validation_verdict",
            ValidationVerdictPayload(
                stage="review",
                tier=validation["tier"],
                gold_eligible=validation["gold"],
                pack_fingerprint=fingerprint,
                validation_report_digest=packet["source_digests"][
                    "validation_report"
                ],
            ),
            tenant_id=args.tenant_id,
            run_id=args.run_id,
            session_digest=state.session_digest,
        )
    elif phase == "frozen":
        binding = state.bindings.frozen_manifest
        assert binding is not None
        manifest = json.loads(
            (args.workspace.resolve() / binding.path).read_text(encoding="utf-8")
        )
        emit_authoring_event(
            sink,
            "release_frozen",
            ReleaseFrozenPayload(
                adapter_kind=manifest["adapter_kind"],
                manifest_digest=manifest["manifest_digest"],
                frozen_pack_fingerprint=manifest["frozen_pack_fingerprint"],
                review_packet_digest=manifest["review_packet_digest"],
                review_approval_digest=manifest["review_approval_digest"],
            ),
            tenant_id=args.tenant_id,
            run_id=args.run_id,
            session_digest=state.session_digest,
        )
    write_canonical_json(
        {
            "schema_version": "bfcl-authoring-head-v1",
            "tenant_id": args.tenant_id,
            "run_id": args.run_id,
            "phase": phase,
            "session_digest": state.session_digest,
        },
        args.workspace.resolve() / "authoring_head.json",
    )
    return state.session_digest


def _detect_adapter(source: Path) -> str:
    if source.is_dir():
        if (source / "backend.py").is_file():
            return "local_python"
        if (source / "endpoint_config.yaml").is_file():
            return "http_package"
    if source.is_file():
        return "mcp_mode_a"
    raise GuidedCliError(
        "source_adapter_unknown",
        f"cannot identify a built-in adapter for {source}",
        recovery="pass --adapter with a reviewed local path",
    )


def _source_path(value: str) -> Path:
    parsed = urllib.parse.urlsplit(value)
    if not parsed.scheme:
        return Path(value).resolve()
    if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
        raise GuidedCliError(
            "source_uri_scheme_unsupported",
            f"source URI scheme {parsed.scheme!r} has no reviewed resolver",
            recovery="use a local path/file URI or install a reviewed source resolver",
        )
    if parsed.query or parsed.fragment:
        raise GuidedCliError(
            "source_uri_invalid",
            "file source URI cannot contain query or fragment",
            recovery="use a canonical file URI",
        )
    return Path(urllib.parse.unquote(parsed.path)).resolve()


def _run_author(args: argparse.Namespace, remainder: list[str]) -> None:
    source = _source_path(args.source)
    brief = args.brief.resolve()
    adapter = cast(
        AdapterKind,
        _detect_adapter(source) if args.adapter == "auto" else args.adapter,
    )
    workspace = args.workspace.resolve()
    output = workspace / "intake"
    resolved = resolve_authoring_config(
        adapter_kind=adapter,
        source=source,
        domain_brief=brief,
        workspace=workspace,
        tenant_id=args.tenant_id,
        run_id=args.run_id,
        pack_id=args.pack_id,
        pack_version=args.pack_version,
        policy_path=args.policy,
        required_certification_tier=(
            AdapterTier(args.required_tier) if args.required_tier is not None else None
        ),
        confirm_pack_id=args.confirm_pack_id,
        confirm_pack_version=args.confirm_pack_version,
        ci=args.ci,
    )
    rollout_policy = resolved.semantic_payload.rollout_policy
    if rollout_policy is None or not rollout_policy.live_authoring_enabled.value:
        raise GuidedCliError(
            "adapter_rollout_disabled",
            f"live {adapter} authoring is not enabled",
            recovery=(
                "enable the adapter in reviewed authoring policy or set its "
                "documented BFCL_ENABLE_* environment flag"
            ),
        )
    config_path = workspace / RESOLVED_AUTHORING_CONFIG_FILE
    declaration_path = workspace / "source_declaration.json"
    lock = WorkspaceLock(
        workspace / ".locks",
        tenant_id=args.tenant_id,
        run_id=args.run_id,
    )
    with lock.acquire() as lease:
        write_resolved_authoring_config(resolved, config_path)
        write_canonical_json(
            {
                "declaration_version": "bfcl-source-declaration-v1",
                adapter: {"path": str(source)},
            },
            declaration_path,
        )
        if adapter == "mcp_mode_a":
            _delegate(
                "nemotron.steps.byob.scripts.build_mcp_intake",
                [
                    "--intake",
                    str(source),
                    "--domain-brief",
                    str(brief),
                    "--output",
                    str(output),
                    "--resolved-authoring-config",
                    str(config_path),
                    *remainder,
                ],
            )
        else:
            _delegate(
                "nemotron.steps.byob.scripts.build_source_intake",
                [
                    "--source",
                    str(source),
                    "--adapter",
                    adapter,
                    "--domain-brief",
                    str(brief),
                    "--output",
                    str(output),
                    "--pack-id",
                    resolved.semantic_payload.pack_id.value,
                    "--pack-version",
                    resolved.semantic_payload.pack_version.value,
                    "--required-tier",
                    resolved.semantic_payload.required_certification_tier.value,
                    "--resolved-authoring-config",
                    str(config_path),
                    *remainder,
                ],
            )
        _commit_intake_session(
            workspace=workspace,
            tenant_id=args.tenant_id,
            run_id=args.run_id,
            declaration_path=declaration_path,
            config_path=config_path,
            evidence_path=output / "evidence_bundle.json",
            resolved_config_digest=resolved.resolved_authoring_config_digest,
            lease=lease,
        )


def _commit_intake_session(
    *,
    workspace: Path,
    tenant_id: str,
    run_id: str,
    declaration_path: Path,
    config_path: Path,
    evidence_path: Path,
    resolved_config_digest: str,
    lease: WorkspaceLease,
) -> None:
    evidence = load_source_evidence(evidence_path)
    gate = AuthoringResumeGate(
        workspace,
        tenant_id=tenant_id,
        run_id=run_id,
    )
    state = build_session_state(
        tenant_id=tenant_id,
        run_id=run_id,
        phase="intake_complete",
        bindings=SessionBindings(
            source=bind_artifact(
                workspace,
                declaration_path,
                digest_kind="canonical_json",
            ),
            evidence=bind_artifact(
                workspace,
                evidence_path,
                digest_kind="canonical_json",
            ),
            resolved_config=bind_artifact(
                workspace,
                config_path,
                digest_kind="canonical_json",
            ),
            source_identity_digest=sha256_json(
                evidence.identity.model_dump(mode="json")
            ),
            evidence_bundle_digest=evidence.bundle_digest,
        ),
    )
    gate.commit_state(state, lease=lease)
    resolved = load_resolved_authoring_config(config_path)
    adapter_kind = cast(AdapterKind, evidence.source_adapter.kind)
    authorization_context_digest = next(
        (
            artifact.digest
            for artifact in evidence.identity.artifacts
            if artifact.role == "authorization_context"
        ),
        None,
    )
    sink = _event_sink(workspace)
    emit_authoring_event(
        sink,
        "adapter_identity_bound",
        AdapterIdentityPayload(
            adapter_kind=adapter_kind,
            source_identity_digest=state.bindings.source_identity_digest,
            evidence_bundle_digest=evidence.bundle_digest,
            descriptor_digest=evidence.certification.descriptor_digest,
            authorization_context_digest=authorization_context_digest,
        ),
        tenant_id=tenant_id,
        run_id=run_id,
        session_digest=state.session_digest,
    )
    emit_authoring_event(
        sink,
        "certification_verified",
        CertificationPayload(
            adapter_kind=adapter_kind,
            attained_tier=evidence.certification.attained_tier,
            required_tier=cast(
                Literal["A0", "A1", "A2"],
                resolved.semantic_payload.required_certification_tier.value,
            ),
            profile_id=evidence.certification.profile_id,
            report_digest=evidence.certification.report_digest,
        ),
        tenant_id=tenant_id,
        run_id=run_id,
        session_digest=state.session_digest,
    )
    write_canonical_json(
        {
            "schema_version": "bfcl-authoring-head-v1",
            "tenant_id": tenant_id,
            "run_id": run_id,
            "phase": state.phase,
            "session_digest": state.session_digest,
            "resolved_authoring_config_digest": resolved_config_digest,
        },
        workspace / "authoring_head.json",
    )


def _run_resume(args: argparse.Namespace) -> None:
    session_digest = args.session_digest
    if session_digest is None:
        try:
            head = json.loads(
                (args.workspace.resolve() / "authoring_head.json").read_text(
                    encoding="utf-8"
                )
            )
            session_digest = head["session_digest"]
        except (OSError, KeyError, json.JSONDecodeError, TypeError) as exc:
            raise GuidedCliError(
                "authoring_head_invalid",
                f"cannot resolve current session: {exc}",
                recovery="provide --session-digest from a verified session",
            ) from exc
    if not isinstance(session_digest, str):
        raise GuidedCliError(
            "authoring_head_invalid",
            "current session digest is not a string",
            recovery="provide --session-digest from a verified session",
        )
    gate = AuthoringResumeGate(
        args.workspace,
        tenant_id=args.tenant_id,
        run_id=args.run_id,
    )
    with gate.open(
        session_digest,
        command=args.next_command,
    ) as resumed:
        _print(
            {
                "status": "resume_verified",
                "phase": resumed.verdict.phase,
                "session_digest": resumed.verdict.session_digest,
                "next_command": resumed.verdict.command,
                "permitted_commands": list(resumed.verdict.permitted_commands),
            }
        )


def _run_answer(
    args: argparse.Namespace,
    *,
    lease: WorkspaceLease | None = None,
) -> tuple[Any, Path]:
    evidence = load_source_evidence(args.evidence)
    questions = load_open_questions(
        args.questions,
        evidence_digest=evidence.bundle_digest,
    )
    answers = load_answer_set(
        args.answers,
        evidence_digest=evidence.bundle_digest,
        question_artifact_digest=questions.artifact_digest,
    )
    revision = apply_answers(evidence, questions, answers)
    lock = WorkspaceLock(
        args.workspace.resolve() / ".locks",
        tenant_id=args.tenant_id,
        run_id=args.run_id,
    )
    if lease is not None:
        target = write_evidence_revision(
            revision,
            args.workspace.resolve() / "revisions",
        )
    else:
        with lock.acquire():
            target = write_evidence_revision(
                revision,
                args.workspace.resolve() / "revisions",
            )
    _print(
        {
            "status": "evidence_revised",
            "evidence_digest": revision.evidence.bundle_digest,
            "revision": str(target),
            "next_commands": ["authorize", "approve --boundary evidence"],
        }
    )
    return revision, target


def _load_subject(path: Path) -> ExposureSubject:
    try:
        document = json.loads(path.resolve().read_text(encoding="utf-8"))
        return ExposureSubject.model_validate(document)
    except Exception as exc:
        raise GuidedCliError(
            "exposure_subject_invalid",
            f"cannot load model exposure subject: {type(exc).__name__}: {exc}",
            recovery="use model_exposure_subject.json from verified intake",
        ) from exc


def _run_authorize(
    args: argparse.Namespace,
    *,
    lease: WorkspaceLease | None = None,
) -> Path:
    subject = _load_subject(args.subject)
    authorization = (
        authorize_model_exposure_by_human(subject, authorized_by=args.authorized_by)
        if args.authorized_by is not None
        else authorize_model_exposure_by_policy(
            subject,
            organizational_policy_digest=args.organizational_policy_digest,
        )
    )
    output = args.output or args.workspace.resolve() / "exposure_authorization.json"
    lock = WorkspaceLock(
        args.workspace.resolve() / ".locks",
        tenant_id=args.tenant_id,
        run_id=args.run_id,
    )
    if lease is not None:
        write_exposure_authorization(authorization, output)
    else:
        with lock.acquire():
            write_exposure_authorization(authorization, output)
    _print(
        {
            "status": "model_exposure_authorized",
            "authorization_digest": authorization.authorization_digest,
            "output": str(output.resolve()),
            "note": "This is not final release approval.",
        }
    )
    return Path(output).resolve()


def _run_evidence_approval(
    args: argparse.Namespace,
    *,
    lease: WorkspaceLease | None = None,
) -> tuple[Path, str]:
    required = {
        "--approved-by": args.approved_by,
        "--source-bundle-digest": args.source_bundle_digest,
        "--normalized-bundle-digest": args.normalized_bundle_digest,
        "--output": args.output,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise GuidedCliError(
            "evidence_approval_inputs_missing",
            f"evidence approval requires: {', '.join(missing)}",
            recovery="provide every digest from the verified intake and an output path",
        )
    document = {
        "approval_version": MIGRATION_APPROVAL_VERSION,
        "approved_by": args.approved_by,
        "source_bundle_digest": args.source_bundle_digest,
        "normalized_bundle_digest": args.normalized_bundle_digest,
        "migration_record_digest": args.migration_record_digest,
        "acknowledged_warnings": sorted(set(args.acknowledge_warning)),
        "acknowledged_findings": sorted(set(args.acknowledge_finding)),
        "note": args.note,
    }
    approval = NormalizedEvidenceApproval.model_validate(document)
    lock = WorkspaceLock(
        args.workspace.resolve() / ".locks",
        tenant_id=args.tenant_id,
        run_id=args.run_id,
    )
    if lease is not None:
        write_canonical_json(approval.model_dump(mode="json"), args.output)
    else:
        with lock.acquire():
            write_canonical_json(approval.model_dump(mode="json"), args.output)
    _print(
        {
            "status": "evidence_approved",
            "approval_digest": sha256_json(approval.model_dump(mode="json")),
            "output": str(args.output.resolve()),
            "next_command": "draft",
            "note": "This approval cannot substitute for final release approval.",
        }
    )
    return args.output.resolve(), approval.normalized_bundle_digest


def main() -> None:
    parser = _parser()
    args, remainder = parser.parse_known_args()
    try:
        if args.command == "author":
            _run_author(args, remainder)
        elif args.command == "resume":
            if remainder:
                raise GuidedCliError(
                    "unexpected_arguments",
                    f"resume received unknown arguments: {remainder!r}",
                    recovery="run bfcl_author.py resume --help",
                )
            _run_resume(args)
        elif args.command == "purge-cache":
            if remainder:
                raise GuidedCliError(
                    "unexpected_arguments",
                    f"purge-cache received unknown arguments: {remainder!r}",
                    recovery="run bfcl_author.py purge-cache --help",
                )
            if args.execute and args.expected_plan_digest is None:
                raise GuidedCliError(
                    "cache_purge_plan_required",
                    "execute requires the digest returned by a prior dry-run",
                    recovery=(
                        "run purge-cache without --execute, review the eligible hashes, "
                        "then pass --expected-plan-digest"
                    ),
                )
            cache_path = args.cache or infer_authoring_cache_path(
                args.workspace,
                tenant_id=args.tenant_id,
                run_id=args.run_id,
            )
            plan, audit = purge_authoring_cache(
                args.workspace,
                cache_path,
                tenant_id=args.tenant_id,
                run_id=args.run_id,
                actor=args.actor,
                reason_code=args.reason_code,
                dry_run=not args.execute,
                expected_plan_digest=args.expected_plan_digest,
            )
            _print(
                {
                    "status": "dry_run" if audit.dry_run else "purged",
                    "cache": str(cache_path.resolve()),
                    "plan_digest": plan.plan_digest,
                    "eligible_request_hashes": list(
                        plan.eligible_request_hashes
                    ),
                    "retained_count": audit.retained_count,
                    "purged_count": audit.purged_count,
                    "audit": str(
                        args.workspace.resolve()
                        / ".events"
                        / CACHE_PURGE_AUDIT_FILE_NAME
                    ),
                    "audit_record_digest": audit.record_digest,
                }
            )
        elif args.command == "answer":
            gate, resumed = _current_session(args, "answer")
            with resumed:
                revision, target = _run_answer(args, lease=resumed.lease)
                evidence_path = target / "evidence.json"
                _commit_transition(
                    args,
                    gate,
                    resumed,
                    phase="evidence_revised",
                    updates={
                        "evidence": bind_artifact(
                            args.workspace,
                            evidence_path,
                            digest_kind="canonical_json",
                        ),
                        "source_identity_digest": sha256_json(
                            revision.evidence.identity.model_dump(mode="json")
                        ),
                        "evidence_bundle_digest": revision.evidence.bundle_digest,
                        "revision_content_address": revision.evidence.bundle_digest,
                    },
                )
        elif args.command == "authorize":
            gate, resumed = _current_session(args, "authorize_exposure")
            with resumed:
                output = _run_authorize(args, lease=resumed.lease)
                _commit_transition(
                    args,
                    gate,
                    resumed,
                    phase="exposure_authorized",
                    updates={
                        "exposure_authorization": bind_artifact(
                            args.workspace,
                            output,
                            digest_kind="canonical_json",
                        )
                    },
                )
        elif args.command == "approve":
            if args.boundary == "release":
                gate, resumed = _current_session(args, "approve_release")
                release_arguments = list(remainder)
                if args.approved_by is not None:
                    release_arguments[:0] = ["--approved-by", args.approved_by]
                if args.output is not None:
                    release_arguments[:0] = ["--output", str(args.output)]
                if args.note is not None:
                    release_arguments[:0] = ["--note", args.note]
                for risk in reversed(args.acknowledge_finding):
                    release_arguments[:0] = ["--acknowledge-risk", risk]
                with resumed:
                    _delegate(
                        "nemotron.steps.byob.scripts.approve_authoring_review",
                        release_arguments,
                    )
                    output = _required_path_argument(release_arguments, "--output")
                    _commit_transition(
                        args,
                        gate,
                        resumed,
                        phase="release_approved",
                        updates={
                            "release_approval": bind_artifact(
                                args.workspace,
                                output,
                                digest_kind="canonical_json",
                            )
                        },
                    )
            else:
                if remainder:
                    raise GuidedCliError(
                        "unexpected_arguments",
                        f"evidence approval received unknown arguments: {remainder!r}",
                        recovery="run bfcl_author.py approve --help",
                    )
                gate, resumed = _current_session(args, "approve_evidence")
                with resumed:
                    output, evidence_digest = _run_evidence_approval(
                        args,
                        lease=resumed.lease,
                    )
                    _commit_transition(
                        args,
                        gate,
                        resumed,
                        phase="evidence_approved",
                        updates={
                            "approval": ApprovalBinding(
                                artifact=bind_artifact(
                                    args.workspace,
                                    output,
                                    digest_kind="canonical_json",
                                ),
                                evidence_digest=evidence_digest,
                            )
                        },
                    )
        elif args.command == "draft":
            if "--resolved-authoring-config" in remainder:
                raise GuidedCliError(
                    "resolved_config_override_forbidden",
                    "guided drafting does not accept a different resolved config",
                    recovery="create a new authoring revision for configuration changes",
                )
            gate, resumed = _current_session(args, "draft")
            delegated = [
                "--resolved-authoring-config",
                str(args.workspace.resolve() / RESOLVED_AUTHORING_CONFIG_FILE),
                *remainder,
            ]
            with resumed:
                _delegate(_DELEGATES["draft"], delegated)
                output = _required_path_argument(delegated, "--output")
                draft_root = output / "drafts"
                try:
                    draft_relative = draft_root.relative_to(
                        args.workspace.resolve()
                    ).as_posix()
                except ValueError as exc:
                    raise GuidedCliError(
                        "artifact_path_escape",
                        "guided draft output must stay inside the workspace",
                        recovery="place --output under the guided workspace",
                    ) from exc
                _commit_transition(
                    args,
                    gate,
                    resumed,
                    phase="draft_complete",
                    updates={
                        "draft_root": draft_relative,
                        "draft_provenance": bind_artifact(
                            args.workspace,
                            output / "draft_provenance.json",
                            digest_kind="canonical_json",
                        ),
                    },
                )
        elif args.command == "review":
            gate, resumed = _current_session(args, "review")
            delegated = ["--adapter-kind", args.adapter_kind, *remainder]
            with resumed:
                _delegate(_DELEGATES["review"], delegated)
                output = _required_path_argument(delegated, "--output")
                _commit_transition(
                    args,
                    gate,
                    resumed,
                    phase="review_ready",
                    updates={
                        "review_packet": bind_artifact(
                            args.workspace,
                            output,
                            digest_kind="canonical_json",
                        )
                    },
                )
        elif args.command == "freeze":
            gate, resumed = _current_session(args, "freeze")
            with resumed:
                _delegate(_DELEGATES["freeze"], remainder)
                output = _required_path_argument(remainder, "--output")
                _commit_transition(
                    args,
                    gate,
                    resumed,
                    phase="frozen",
                    updates={
                        "frozen_manifest": bind_artifact(
                            args.workspace,
                            output / "freeze_manifest.json",
                            digest_kind="canonical_json",
                        )
                    },
                )
        elif args.command == "publish":
            gate, resumed = _current_session(args, "publish")
            with resumed:
                _delegate(_DELEGATES["publish"], remainder)
                config = BfclConfig.from_yaml(
                    _required_path_argument(remainder, "--config")
                )
                session_digest = _commit_transition(
                    args,
                    gate,
                    resumed,
                    phase="published",
                    updates={
                        "publication_manifest": bind_artifact(
                            args.workspace,
                            config.output_dir
                            / config.expt_name
                            / "run_manifest.json",
                            digest_kind="canonical_json",
                        )
                    },
                )
                report_path = (
                    config.output_dir
                    / config.expt_name
                    / "stage_cache"
                    / "oracle_validation_report.json"
                )
                if report_path.is_file():
                    report = json.loads(report_path.read_text(encoding="utf-8"))
                    config_fingerprint = report.get("validation_config_fingerprint")
                    if (
                        isinstance(config_fingerprint, str)
                        and not config_fingerprint.startswith("sha256:")
                    ):
                        config_fingerprint = f"sha256:{config_fingerprint}"
                    pack_fingerprint = str(report["pack_fingerprint"])
                    if not pack_fingerprint.startswith("sha256:"):
                        pack_fingerprint = f"sha256:{pack_fingerprint}"
                    emit_authoring_event(
                        _event_sink(args.workspace),
                        "validation_verdict",
                        ValidationVerdictPayload(
                            stage="publication",
                            tier=report["tier"],
                            gold_eligible=report["gold_eligible"],
                            pack_fingerprint=pack_fingerprint,
                            validation_report_digest=(
                                "sha256:"
                                + hashlib.sha256(report_path.read_bytes()).hexdigest()
                            ),
                            validation_config_fingerprint=config_fingerprint,
                        ),
                        tenant_id=args.tenant_id,
                        run_id=args.run_id,
                        session_digest=session_digest,
                    )
        else:  # pragma: no cover - argparse keeps this unreachable
            parser.error(f"unsupported command {args.command}")
    except GuidedCliError as exc:
        _emit_cli_refusal(args, exc.code)
        _print(
            {
                "status": "fail",
                "code": exc.code,
                "reason": exc.detail,
                "recovery": exc.recovery,
            }
        )
        raise SystemExit(1) from exc
    except (OSError, ValueError) as exc:
        _emit_cli_refusal(
            args,
            str(getattr(exc, "code", "guided_command_failed")),
        )
        _print(
            {
                "status": "fail",
                "code": getattr(exc, "code", "guided_command_failed"),
                "reason": str(exc),
                "recovery": getattr(
                    exc,
                    "recovery",
                    f"run bfcl_author.py {args.command} --help",
                ),
            }
        )
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
