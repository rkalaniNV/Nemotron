from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from nemotron.steps.byob.runtime.mcp.ablation import (
    AblationInput,
    build_ablation_report,
)
from nemotron.steps.byob.runtime.mcp.ablation_collection import digest_artifact_tree
from nemotron.steps.byob.runtime.mcp.ablation_domain_bundle import (
    DomainBundleError,
    load_domain_bundle,
)
from nemotron.steps.byob.runtime.mcp.ablation_evaluator_pin import (
    EvaluatorPin,
    EvaluatorPinError,
    build_pinned_evaluator,
    build_unpinned_evaluator,
    parse_evaluator_pin,
    write_evaluator_pin,
)
from nemotron.steps.byob.runtime.mcp.ablation_review import (
    REQUIRED_REVIEW_CHECKLIST,
    DomainReviewAttestation,
    DomainReviewError,
    ReviewAuthority,
    ReviewedBundle,
    build_domain_review_attestation,
)
from nemotron.steps.byob.runtime.mcp.ablation_rollout import (
    CompletedDomainEvidence,
    RolloutEvidenceError,
    RunExclusion,
    VerifiedDomainBundle,
    build_missing_domain,
    build_rollout,
    load_rollout,
    publish_domain_evidence,
    verify_domain_bundle,
    write_domain_evidence,
    write_rollout,
)
from nemotron.steps.byob.runtime.pack_authoring.artifacts import (
    sha256_json,
    write_canonical_json,
)

_SCHEDULE = (
    ("manual", 1),
    ("llm_backend", 1),
    ("llm_mcp", 1),
    ("llm_mcp", 2),
    ("manual", 2),
    ("llm_backend", 2),
    ("llm_backend", 3),
    ("llm_mcp", 3),
    ("manual", 3),
)
_OPERATOR = "operator@example"
_REVIEWER = "reviewer@example"
_KEY_ID = "reviewer-key-1"
_LAST_RUN_FINISHED = datetime(2026, 8, 29, 0, 2, tzinfo=timezone.utc)
_REVIEWED_AT = _LAST_RUN_FINISHED + timedelta(hours=1)


def _pinned_evaluator() -> EvaluatorPin:
    return build_pinned_evaluator(
        provider="azure",
        served_model="openai/gpt-5.6-sol",
        api_base="https://integrate.api.nvidia.com/v1",
        credential_env_var="NGC_API_KEY",
        weight_source="azure",
        weight_model="openai/gpt-5.6-sol",
        weights_digest="sha256:" + "b" * 64,
        pin_evidence_digest="sha256:" + "c" * 64,
    )


def _authority(reviewer_identity: str = _REVIEWER, key_id: str = _KEY_ID) -> ReviewAuthority:
    return ReviewAuthority(
        reviewer_identity=reviewer_identity,
        key_id=key_id,
        private_key=Ed25519PrivateKey.generate(),
    )


def _ablation_input(domain: str, evaluator_model: str, *, scored: bool) -> AblationInput:
    domain_digest = "sha256:" + hashlib.sha256(domain.encode("utf-8")).hexdigest()
    observations = []
    for sequence, (flow, repetition) in enumerate(_SCHEDULE, start=1):
        observations.append(
            {
                "flow": flow,
                "repetition": repetition,
                "sequence": sequence,
                "run_digest": f"sha256:{sequence:064x}",
                "user_authored_fields": sequence,
                "authoring_minutes": 1.0,
                "review_minutes": 1.0,
                "validation_pass_rate": 1.0,
                "tool_coverage": 1.0,
                "replay_stability": 1.0,
                "benchmark_rows": 3,
                "evaluation_score": 0.8 if scored else None,
                "evaluation_score_stderr": 0.01 if scored else None,
            }
        )
    return AblationInput.model_validate(
        {
            "schema_version": "bfcl-onboarding-ablation-input-v2",
            "experiment_id": f"{domain}-v1",
            "domain_artifact_digest": domain_digest,
            "evaluator_model": evaluator_model,
            "evaluation_config_digest": "sha256:" + "a" * 64,
            "held_out_policy_digest": "sha256:" + "a" * 64,
            "repetitions_per_flow": 3,
            "observations": observations,
        }
    )


@dataclass
class RawDomain:
    domain_id: str
    root: Path
    protocol: Path
    input_path: Path
    report_path: Path
    observations: list[Path]
    states: list[Path]
    artifacts: list[Path]
    exclusions: list[RunExclusion]
    pin: EvaluatorPin


def _raw_domain(
    tmp_path: Path,
    domain: str = "inventory",
    *,
    pin: EvaluatorPin | None = None,
) -> RawDomain:
    """Write nine real run bundles whose digests bind to their own artifacts."""
    evaluator_pin = pin if pin is not None else _pinned_evaluator()
    scored = evaluator_pin.status == "pinned"
    root = tmp_path / domain
    root.mkdir()
    protocol = root / "protocol.md"
    protocol.write_text(f"# Locked {domain} protocol\n", encoding="utf-8")
    source = _ablation_input(domain, evaluator_pin.evaluator_model, scored=scored)
    input_path = write_canonical_json(source.model_dump(mode="json"), root / "input.json")
    report_path = write_canonical_json(build_ablation_report(source), root / "report.json")
    observations: list[Path] = []
    states: list[Path] = []
    artifacts: list[Path] = []
    exclusions: list[RunExclusion] = []
    for observation in source.observations:
        artifact = root / f"artifact-{observation.sequence}"
        artifact.mkdir()
        (artifact / "run.json").write_text(
            json.dumps({"domain": domain, "sequence": observation.sequence}),
            encoding="utf-8",
        )
        # Bind the fixture's observation to the actual immutable artifact.
        document = observation.model_dump(mode="json")
        document["run_digest"] = digest_artifact_tree(artifact)
        observations.append(
            write_canonical_json(document, root / f"observation-{observation.sequence}.json")
        )
        artifacts.append(artifact)
        states.append(
            write_canonical_json(
                {
                    "schema_version": "bfcl-onboarding-ablation-collection-v1",
                    "flow": observation.flow,
                    "repetition": observation.repetition,
                    "sequence": observation.sequence,
                    "started_at": "2026-08-29T00:00:00+00:00",
                    "review_started_at": "2026-08-29T00:01:00+00:00",
                    "finished_at": _LAST_RUN_FINISHED.isoformat(),
                    "observation_written": True,
                },
                root / f"state-{observation.sequence}.json",
            )
        )
        exclusions.append(
            RunExclusion(
                flow=observation.flow,
                repetition=observation.repetition,
                sequence=observation.sequence,
                excluded_authoring_minutes=0.0,
                excluded_review_minutes=0.0,
            )
        )

    # Rebind the assembled input and report to the actual run digests.
    rebound = source.model_copy(
        update={
            "observations": tuple(
                type(observation).model_validate(json.loads(path.read_text(encoding="utf-8")))
                for observation, path in zip(source.observations, observations, strict=True)
            )
        }
    )
    write_canonical_json(rebound.model_dump(mode="json"), input_path)
    write_canonical_json(build_ablation_report(rebound), report_path)
    return RawDomain(
        domain_id=domain,
        root=root,
        protocol=protocol,
        input_path=input_path,
        report_path=report_path,
        observations=observations,
        states=states,
        artifacts=artifacts,
        exclusions=exclusions,
        pin=evaluator_pin,
    )


def _verified(raw: RawDomain, *, operator_identity: str = _OPERATOR) -> VerifiedDomainBundle:
    return verify_domain_bundle(
        domain_id=raw.domain_id,
        protocol_path=raw.protocol,
        ablation_input_path=raw.input_path,
        ablation_report_path=raw.report_path,
        observation_paths=raw.observations,
        state_paths=raw.states,
        run_artifact_paths=raw.artifacts,
        exclusions=raw.exclusions,
        operator_identity=operator_identity,
        evaluator_pin=raw.pin,
    )


def _attestation(
    verified: VerifiedDomainBundle,
    authority: ReviewAuthority,
    *,
    reviewed_at: datetime = _REVIEWED_AT,
    bundle: ReviewedBundle | None = None,
    checklist: dict[str, bool] | None = None,
) -> DomainReviewAttestation:
    return build_domain_review_attestation(
        authority=authority,
        domain_id=verified.domain_id,
        experiment_id=verified.experiment_id,
        operator_identity=verified.operator_identity,
        reviewed_at=reviewed_at,
        bundle=bundle if bundle is not None else verified.reviewed_bundle,
        checklist=(
            checklist
            if checklist is not None
            else dict.fromkeys(sorted(REQUIRED_REVIEW_CHECKLIST), True)
        ),
    )


def _complete_domain(
    tmp_path: Path,
    domain: str = "inventory",
    *,
    pin: EvaluatorPin | None = None,
):
    raw = _raw_domain(tmp_path, domain, pin=pin)
    verified = _verified(raw)
    authority = _authority()
    evidence = publish_domain_evidence(
        verified,
        review_attestation=_attestation(verified, authority),
        trusted_reviewer_keys={authority.key_id: authority.public_key},
    )
    return evidence, raw


def test_descriptive_protocol_cannot_be_promoted_to_causal(tmp_path: Path) -> None:
    domains = [
        _complete_domain(tmp_path, domain)[0]
        for domain in ("tiny_library", "inventory", "banking_vn")
    ]
    rollout = build_rollout(
        domains,
        evidence_kind="descriptive",
        decided_by="release-board@example",
        rationale="The current protocol is descriptive.",
    )
    output = write_rollout(rollout, tmp_path / "rollout.json")

    assert rollout.decision.blockers == ("cross_domain:causal_design_unimplemented",)
    with pytest.raises(RolloutEvidenceError, match="unverified completed-domain"):
        load_rollout(output)
    with pytest.raises(RolloutEvidenceError, match="causal_design_unimplemented"):
        build_rollout(
            domains,
            evidence_kind="causal",
            decided_by="release-board@example",
            rationale="This must fail.",
        )


def test_missing_runs_force_descriptive_decision(tmp_path: Path) -> None:
    complete = _complete_domain(tmp_path)[0]
    domains = [
        complete,
        build_missing_domain(
            "banking_vn",
            reason_code="live_run_missing",
            detail="Nine real runs have not been collected.",
        ),
        build_missing_domain(
            "tiny_library",
            reason_code="reviewer_missing",
            detail="The pilot has no independent reviewer attestation.",
        ),
    ]

    descriptive = build_rollout(
        domains,
        evidence_kind="descriptive",
        decided_by="bfcl-readiness",
        rationale="Publish gaps without filling observations.",
    )
    assert descriptive.decision.evidence_kind == "descriptive"
    assert descriptive.decision.blockers == (
        "banking_vn:missing_runs",
        "cross_domain:causal_design_unimplemented",
        "tiny_library:missing_runs",
    )
    path = write_rollout(descriptive, tmp_path / "missing-rollout.json")
    with pytest.raises(RolloutEvidenceError, match="unverified completed-domain"):
        load_rollout(path)
    with pytest.raises(RolloutEvidenceError, match="causal evidence requirements"):
        build_rollout(
            domains,
            evidence_kind="causal",
            decided_by="bfcl-readiness",
            rationale="This must fail.",
        )
    readiness = build_rollout(
        [
            build_missing_domain(
                domain,
                reason_code="live_run_missing",
                detail="Evidence is not yet publishable.",
            )
            for domain in ("tiny_library", "inventory", "banking_vn")
        ],
        evidence_kind="descriptive",
        decided_by="bfcl-readiness",
        rationale="Publish only explicit missing-run status.",
    )
    readiness_path = write_rollout(readiness, tmp_path / "readiness.json")
    assert load_rollout(readiness_path) == readiness


def test_unpinned_target_route_cannot_support_causal_decision(tmp_path: Path) -> None:
    unpinned = build_unpinned_evaluator(
        reason_code="immutable_pin_unavailable",
        declared_route="azure/openai/gpt-5.6-sol",
        detail="The hosted route does not publish an immutable revision or weights digest.",
    )
    domains = [
        _complete_domain(tmp_path, domain, pin=unpinned)[0]
        for domain in ("tiny_library", "inventory", "banking_vn")
    ]
    descriptive = build_rollout(
        domains,
        evidence_kind="descriptive",
        decided_by="bfcl-readiness",
        rationale="The route does not identify immutable weights.",
    )
    assert descriptive.decision.blockers == (
        "banking_vn:target_evaluation_missing",
        "banking_vn:target_model_pin_missing",
        "cross_domain:causal_design_unimplemented",
        "inventory:target_evaluation_missing",
        "inventory:target_model_pin_missing",
        "tiny_library:target_evaluation_missing",
        "tiny_library:target_model_pin_missing",
    )
    with pytest.raises(RolloutEvidenceError, match="target_model_pin_missing"):
        build_rollout(
            domains,
            evidence_kind="causal",
            decided_by="bfcl-readiness",
            rationale="This must fail.",
        )


@pytest.mark.parametrize("revision", ["main", "latest", "refs/heads/release", "v2.1", "stable"])
def test_moving_pointer_is_never_an_immutable_pin(revision: str) -> None:
    with pytest.raises(EvaluatorPinError, match="not immutably pinned"):
        build_pinned_evaluator(
            provider="azure",
            served_model="openai/gpt-5.6-sol",
            api_base="https://integrate.api.nvidia.com/v1",
            credential_env_var="NGC_API_KEY",
            weight_source="azure",
            weight_model="openai/gpt-5.6-sol",
            revision=revision,
            pin_evidence_digest="sha256:" + "c" * 64,
        )


def test_pin_requires_an_immutable_identity_and_refuses_credentials() -> None:
    with pytest.raises(EvaluatorPinError, match="not immutably pinned"):
        build_pinned_evaluator(
            provider="azure",
            served_model="openai/gpt-5.6-sol",
            api_base="https://integrate.api.nvidia.com/v1",
            credential_env_var="NGC_API_KEY",
            weight_source="azure",
            weight_model="openai/gpt-5.6-sol",
            pin_evidence_digest="sha256:" + "c" * 64,
        )
    with pytest.raises(ValueError, match="credential material"):
        build_pinned_evaluator(
            provider="azure",
            served_model="openai/gpt-5.6-sol",
            api_base="https://integrate.api.nvidia.com/v1",
            credential_env_var="nvapi-0123456789",
            weight_source="azure",
            weight_model="openai/gpt-5.6-sol",
            weights_digest="sha256:" + "b" * 64,
            pin_evidence_digest="sha256:" + "c" * 64,
        )
    with pytest.raises(ValueError, match="https"):
        build_pinned_evaluator(
            provider="azure",
            served_model="openai/gpt-5.6-sol",
            api_base="http://integrate.api.nvidia.com/v1",
            credential_env_var="NGC_API_KEY",
            weight_source="azure",
            weight_model="openai/gpt-5.6-sol",
            weights_digest="sha256:" + "b" * 64,
            pin_evidence_digest="sha256:" + "c" * 64,
        )


def test_pin_must_describe_the_evaluator_the_runs_were_scored_by(tmp_path: Path) -> None:
    raw = _raw_domain(tmp_path)
    other = build_pinned_evaluator(
        provider="azure",
        served_model="openai/gpt-5.6-sol",
        api_base="https://integrate.api.nvidia.com/v1",
        credential_env_var="NGC_API_KEY",
        weight_source="azure",
        weight_model="openai/gpt-5.6-sol",
        weights_digest="sha256:" + "d" * 64,
        pin_evidence_digest="sha256:" + "c" * 64,
    )
    raw.pin = other
    with pytest.raises(RolloutEvidenceError, match="does not describe the ablation input"):
        _verified(raw)


def test_scored_runs_cannot_claim_the_target_model_was_never_run(tmp_path: Path) -> None:
    evidence, _raw = _complete_domain(tmp_path)
    document = evidence.model_dump(mode="json")
    document["evaluator_pin"] = build_unpinned_evaluator(
        reason_code="target_evaluation_not_run",
        detail="The target model was never called.",
    ).model_dump(mode="json")
    document["evaluator_model"] = "not_run"
    document["evidence_digest"] = sha256_json(
        {key: value for key, value in document.items() if key != "evidence_digest"}
    )
    with pytest.raises(ValueError, match="never run"):
        CompletedDomainEvidence.model_validate(document)


def test_never_run_target_model_is_publishable_without_a_fabricated_pin(
    tmp_path: Path,
) -> None:
    """The pilot shape: real runs, real review, and no target-model evaluation."""
    not_run = build_unpinned_evaluator(
        reason_code="target_evaluation_not_run",
        detail="The onboarding pilot measured effort and quality only.",
    )
    evidence, _raw = _complete_domain(tmp_path, "tiny_library", pin=not_run)
    assert evidence.evaluator_model == "not_run"
    assert evidence.evaluation_scores_complete is False
    assert evidence.evaluator_pin.status == "unpinned"

    rollout = build_rollout(
        [
            evidence,
            build_missing_domain(
                "inventory",
                reason_code="live_run_missing",
                detail="Nine real runs have not been collected.",
            ),
            build_missing_domain(
                "banking_vn",
                reason_code="live_run_missing",
                detail="Nine real runs have not been collected.",
            ),
        ],
        evidence_kind="descriptive",
        decided_by="bfcl-readiness",
        rationale="One reviewed domain, two honestly missing domains.",
    )
    assert rollout.decision.blockers == (
        "banking_vn:missing_runs",
        "cross_domain:causal_design_unimplemented",
        "inventory:missing_runs",
        "tiny_library:target_evaluation_missing",
        "tiny_library:target_model_pin_missing",
    )


def test_review_signature_must_come_from_a_trusted_reviewer_key(tmp_path: Path) -> None:
    raw = _raw_domain(tmp_path)
    verified = _verified(raw)
    authority = _authority()
    attestation = _attestation(verified, authority)
    stranger = _authority(key_id="stranger-key")

    with pytest.raises(DomainReviewError, match="not trusted"):
        publish_domain_evidence(
            verified,
            review_attestation=attestation,
            trusted_reviewer_keys={stranger.key_id: stranger.public_key},
        )
    with pytest.raises(DomainReviewError, match="signature is invalid"):
        publish_domain_evidence(
            verified,
            review_attestation=attestation,
            trusted_reviewer_keys={authority.key_id: stranger.public_key},
        )
    with pytest.raises(DomainReviewError, match="not trusted"):
        publish_domain_evidence(
            verified,
            review_attestation=attestation,
            trusted_reviewer_keys={},
        )


def test_review_cannot_cover_a_different_bundle_than_the_one_published(tmp_path: Path) -> None:
    raw = _raw_domain(tmp_path)
    verified = _verified(raw)
    authority = _authority()

    # The reviewer signed the bundle, then the operator edited one run artifact.
    attestation = _attestation(verified, authority)
    (raw.artifacts[0] / "smuggled.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RolloutEvidenceError, match="run artifact digest does not match"):
        _verified(raw)
    # The signature still describes the tree as reviewed, so it cannot cover the edit.
    assert attestation.bundle == verified.reviewed_bundle

    # A reviewer who signs digests that were never derived from these files.
    forged = verified.reviewed_bundle.model_copy(
        update={"protocol_digest": "sha256:" + "e" * 64}
    )
    with pytest.raises(DomainReviewError, match="different evidence bundle"):
        publish_domain_evidence(
            verified,
            review_attestation=_attestation(verified, authority, bundle=forged),
            trusted_reviewer_keys={authority.key_id: authority.public_key},
        )


def test_reviewer_must_be_independent_of_the_operator(tmp_path: Path) -> None:
    raw = _raw_domain(tmp_path)
    verified = _verified(raw)
    self_reviewer = _authority(reviewer_identity=_OPERATOR)

    with pytest.raises(DomainReviewError, match="cannot attest to a domain they operated"):
        _attestation(verified, self_reviewer)

    # A reviewer of a different domain's operator cannot be reused here either.
    other = _authority(reviewer_identity="second-operator@example")
    attestation = _attestation(verified, other)
    document = attestation.model_dump(mode="json")
    document["operator_identity"] = document["reviewer_identity"]
    with pytest.raises(ValueError, match="must differ"):
        DomainReviewAttestation.model_validate(document)


def test_review_cannot_predate_the_runs_or_skip_a_checklist_item(tmp_path: Path) -> None:
    raw = _raw_domain(tmp_path)
    verified = _verified(raw)
    authority = _authority()

    early = _attestation(
        verified,
        authority,
        reviewed_at=_LAST_RUN_FINISHED - timedelta(minutes=1),
    )
    with pytest.raises(DomainReviewError, match="cannot predate"):
        publish_domain_evidence(
            verified,
            review_attestation=early,
            trusted_reviewer_keys={authority.key_id: authority.public_key},
        )

    partial = dict.fromkeys(sorted(REQUIRED_REVIEW_CHECKLIST), True)
    partial["run_artifacts_verified"] = False
    with pytest.raises(ValueError, match="checklist is incomplete"):
        _attestation(verified, authority, checklist=partial)

    dropped = dict(partial)
    del dropped["run_artifacts_verified"]
    with pytest.raises(ValueError, match="exactly the required review items"):
        _attestation(verified, authority, checklist=dropped)


def test_published_evidence_names_the_reviewer_and_the_signed_attestation(
    tmp_path: Path,
) -> None:
    evidence, _raw = _complete_domain(tmp_path)
    assert evidence.reviewer_identity == _REVIEWER
    assert evidence.reviewer_key_id == _KEY_ID
    assert evidence.reviewed_at == _REVIEWED_AT.isoformat()
    assert evidence.review_attestation_digest.startswith("sha256:")
    assert evidence.operator_identity != evidence.reviewer_identity
    assert evidence.evaluator_pin.status == "pinned"
    assert evidence.evaluator_model == evidence.evaluator_pin.canonical_id
    assert "NGC_API_KEY" in json.dumps(evidence.model_dump(mode="json"))
    assert "nvapi-" not in json.dumps(evidence.model_dump(mode="json"))


def test_tampered_attestation_digest_or_signature_fails_closed(tmp_path: Path) -> None:
    raw = _raw_domain(tmp_path)
    verified = _verified(raw)
    authority = _authority()
    attestation = _attestation(verified, authority)

    tampered = attestation.model_dump(mode="json")
    tampered["reviewer_identity"] = "someone-else@example"
    with pytest.raises(ValueError, match="attestation_digest mismatch"):
        DomainReviewAttestation.model_validate(tampered)

    truncated = attestation.model_dump(mode="json")
    truncated["signature"] = truncated["signature"][:-4]
    with pytest.raises(ValueError):
        DomainReviewAttestation.model_validate(truncated)


def test_one_domain_cannot_fill_three_rollout_slots(tmp_path: Path) -> None:
    source = _complete_domain(tmp_path)[0]
    cloned = []
    for domain_id in ("tiny_library", "inventory", "banking_vn"):
        document = source.model_dump(mode="json")
        document["domain_id"] = domain_id
        document["evidence_digest"] = sha256_json(
            {key: value for key, value in document.items() if key != "evidence_digest"}
        )
        cloned.append(type(source).model_validate(document))

    with pytest.raises(ValueError, match="unique experiment_id"):
        build_rollout(
            cloned,
            evidence_kind="descriptive",
            decided_by="bfcl-readiness",
            rationale="Cloned evidence must fail.",
        )


def test_partial_domain_publishes_only_actual_missing_run() -> None:
    missing = build_missing_domain(
        "inventory",
        reason_code="live_run_missing",
        detail="Sequence 9 failed and was not substituted.",
        missing_sequences=[9],
    )
    assert [(item.flow, item.repetition, item.sequence) for item in missing.missing_runs] == [
        ("manual", 3, 9)
    ]


def test_domain_evidence_rejects_synthetic_substitution_and_artifact_drift(
    tmp_path: Path,
) -> None:
    raw = _raw_domain(tmp_path)
    document = json.loads(raw.observations[0].read_text(encoding="utf-8"))
    document["benchmark_rows"] = 999
    raw.observations[0].write_text(json.dumps(document), encoding="utf-8")

    # The published input still names the original observation.
    with pytest.raises(RolloutEvidenceError, match="not bound to ablation input"):
        _verified(raw)


def _bundle_manifest(raw: RawDomain, *, attestation: Path | None = None) -> Path:
    """Declare the raw bundle with paths relative to the manifest, as an operator would."""
    root = raw.root
    document = {
        "schema_version": "bfcl-onboarding-domain-bundle-v1",
        "domain_id": raw.domain_id,
        "operator_identity": _OPERATOR,
        "protocol": raw.protocol.name,
        "ablation_input": raw.input_path.name,
        "ablation_report": raw.report_path.name,
        "evaluator_pin": "evaluator-pin.json",
        "runs": [
            {
                "sequence": sequence,
                "observation": raw.observations[sequence - 1].name,
                "state": raw.states[sequence - 1].name,
                "run_artifact": raw.artifacts[sequence - 1].name,
            }
            for sequence in range(1, 10)
        ],
    }
    if attestation is not None:
        document["review_attestation"] = attestation.name
    write_evaluator_pin(raw.pin, root / "evaluator-pin.json")
    return write_canonical_json(document, root / "bundle.json")


def _cli(module: str, *arguments: str) -> dict:
    result = subprocess.run(
        [sys.executable, "-m", f"nemotron.steps.byob.scripts.{module}", *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    payload["exit_code"] = result.returncode
    return payload


def test_reviewer_and_publisher_command_path_binds_one_bundle(tmp_path: Path) -> None:
    raw = _raw_domain(tmp_path, "tiny_library")
    manifest = _bundle_manifest(raw)
    keys = tmp_path / "keys"

    generated = _cli(
        "review_bfcl_domain_evidence",
        "keygen",
        "--private-key",
        str(keys / "reviewer.pem"),
        "--public-key",
        str(keys / "reviewer.pub"),
    )
    assert generated["status"] == "generated"
    assert (keys / "reviewer.pem").stat().st_mode & 0o077 == 0

    verified = _cli(
        "review_bfcl_domain_evidence",
        "verify",
        "--bundle",
        str(manifest),
    )
    assert verified["status"] == "verified"
    assert verified["operator_identity"] == _OPERATOR
    assert len(verified["reviewed_bundle"]["run_artifact_digests"]) == 9

    unconfirmed = _cli(
        "review_bfcl_domain_evidence",
        "sign",
        "--bundle",
        str(manifest),
        "--private-key",
        str(keys / "reviewer.pem"),
        "--reviewer-identity",
        _REVIEWER,
        "--reviewer-key-id",
        _KEY_ID,
        "--output",
        str(raw.root / "review.json"),
        "--reviewed-at",
        _REVIEWED_AT.isoformat(),
        "--confirm",
        "protocol_followed",
    )
    assert unconfirmed["exit_code"] == 1
    assert "missing" in unconfirmed["reason"]

    signed = _cli(
        "review_bfcl_domain_evidence",
        "sign",
        "--bundle",
        str(manifest),
        "--private-key",
        str(keys / "reviewer.pem"),
        "--reviewer-identity",
        _REVIEWER,
        "--reviewer-key-id",
        _KEY_ID,
        "--output",
        str(raw.root / "review.json"),
        "--reviewed-at",
        _REVIEWED_AT.isoformat(),
        *[
            argument
            for item in sorted(REQUIRED_REVIEW_CHECKLIST)
            for argument in ("--confirm", item)
        ],
    )
    assert signed["status"] == "signed"
    assert signed["reviewer_identity"] == _REVIEWER

    accepted = _cli(
        "review_bfcl_domain_evidence",
        "check",
        "--bundle",
        str(manifest),
        "--attestation",
        str(raw.root / "review.json"),
        "--public-key",
        str(keys / "reviewer.pub"),
        "--reviewer-key-id",
        _KEY_ID,
    )
    assert accepted["status"] == "accepted"

    # Publish one reviewed domain beside two honestly missing ones.
    manifest = _bundle_manifest(raw, attestation=raw.root / "review.json")
    for domain in ("inventory", "banking_vn"):
        write_domain_evidence(
            build_missing_domain(
                domain,
                reason_code="live_run_missing",
                detail="Nine real runs have not been collected.",
            ),
            tmp_path / f"{domain}.missing.json",
        )
    published = _cli(
        "assemble_bfcl_ablation_rollout",
        "--domain-bundle",
        str(manifest),
        "--trusted-reviewer-key",
        f"{_KEY_ID}={keys / 'reviewer.pub'}",
        "--domain-evidence",
        str(tmp_path / "inventory.missing.json"),
        "--domain-evidence",
        str(tmp_path / "banking_vn.missing.json"),
        "--evidence-kind",
        "descriptive",
        "--decided-by",
        "bfcl-readiness",
        "--rationale",
        "One reviewed domain, two missing domains.",
        "--output",
        str(tmp_path / "rollout.json"),
    )
    assert published["status"] == "written"
    assert published["domains"] == {
        "banking_vn": "missing",
        "inventory": "missing",
        "tiny_library": "complete",
    }
    assert published["blockers"] == [
        "banking_vn:missing_runs",
        "cross_domain:causal_design_unimplemented",
        "inventory:missing_runs",
    ]

    # The publisher's own key list is what decides trust, not the manifest.
    stranger = tmp_path / "stranger"
    _cli(
        "review_bfcl_domain_evidence",
        "keygen",
        "--private-key",
        str(stranger / "other.pem"),
        "--public-key",
        str(stranger / "other.pub"),
    )
    refused = _cli(
        "assemble_bfcl_ablation_rollout",
        "--domain-bundle",
        str(manifest),
        "--trusted-reviewer-key",
        f"{_KEY_ID}={stranger / 'other.pub'}",
        "--domain-evidence",
        str(tmp_path / "inventory.missing.json"),
        "--domain-evidence",
        str(tmp_path / "banking_vn.missing.json"),
        "--evidence-kind",
        "descriptive",
        "--decided-by",
        "bfcl-readiness",
        "--rationale",
        "This must fail.",
        "--output",
        str(tmp_path / "refused.json"),
    )
    assert refused["exit_code"] == 1
    assert "signature is invalid" in refused["reason"]
    assert not (tmp_path / "refused.json").exists()


def test_bundle_manifest_refuses_paths_outside_its_own_directory(tmp_path: Path) -> None:
    raw = _raw_domain(tmp_path)
    _bundle_manifest(raw)
    document = json.loads((raw.root / "bundle.json").read_text(encoding="utf-8"))
    document["protocol"] = "../escaped.md"
    (tmp_path / "escaped.md").write_text("# outside\n", encoding="utf-8")
    escaped = write_canonical_json(document, raw.root / "escaped-bundle.json")
    with pytest.raises(DomainBundleError, match="stay inside the manifest directory"):
        load_domain_bundle(escaped)

    document["protocol"] = str((tmp_path / "escaped.md").resolve())
    absolute = write_canonical_json(document, raw.root / "absolute-bundle.json")
    with pytest.raises(DomainBundleError, match="must be relative"):
        load_domain_bundle(absolute)


def test_exclusions_need_a_reason_and_pins_round_trip(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="require a reason"):
        RunExclusion(
            flow="manual",
            repetition=1,
            sequence=1,
            excluded_authoring_minutes=1.0,
            excluded_review_minutes=0.0,
        )

    pin = _pinned_evaluator()
    assert parse_evaluator_pin(pin.model_dump(mode="json")) == pin

    # Editing the weights breaks the canonical identity before the digest is reached.
    swapped_weights = pin.model_dump(mode="json")
    swapped_weights["weights_digest"] = "sha256:" + "f" * 64
    with pytest.raises(EvaluatorPinError, match="canonical_id must be derived"):
        parse_evaluator_pin(swapped_weights)

    # Editing anything the identity does not cover is caught by the pin digest.
    swapped_evidence = pin.model_dump(mode="json")
    swapped_evidence["pin_evidence_digest"] = "sha256:" + "f" * 64
    with pytest.raises(EvaluatorPinError, match="pin_digest mismatch"):
        parse_evaluator_pin(swapped_evidence)
