"""Fail-closed publication contract for multi-domain onboarding evidence."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, cast

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    model_validator,
)

from nemotron.steps.byob.runtime.mcp.ablation import (
    AblationError,
    FlowName,
    FlowObservation,
    build_ablation_report,
    load_ablation_input,
)
from nemotron.steps.byob.runtime.mcp.ablation_collection import digest_artifact_tree
from nemotron.steps.byob.runtime.mcp.ablation_evaluator_pin import (
    EvaluatorPin,
    PinnedEvaluator,
    UnpinnedEvaluator,
    verify_evaluator_model_binding,
)
from nemotron.steps.byob.runtime.mcp.ablation_review import (
    DomainReviewAttestation,
    ReviewedBundle,
    exclusions_digest,
    verify_domain_review_attestation,
)
from nemotron.steps.byob.runtime.pack_authoring.artifacts import (
    sha256_json,
    write_canonical_json,
)

DOMAIN_EVIDENCE_VERSION = "bfcl-onboarding-domain-evidence-v2"
ROLLOUT_VERSION = "bfcl-onboarding-ablation-rollout-v2"
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTITY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@/+~-]{2,255}$")
_LOCKED_DOMAINS = frozenset({"tiny_library", "inventory", "banking_vn"})
_SCHEDULE: tuple[tuple[FlowName, int], ...] = (
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


class RolloutEvidenceError(AblationError):
    """Raised when rollout evidence cannot support its requested claim."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RunKey(_StrictModel):
    flow: FlowName
    repetition: StrictInt = Field(ge=1, le=3)
    sequence: StrictInt = Field(ge=1, le=9)


class RunExclusion(RunKey):
    excluded_authoring_minutes: StrictFloat = Field(ge=0)
    excluded_review_minutes: StrictFloat = Field(ge=0)
    reason: StrictStr | None = None

    @model_validator(mode="after")
    def validate_reason(self) -> RunExclusion:
        values = (self.excluded_authoring_minutes, self.excluded_review_minutes)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("excluded minutes must be finite")
        if any(value > 0 for value in values) and (
            self.reason is None or not self.reason.strip()
        ):
            raise ValueError("positive excluded minutes require a reason")
        if self.reason is not None and not self.reason.strip():
            raise ValueError("exclusion reason cannot be blank")
        return self


class PublishedRunEvidence(RunKey):
    observation_digest: StrictStr
    collection_state_digest: StrictStr
    run_artifact_digest: StrictStr

    @model_validator(mode="after")
    def validate_digests(self) -> PublishedRunEvidence:
        for field in (
            "observation_digest",
            "collection_state_digest",
            "run_artifact_digest",
        ):
            if _DIGEST.fullmatch(str(getattr(self, field))) is None:
                raise ValueError(f"{field} must be sha256:<64 lowercase hex>")
        return self


class CompletedDomainEvidence(_StrictModel):
    schema_version: Literal["bfcl-onboarding-domain-evidence-v2"]
    status: Literal["complete"]
    domain_id: StrictStr
    experiment_id: StrictStr
    protocol_digest: StrictStr
    domain_artifact_digest: StrictStr
    ablation_input_digest: StrictStr
    ablation_report_digest: StrictStr
    evaluator_model: StrictStr
    evaluator_pin: EvaluatorPin
    evaluation_config_digest: StrictStr
    held_out_policy_digest: StrictStr
    operator_identity: StrictStr
    reviewer_identity: StrictStr
    reviewer_key_id: StrictStr
    review_attestation_digest: StrictStr
    reviewed_at: StrictStr
    observations_are_live: Literal[True]
    synthetic_substitution_allowed: Literal[False]
    evaluation_scores_complete: StrictBool
    runs: tuple[PublishedRunEvidence, ...]
    exclusions: tuple[RunExclusion, ...]
    evidence_digest: StrictStr

    @model_validator(mode="after")
    def validate_domain(self) -> CompletedDomainEvidence:
        _validate_identity(self.domain_id, "domain_id")
        _validate_identity(self.operator_identity, "operator_identity")
        _validate_identity(self.reviewer_identity, "reviewer_identity")
        if self.operator_identity.casefold() == self.reviewer_identity.casefold():
            raise ValueError("reviewer_identity must differ from operator_identity")
        for field in (
            "protocol_digest",
            "domain_artifact_digest",
            "ablation_input_digest",
            "ablation_report_digest",
            "evaluation_config_digest",
            "held_out_policy_digest",
            "review_attestation_digest",
            "evidence_digest",
        ):
            if _DIGEST.fullmatch(str(getattr(self, field))) is None:
                raise ValueError(f"{field} must be sha256:<64 lowercase hex>")
        verify_evaluator_model_binding(
            self.evaluator_pin,
            evaluator_model=self.evaluator_model,
            evaluation_scores_complete=bool(self.evaluation_scores_complete),
        )
        _validate_schedule(self.runs, "runs")
        _validate_schedule(self.exclusions, "exclusions")
        unsigned = self.model_dump(mode="json", exclude={"evidence_digest"})
        if self.evidence_digest != sha256_json(unsigned):
            raise ValueError("domain evidence_digest mismatch")
        return self


class MissingRun(RunKey):
    reason_code: Literal[
        "domain_artifact_missing",
        "live_run_missing",
        "reviewer_missing",
        "target_model_pin_missing",
        "target_evaluation_missing",
        "evidence_invalid",
    ]


class MissingDomainEvidence(_StrictModel):
    status: Literal["missing"]
    domain_id: StrictStr
    missing_runs: tuple[MissingRun, ...]
    detail: StrictStr

    @model_validator(mode="after")
    def validate_missing(self) -> MissingDomainEvidence:
        _validate_identity(self.domain_id, "domain_id")
        if not self.detail.strip():
            raise ValueError("missing-domain detail cannot be blank")
        if not self.missing_runs:
            raise ValueError("missing domain must identify at least one missing run")
        sequences = [item.sequence for item in self.missing_runs]
        if sequences != sorted(set(sequences)):
            raise ValueError("missing_runs must have unique ascending sequence values")
        for item in self.missing_runs:
            expected_flow, expected_repetition = _SCHEDULE[item.sequence - 1]
            if (item.flow, item.repetition) != (
                expected_flow,
                expected_repetition,
            ):
                raise ValueError("missing_runs must match the locked schedule")
        return self


DomainEvidence = CompletedDomainEvidence | MissingDomainEvidence


@dataclass(frozen=True)
class VerifiedDomainBundle:
    """What one domain's raw files prove on their own, with no review attached yet.

    Keeping this separate from :class:`CompletedDomainEvidence` is what lets a
    reviewer sign the same digests publication will recompute: the reviewer runs
    the verification, signs ``reviewed_bundle``, and publication refuses any
    attestation whose bundle differs by a byte.
    """

    domain_id: str
    operator_identity: str
    experiment_id: str
    domain_artifact_digest: str
    evaluator_model: str
    evaluation_config_digest: str
    held_out_policy_digest: str
    evaluator_pin: EvaluatorPin
    runs: tuple[PublishedRunEvidence, ...]
    exclusions: tuple[RunExclusion, ...]
    evaluation_scores_complete: bool
    last_run_finished_at: datetime
    reviewed_bundle: ReviewedBundle


class RolloutDecision(_StrictModel):
    evidence_kind: Literal["descriptive", "causal"]
    decided_by: StrictStr
    rationale: StrictStr
    blockers: tuple[StrictStr, ...]

    @model_validator(mode="after")
    def validate_decision(self) -> RolloutDecision:
        _validate_identity(self.decided_by, "decided_by")
        if not self.rationale.strip():
            raise ValueError("decision rationale cannot be blank")
        if self.evidence_kind == "causal" and self.blockers:
            raise ValueError("causal decision cannot contain blockers")
        return self


class AblationRollout(_StrictModel):
    schema_version: Literal["bfcl-onboarding-ablation-rollout-v2"]
    domains: tuple[DomainEvidence, ...]
    decision: RolloutDecision
    rollout_digest: StrictStr

    @model_validator(mode="after")
    def validate_rollout(self) -> AblationRollout:
        if len(self.domains) != 3:
            raise ValueError("rollout must declare exactly three domains")
        domain_ids = [domain.domain_id for domain in self.domains]
        if len(set(domain_ids)) != len(domain_ids):
            raise ValueError("rollout domain_id values must be unique")
        if set(domain_ids) != _LOCKED_DOMAINS:
            raise ValueError(
                "rollout must contain tiny_library, inventory, and banking_vn"
            )
        complete = [
            domain
            for domain in self.domains
            if isinstance(domain, CompletedDomainEvidence)
        ]
        for field in ("experiment_id", "domain_artifact_digest"):
            values = [str(getattr(domain, field)) for domain in complete]
            if len(values) != len(set(values)):
                raise ValueError(f"completed domains must have unique {field} values")
        run_digests = [
            run.run_artifact_digest for domain in complete for run in domain.runs
        ]
        if len(run_digests) != len(set(run_digests)):
            raise ValueError("completed domains must have disjoint run artifacts")
        expected_blockers = _causal_blockers(self.domains)
        if self.decision.evidence_kind == "causal" and expected_blockers:
            raise ValueError(
                "causal evidence requirements are not met: "
                + "; ".join(expected_blockers)
            )
        if tuple(expected_blockers) != self.decision.blockers:
            raise ValueError("rollout decision blockers do not match evidence")
        unsigned = self.model_dump(mode="json", exclude={"rollout_digest"})
        if self.rollout_digest != sha256_json(unsigned):
            raise ValueError("rollout_digest mismatch")
        return self


def _validate_identity(value: str, field: str) -> None:
    if _IDENTITY.fullmatch(value) is None:
        raise ValueError(f"{field} must be a stable non-secret identifier")


def _validate_schedule(items: tuple[Any, ...], field: str) -> None:
    observed = tuple((item.flow, item.repetition) for item in items)
    expected = tuple(_SCHEDULE)
    sequences = tuple(item.sequence for item in items)
    if observed != expected or sequences != tuple(range(1, 10)):
        raise ValueError(f"{field} must follow the locked nine-run schedule")


def _sha256_file(path: Path) -> str:
    try:
        return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise RolloutEvidenceError(f"cannot digest evidence file {path}: {exc}") from exc


def _load_json(path: Path) -> Any:
    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RolloutEvidenceError(f"{path} repeats JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise RolloutEvidenceError(f"{path} contains non-finite constant {value}")

    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=unique_pairs,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RolloutEvidenceError(f"cannot load evidence file {path}: {exc}") from exc


def _parse_state_time(value: Any, field: str, path: Path) -> datetime:
    if not isinstance(value, str):
        raise RolloutEvidenceError(f"{path} {field} must be an ISO timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise RolloutEvidenceError(f"{path} {field} must be an ISO timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise RolloutEvidenceError(f"{path} {field} must include a UTC offset")
    return parsed


def _state_key(path: Path) -> tuple[RunKey, datetime, datetime, datetime]:
    document = _load_json(path)
    if not isinstance(document, dict):
        raise RolloutEvidenceError(f"collection state must be an object: {path}")
    expected_fields = {
        "schema_version",
        "flow",
        "repetition",
        "sequence",
        "started_at",
        "review_started_at",
        "finished_at",
        "observation_written",
    }
    if set(document) != expected_fields:
        raise RolloutEvidenceError(f"collection state has an invalid field set: {path}")
    if document.get("schema_version") != "bfcl-onboarding-ablation-collection-v1":
        raise RolloutEvidenceError(f"unsupported collection state: {path}")
    if document.get("observation_written") is not True:
        raise RolloutEvidenceError(f"collection state is not complete: {path}")
    started = _parse_state_time(document["started_at"], "started_at", path)
    review_started = _parse_state_time(
        document["review_started_at"],
        "review_started_at",
        path,
    )
    finished = _parse_state_time(document["finished_at"], "finished_at", path)
    if not started < review_started < finished:
        raise RolloutEvidenceError(
            f"collection timestamps must satisfy started < review < finished: {path}"
        )
    try:
        key = cast(
            RunKey,
            RunKey.model_validate(
                {
                    "flow": document.get("flow"),
                    "repetition": document.get("repetition"),
                    "sequence": document.get("sequence"),
                }
            ),
        )
    except ValueError as exc:
        raise RolloutEvidenceError(f"invalid collection state {path}: {exc}") from exc
    return key, started, review_started, finished


def verify_domain_bundle(
    *,
    domain_id: str,
    protocol_path: Path,
    ablation_input_path: Path,
    ablation_report_path: Path,
    observation_paths: list[Path],
    state_paths: list[Path],
    run_artifact_paths: list[Path],
    exclusions: list[RunExclusion],
    operator_identity: str,
    evaluator_pin: EvaluatorPin,
) -> VerifiedDomainBundle:
    """Prove what one domain's raw files say, before anyone has reviewed them.

    A reviewer signs the result of this function and publication recomputes it,
    so an operator cannot present one bundle for review and publish another.
    """
    source = load_ablation_input(ablation_input_path)
    report = _load_json(ablation_report_path)
    expected_report = build_ablation_report(source)
    if report != expected_report:
        raise RolloutEvidenceError("ablation report does not reproduce from its input")
    if not (
        len(observation_paths)
        == len(state_paths)
        == len(run_artifact_paths)
        == 9
    ):
        raise RolloutEvidenceError("complete domain evidence requires nine raw run bundles")

    source_by_sequence = {item.sequence: item for item in source.observations}
    ordered_exclusions = sorted(exclusions, key=lambda item: item.sequence)
    try:
        _validate_schedule(tuple(ordered_exclusions), "exclusions")
    except ValueError as exc:
        raise RolloutEvidenceError(str(exc)) from exc
    exclusion_by_sequence = {item.sequence: item for item in ordered_exclusions}
    runs: list[PublishedRunEvidence] = []
    finished_times: list[datetime] = []
    for observation_path, state_path, artifact_path in zip(
        observation_paths,
        state_paths,
        run_artifact_paths,
        strict=True,
    ):
        try:
            observation = cast(
                FlowObservation,
                FlowObservation.model_validate(_load_json(observation_path)),
            )
        except ValueError as exc:
            raise RolloutEvidenceError(
                f"invalid raw observation {observation_path}: {exc}"
            ) from exc
        if observation != source_by_sequence.get(observation.sequence):
            raise RolloutEvidenceError(
                f"raw observation is not bound to ablation input: {observation_path}"
            )
        state_key, started, review_started, finished = _state_key(state_path)
        finished_times.append(finished)
        key = (observation.flow, observation.repetition, observation.sequence)
        if key != (state_key.flow, state_key.repetition, state_key.sequence):
            raise RolloutEvidenceError("observation and collection state disagree")
        exclusion = exclusion_by_sequence[observation.sequence]
        expected_authoring = (
            (review_started - started).total_seconds() / 60
            - exclusion.excluded_authoring_minutes
        )
        expected_review = (
            (finished - review_started).total_seconds() / 60
            - exclusion.excluded_review_minutes
        )
        if (
            expected_authoring < 0
            or expected_review < 0
            or not math.isclose(
                float(observation.authoring_minutes),
                expected_authoring,
                rel_tol=0,
                abs_tol=1e-9,
            )
            or not math.isclose(
                float(observation.review_minutes),
                expected_review,
                rel_tol=0,
                abs_tol=1e-9,
            )
        ):
            raise RolloutEvidenceError(
                "observation timings do not match collection state and exclusions"
            )
        artifact_digest = digest_artifact_tree(artifact_path)
        if artifact_digest != observation.run_digest:
            raise RolloutEvidenceError("run artifact digest does not match observation")
        runs.append(
            PublishedRunEvidence(
                flow=observation.flow,
                repetition=observation.repetition,
                sequence=observation.sequence,
                observation_digest=_sha256_file(observation_path),
                collection_state_digest=_sha256_file(state_path),
                run_artifact_digest=artifact_digest,
            )
        )
    runs.sort(key=lambda item: item.sequence)
    scores_complete = all(item.evaluation_score is not None for item in source.observations)
    try:
        verify_evaluator_model_binding(
            evaluator_pin,
            evaluator_model=source.evaluator_model,
            evaluation_scores_complete=scores_complete,
        )
    except ValueError as exc:
        raise RolloutEvidenceError(str(exc)) from exc

    protocol_digest = _sha256_file(protocol_path)
    input_digest = _sha256_file(ablation_input_path)
    report_digest = str(report["report_digest"])
    try:
        bundle = cast(
            ReviewedBundle,
            ReviewedBundle.model_validate(
                {
                    "protocol_digest": protocol_digest,
                    "ablation_input_digest": input_digest,
                    "ablation_report_digest": report_digest,
                    "evaluator_pin_digest": evaluator_pin.pin_digest,
                    "exclusions_digest": exclusions_digest(
                        [item.model_dump(mode="json") for item in ordered_exclusions]
                    ),
                    "observation_digests": sorted(item.observation_digest for item in runs),
                    "run_artifact_digests": sorted(item.run_artifact_digest for item in runs),
                }
            ),
        )
    except ValueError as exc:
        raise RolloutEvidenceError(f"cannot describe the reviewed bundle: {exc}") from exc
    return VerifiedDomainBundle(
        domain_id=domain_id,
        operator_identity=operator_identity,
        experiment_id=source.experiment_id,
        domain_artifact_digest=source.domain_artifact_digest,
        evaluator_model=source.evaluator_model,
        evaluation_config_digest=source.evaluation_config_digest,
        held_out_policy_digest=source.held_out_policy_digest,
        evaluator_pin=evaluator_pin,
        runs=tuple(runs),
        exclusions=tuple(ordered_exclusions),
        evaluation_scores_complete=scores_complete,
        last_run_finished_at=max(finished_times),
        reviewed_bundle=bundle,
    )


def publish_domain_evidence(
    verified: VerifiedDomainBundle,
    *,
    review_attestation: DomainReviewAttestation,
    trusted_reviewer_keys: Mapping[str, Ed25519PublicKey],
) -> CompletedDomainEvidence:
    """Bind a verified bundle to the independent review that covers exactly it."""
    verify_domain_review_attestation(
        review_attestation,
        trusted_reviewer_keys=trusted_reviewer_keys,
        domain_id=verified.domain_id,
        experiment_id=verified.experiment_id,
        operator_identity=verified.operator_identity,
        bundle=verified.reviewed_bundle,
        last_run_finished_at=verified.last_run_finished_at,
    )
    payload: dict[str, Any] = {
        "schema_version": DOMAIN_EVIDENCE_VERSION,
        "status": "complete",
        "domain_id": verified.domain_id,
        "experiment_id": verified.experiment_id,
        "protocol_digest": verified.reviewed_bundle.protocol_digest,
        "domain_artifact_digest": verified.domain_artifact_digest,
        "ablation_input_digest": verified.reviewed_bundle.ablation_input_digest,
        "ablation_report_digest": verified.reviewed_bundle.ablation_report_digest,
        "evaluator_model": verified.evaluator_model,
        "evaluator_pin": verified.evaluator_pin.model_dump(mode="json"),
        "evaluation_config_digest": verified.evaluation_config_digest,
        "held_out_policy_digest": verified.held_out_policy_digest,
        "operator_identity": verified.operator_identity,
        "reviewer_identity": review_attestation.reviewer_identity,
        "reviewer_key_id": review_attestation.reviewer_key_id,
        "review_attestation_digest": review_attestation.attestation_digest,
        "reviewed_at": review_attestation.reviewed_at,
        "observations_are_live": True,
        "synthetic_substitution_allowed": False,
        "evaluation_scores_complete": verified.evaluation_scores_complete,
        "runs": [item.model_dump(mode="json") for item in verified.runs],
        "exclusions": [item.model_dump(mode="json") for item in verified.exclusions],
    }
    payload["evidence_digest"] = sha256_json(payload)
    try:
        return cast(
            CompletedDomainEvidence,
            CompletedDomainEvidence.model_validate(payload),
        )
    except ValueError as exc:
        raise RolloutEvidenceError(f"invalid completed domain evidence: {exc}") from exc


def build_missing_domain(
    domain_id: str,
    *,
    reason_code: str,
    detail: str,
    missing_sequences: list[int] | None = None,
) -> MissingDomainEvidence:
    selected = missing_sequences or list(range(1, 10))
    if not selected or selected != sorted(set(selected)) or any(
        sequence not in range(1, 10) for sequence in selected
    ):
        raise RolloutEvidenceError(
            "missing_sequences must be unique ascending values from 1 through 9"
        )
    runs = [
        {
            "flow": flow,
            "repetition": repetition,
            "sequence": sequence,
            "reason_code": reason_code,
        }
        for sequence, (flow, repetition) in enumerate(_SCHEDULE, start=1)
        if sequence in selected
    ]
    try:
        return cast(
            MissingDomainEvidence,
            MissingDomainEvidence.model_validate(
                {
                    "status": "missing",
                    "domain_id": domain_id,
                    "missing_runs": runs,
                    "detail": detail,
                }
            ),
        )
    except ValueError as exc:
        raise RolloutEvidenceError(f"invalid missing-domain evidence: {exc}") from exc


def _causal_blockers(domains: tuple[DomainEvidence, ...]) -> list[str]:
    blockers: list[str] = ["cross_domain:causal_design_unimplemented"]
    complete: list[CompletedDomainEvidence] = []
    for domain in domains:
        if isinstance(domain, MissingDomainEvidence):
            blockers.append(f"{domain.domain_id}:missing_runs")
        else:
            complete.append(domain)
            if not domain.evaluation_scores_complete:
                blockers.append(f"{domain.domain_id}:target_evaluation_missing")
            if isinstance(domain.evaluator_pin, UnpinnedEvaluator):
                blockers.append(f"{domain.domain_id}:target_model_pin_missing")
    pinned = {
        domain.evaluator_pin.canonical_id
        for domain in complete
        if isinstance(domain.evaluator_pin, PinnedEvaluator)
    }
    if len(complete) == 3 and len(pinned) > 1:
        blockers.append("cross_domain:target_model_mismatch")
    return sorted(blockers)


def build_rollout(
    domains: list[DomainEvidence],
    *,
    evidence_kind: Literal["descriptive", "causal"],
    decided_by: str,
    rationale: str,
) -> AblationRollout:
    ordered = tuple(sorted(domains, key=lambda item: item.domain_id))
    blockers = tuple(_causal_blockers(ordered))
    if evidence_kind == "causal" and blockers:
        raise RolloutEvidenceError(
            "causal evidence requirements are not met: " + "; ".join(blockers)
        )
    payload: dict[str, Any] = {
        "schema_version": ROLLOUT_VERSION,
        "domains": [item.model_dump(mode="json") for item in ordered],
        "decision": {
            "evidence_kind": evidence_kind,
            "decided_by": decided_by,
            "rationale": rationale,
            "blockers": list(blockers),
        },
    }
    payload["rollout_digest"] = sha256_json(payload)
    try:
        return cast(AblationRollout, AblationRollout.model_validate(payload))
    except ValueError as exc:
        raise RolloutEvidenceError(f"invalid rollout evidence: {exc}") from exc


def load_domain_evidence(path: Path) -> DomainEvidence:
    document = _load_json(path)
    try:
        if isinstance(document, dict) and document.get("status") == "complete":
            raise RolloutEvidenceError(
                "standalone completed-domain summaries are not trusted; "
                "rebuild from raw observations with verify_domain_bundle and "
                "publish_domain_evidence"
            )
        return cast(MissingDomainEvidence, MissingDomainEvidence.model_validate(document))
    except ValueError as exc:
        raise RolloutEvidenceError(f"cannot load domain evidence {path}: {exc}") from exc


def load_rollout(path: Path) -> AblationRollout:
    try:
        rollout = cast(AblationRollout, AblationRollout.model_validate(_load_json(path)))
    except ValueError as exc:
        raise RolloutEvidenceError(f"cannot load rollout evidence {path}: {exc}") from exc
    if any(isinstance(domain, CompletedDomainEvidence) for domain in rollout.domains):
        raise RolloutEvidenceError(
            "standalone rollout contains unverified completed-domain summaries; "
            "rebuild completed domains from their raw evidence bundle"
        )
    return rollout


def write_domain_evidence(evidence: DomainEvidence, path: Path) -> Path:
    return cast(Path, write_canonical_json(evidence.model_dump(mode="json"), path))


def write_rollout(rollout: AblationRollout, path: Path) -> Path:
    return cast(Path, write_canonical_json(rollout.model_dump(mode="json"), path))
