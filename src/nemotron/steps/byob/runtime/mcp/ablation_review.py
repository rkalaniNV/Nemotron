"""Independent review of one domain's ablation evidence, signed rather than typed.

Before this contract a domain record carried ``reviewer_identity`` as a string
the operator supplied while publishing their own runs. That string proves
nothing: the party being checked wrote it, and it stays valid no matter what the
evidence says afterwards. UA-1206 asks for an *independent* reviewer, so the
identity has to be something the operator cannot produce.

An attestation is therefore signed with a reviewer-held Ed25519 key and binds
the exact bundle that was reviewed: the protocol, the ablation input and report,
the evaluator pin, the exclusions, and the digest of all nine observations and
run trees. Verification recomputes those digests from the raw files being
published, so a signature over a different bundle is not a weaker signature but
no signature at all. The reviewer also names the operator they are independent
of, which turns "the reviewer differs from the operator" from an operator's
claim into a signed one.

Key handling mirrors :mod:`..authoring_release.revocation`: the private key
never enters an artifact, verification takes a trusted ``key_id`` map, and an
untrusted key, a bad signature, a stale digest, or a review timestamped before
the last run all fail closed.
"""

from __future__ import annotations

import json
import re
from base64 import b64decode, b64encode
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, cast

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import (
    load_pem_private_key,
    load_pem_public_key,
)
from pydantic import BaseModel, ConfigDict, StrictBool, StrictStr, model_validator

from nemotron.steps.byob.runtime.mcp.ablation import AblationError
from nemotron.steps.byob.runtime.pack_authoring.artifacts import (
    sha256_json,
    write_canonical_json,
)

DOMAIN_REVIEW_VERSION = "bfcl-onboarding-domain-review-v1"

# What an independent reviewer must have checked for a domain record to be
# publishable. Every item is a property of the evidence bundle, not an opinion
# about the result, so a reviewer who cannot confirm one must not sign at all.
REQUIRED_REVIEW_CHECKLIST = frozenset(
    {
        "protocol_followed",
        "observations_are_live",
        "no_synthetic_substitution",
        "run_artifacts_verified",
        "exclusions_justified",
        "evaluator_pin_checked",
        "independent_of_operator",
    }
)

_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTITY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@/+~-]{2,255}$")
_KEY_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class DomainReviewError(AblationError):
    """Raised when a domain's review cannot be trusted as independent."""


@dataclass(frozen=True)
class ReviewAuthority:
    """A reviewer's signing identity. The private key stays out of every artifact."""

    reviewer_identity: str
    key_id: str
    private_key: Ed25519PrivateKey = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if _IDENTITY.fullmatch(self.reviewer_identity) is None:
            raise DomainReviewError("reviewer_identity must be a stable non-secret identifier")
        if _KEY_ID.fullmatch(self.key_id) is None:
            raise DomainReviewError("reviewer key_id must be a safe identifier")

    @property
    def public_key(self) -> Ed25519PublicKey:
        return self.private_key.public_key()


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)


class ReviewedBundle(_StrictModel):
    """Every digest the reviewer's signature has to cover."""

    protocol_digest: StrictStr
    ablation_input_digest: StrictStr
    ablation_report_digest: StrictStr
    evaluator_pin_digest: StrictStr
    exclusions_digest: StrictStr
    observation_digests: tuple[StrictStr, ...]
    run_artifact_digests: tuple[StrictStr, ...]

    @model_validator(mode="after")
    def validate_bundle(self) -> ReviewedBundle:
        for field_name in (
            "protocol_digest",
            "ablation_input_digest",
            "ablation_report_digest",
            "evaluator_pin_digest",
            "exclusions_digest",
        ):
            if _DIGEST.fullmatch(str(getattr(self, field_name))) is None:
                raise ValueError(f"{field_name} must be sha256:<64 lowercase hex>")
        for field_name in ("observation_digests", "run_artifact_digests"):
            digests = cast(tuple[str, ...], getattr(self, field_name))
            if len(digests) != 9:
                raise ValueError(f"{field_name} must cover all nine runs")
            if any(_DIGEST.fullmatch(digest) is None for digest in digests):
                raise ValueError(f"{field_name} entries must be sha256:<64 lowercase hex>")
            if len(set(digests)) != 9:
                raise ValueError(f"{field_name} entries must be unique")
            if list(digests) != sorted(digests):
                raise ValueError(f"{field_name} must be sorted so the signature is order-independent")
        return self


class DomainReviewAttestation(_StrictModel):
    """One reviewer's signed statement about one domain's evidence bundle."""

    schema_version: Literal["bfcl-onboarding-domain-review-v1"]
    domain_id: StrictStr
    experiment_id: StrictStr
    reviewer_identity: StrictStr
    reviewer_key_id: StrictStr
    operator_identity: StrictStr
    independent_of_operator: StrictBool
    reviewed_at: StrictStr
    bundle: ReviewedBundle
    checklist: dict[str, StrictBool]
    note: StrictStr | None
    attestation_digest: StrictStr
    signature: StrictStr

    @model_validator(mode="after")
    def validate_attestation(self) -> DomainReviewAttestation:
        for field_name in ("domain_id", "experiment_id", "reviewer_identity", "operator_identity"):
            if _IDENTITY.fullmatch(str(getattr(self, field_name))) is None:
                raise ValueError(f"{field_name} must be a stable non-secret identifier")
        if _KEY_ID.fullmatch(self.reviewer_key_id) is None:
            raise ValueError("reviewer_key_id must be a safe identifier")
        if self.reviewer_identity.casefold() == self.operator_identity.casefold():
            raise ValueError("reviewer_identity must differ from operator_identity")
        if not self.independent_of_operator:
            raise ValueError("an attestation that declines independence cannot be published")
        _require_timestamp(self.reviewed_at, "reviewed_at")
        if set(self.checklist) != REQUIRED_REVIEW_CHECKLIST:
            raise ValueError("checklist must answer exactly the required review items")
        unchecked = sorted(name for name, value in self.checklist.items() if value is not True)
        if unchecked:
            raise ValueError(f"review checklist is incomplete: {', '.join(unchecked)}")
        if self.note is not None and not self.note.strip():
            raise ValueError("review note cannot be blank")
        if _DIGEST.fullmatch(self.attestation_digest) is None:
            raise ValueError("attestation_digest must be sha256:<64 lowercase hex>")
        _require_signature(self.signature)
        unsigned = self.model_dump(mode="json", exclude={"attestation_digest", "signature"})
        if self.attestation_digest != sha256_json(unsigned):
            raise ValueError("attestation_digest mismatch")
        return self


def exclusions_digest(exclusions: list[dict[str, Any]]) -> str:
    """Digest the exclusion set the reviewer accepted, in schedule order."""
    return sha256_json(sorted(exclusions, key=lambda item: int(item["sequence"])))


def build_domain_review_attestation(
    *,
    authority: ReviewAuthority,
    domain_id: str,
    experiment_id: str,
    operator_identity: str,
    reviewed_at: datetime,
    bundle: ReviewedBundle,
    checklist: Mapping[str, bool],
    note: str | None = None,
) -> DomainReviewAttestation:
    """Sign one reviewed bundle. The caller cannot sign on the reviewer's behalf."""
    if authority.reviewer_identity.casefold() == operator_identity.casefold():
        raise DomainReviewError("a reviewer cannot attest to a domain they operated")
    unsigned: dict[str, Any] = {
        "schema_version": DOMAIN_REVIEW_VERSION,
        "domain_id": domain_id,
        "experiment_id": experiment_id,
        "reviewer_identity": authority.reviewer_identity,
        "reviewer_key_id": authority.key_id,
        "operator_identity": operator_identity,
        "independent_of_operator": True,
        "reviewed_at": _timestamp(reviewed_at),
        "bundle": bundle.model_dump(mode="json"),
        "checklist": dict(sorted(checklist.items())),
        "note": note,
    }
    digest = sha256_json(unsigned)
    try:
        return cast(
            DomainReviewAttestation,
            DomainReviewAttestation.model_validate(
                {
                    **unsigned,
                    "attestation_digest": digest,
                    "signature": _sign(authority.private_key, digest),
                }
            ),
        )
    except ValueError as exc:
        raise DomainReviewError(f"invalid domain review attestation: {exc}") from exc


def verify_domain_review_attestation(
    attestation: DomainReviewAttestation,
    *,
    trusted_reviewer_keys: Mapping[str, Ed25519PublicKey],
    domain_id: str,
    experiment_id: str,
    operator_identity: str,
    bundle: ReviewedBundle,
    last_run_finished_at: datetime,
) -> None:
    """Refuse any attestation that does not cover this exact published bundle."""
    if attestation.domain_id != domain_id:
        raise DomainReviewError("review attestation names a different domain")
    if attestation.experiment_id != experiment_id:
        raise DomainReviewError("review attestation names a different experiment")
    if attestation.operator_identity.casefold() != operator_identity.casefold():
        raise DomainReviewError("review attestation names a different operator")
    if attestation.reviewer_identity.casefold() == operator_identity.casefold():
        raise DomainReviewError("reviewer_identity must differ from operator_identity")
    if attestation.bundle != bundle:
        raise DomainReviewError(
            "review attestation covers a different evidence bundle than the one being published"
        )
    reviewed_at = _require_timestamp(attestation.reviewed_at, "reviewed_at")
    if reviewed_at < _aware_utc(last_run_finished_at):
        raise DomainReviewError("review cannot predate the last run it claims to have reviewed")
    key = trusted_reviewer_keys.get(attestation.reviewer_key_id)
    if key is None:
        raise DomainReviewError("reviewer signing key is not trusted")
    try:
        key.verify(
            b64decode(attestation.signature, validate=True),
            attestation.attestation_digest.encode("ascii"),
        )
    except (InvalidSignature, ValueError) as exc:
        raise DomainReviewError("review signature is invalid") from exc


def load_review_authority(
    path: Path,
    *,
    reviewer_identity: str,
    key_id: str,
    password: bytes | None = None,
) -> ReviewAuthority:
    try:
        key = load_pem_private_key(path.resolve().read_bytes(), password=password)
    except (OSError, ValueError, TypeError) as exc:
        raise DomainReviewError("cannot load reviewer private key") from exc
    if not isinstance(key, Ed25519PrivateKey):
        raise DomainReviewError("reviewer private key must be Ed25519")
    return ReviewAuthority(reviewer_identity=reviewer_identity, key_id=key_id, private_key=key)


def load_trusted_reviewer_key(path: Path, *, key_id: str) -> dict[str, Ed25519PublicKey]:
    try:
        key = load_pem_public_key(path.resolve().read_bytes())
    except (OSError, ValueError, TypeError) as exc:
        raise DomainReviewError("cannot load trusted reviewer public key") from exc
    if not isinstance(key, Ed25519PublicKey):
        raise DomainReviewError("trusted reviewer key must be Ed25519")
    if _KEY_ID.fullmatch(key_id) is None:
        raise DomainReviewError("reviewer key_id must be a safe identifier")
    return {key_id: key}


def write_domain_review_attestation(attestation: DomainReviewAttestation, path: Path) -> Path:
    return cast(Path, write_canonical_json(attestation.model_dump(mode="json"), path))


def load_domain_review_attestation(path: Path) -> DomainReviewAttestation:
    try:
        document = json.loads(
            path.resolve().read_text(encoding="utf-8"),
            object_pairs_hook=_unique_pairs,
            parse_constant=_reject_constant,
        )
        return cast(DomainReviewAttestation, DomainReviewAttestation.model_validate(document))
    except DomainReviewError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise DomainReviewError(f"cannot load review attestation {path}: {exc}") from exc


def _sign(key: Ed25519PrivateKey, digest: str) -> str:
    return b64encode(key.sign(digest.encode("ascii"))).decode("ascii")


def _require_signature(value: str) -> None:
    try:
        decoded = b64decode(value, validate=True)
    except ValueError as exc:
        raise ValueError("review signature must be canonical base64") from exc
    if len(decoded) != 64 or b64encode(decoded).decode("ascii") != value:
        raise ValueError("review signature must encode 64 bytes")


def _timestamp(value: datetime) -> str:
    return _aware_utc(value).isoformat()


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise DomainReviewError("review timestamps must include a UTC offset")
    return value.astimezone(timezone.utc)


def _require_timestamp(value: str, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"review {label} must be ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"review {label} must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    document: dict[str, Any] = {}
    for key, value in pairs:
        if key in document:
            raise DomainReviewError(f"review attestation repeats JSON key {key!r}")
        document[key] = value
    return document


def _reject_constant(token: str) -> None:
    raise DomainReviewError(f"review attestation contains non-finite constant {token}")
