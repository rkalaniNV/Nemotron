#!/usr/bin/env python3
"""Independently verify and sign one BFCL domain's ablation evidence bundle.

An independent reviewer runs this, not the operator who produced the runs. The
``verify`` command re-derives every digest from the raw files and prints them
without signing anything, so a reviewer can inspect the bundle first. ``sign``
repeats that verification and signs the result, which means a reviewer cannot
sign a bundle they have not just verified, and a bundle that changed afterwards
no longer matches the signature.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
)

from nemotron.steps.byob.runtime.mcp.ablation import AblationError
from nemotron.steps.byob.runtime.mcp.ablation_domain_bundle import load_domain_bundle
from nemotron.steps.byob.runtime.mcp.ablation_review import (
    REQUIRED_REVIEW_CHECKLIST,
    DomainReviewError,
    build_domain_review_attestation,
    load_domain_review_attestation,
    load_review_authority,
    load_trusted_reviewer_key,
    verify_domain_review_attestation,
    write_domain_review_attestation,
)
from nemotron.steps.byob.runtime.mcp.ablation_rollout import verify_domain_bundle


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    keygen = commands.add_parser("keygen", help="Create a reviewer Ed25519 key pair.")
    keygen.add_argument("--private-key", type=Path, required=True)
    keygen.add_argument("--public-key", type=Path, required=True)

    verify = commands.add_parser("verify", help="Re-derive a bundle's digests without signing.")
    verify.add_argument("--bundle", type=Path, required=True)

    sign = commands.add_parser("sign", help="Verify a bundle and sign the reviewed digests.")
    sign.add_argument("--bundle", type=Path, required=True)
    sign.add_argument("--private-key", type=Path, required=True)
    sign.add_argument("--reviewer-identity", required=True)
    sign.add_argument("--reviewer-key-id", required=True)
    sign.add_argument("--output", type=Path, required=True)
    sign.add_argument("--reviewed-at", help="ISO-8601 with a UTC offset. Defaults to now.")
    sign.add_argument("--note")
    sign.add_argument(
        "--confirm",
        action="append",
        default=[],
        metavar="CHECKLIST_ITEM",
        help=(
            "Repeat once per checklist item. Required items: "
            + ", ".join(sorted(REQUIRED_REVIEW_CHECKLIST))
        ),
    )

    check = commands.add_parser("check", help="Verify a signed attestation against a bundle.")
    check.add_argument("--bundle", type=Path, required=True)
    check.add_argument("--attestation", type=Path, required=True)
    check.add_argument("--public-key", type=Path, required=True)
    check.add_argument("--reviewer-key-id", required=True)
    return parser


def _keygen(args: argparse.Namespace) -> dict[str, Any]:
    private = Ed25519PrivateKey.generate()
    private_path = args.private_key.resolve()
    public_path = args.public_key.resolve()
    for path in (private_path, public_path):
        if path.exists():
            raise DomainReviewError(f"refusing to overwrite an existing key: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(private_path, os.O_CREAT | os.O_WRONLY | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(
            private.private_bytes(
                encoding=Encoding.PEM,
                format=PrivateFormat.PKCS8,
                encryption_algorithm=NoEncryption(),
            )
        )
    public_path.write_bytes(
        private.public_key().public_bytes(
            encoding=Encoding.PEM,
            format=PublicFormat.SubjectPublicKeyInfo,
        )
    )
    return {
        "status": "generated",
        "private_key": str(private_path),
        "public_key": str(public_path),
        "reminder": "the private key belongs to the reviewer and must never enter evidence",
    }


def _verified_bundle(bundle_path: Path) -> Any:
    loaded = load_domain_bundle(bundle_path)
    verified = verify_domain_bundle(
        domain_id=loaded.domain_id,
        protocol_path=loaded.protocol_path,
        ablation_input_path=loaded.ablation_input_path,
        ablation_report_path=loaded.ablation_report_path,
        observation_paths=loaded.observation_paths,
        state_paths=loaded.state_paths,
        run_artifact_paths=loaded.run_artifact_paths,
        exclusions=loaded.exclusions,
        operator_identity=loaded.operator_identity,
        evaluator_pin=loaded.evaluator_pin,
    )
    return verified


def _verify(args: argparse.Namespace) -> dict[str, Any]:
    verified = _verified_bundle(args.bundle)
    return {
        "status": "verified",
        "domain_id": verified.domain_id,
        "experiment_id": verified.experiment_id,
        "operator_identity": verified.operator_identity,
        "evaluator_pin_status": verified.evaluator_pin.status,
        "evaluator_model": verified.evaluator_model,
        "evaluation_scores_complete": verified.evaluation_scores_complete,
        "last_run_finished_at": verified.last_run_finished_at.isoformat(),
        "reviewed_bundle": verified.reviewed_bundle.model_dump(mode="json"),
    }


def _sign(args: argparse.Namespace) -> dict[str, Any]:
    confirmed = set(args.confirm)
    missing = sorted(REQUIRED_REVIEW_CHECKLIST - confirmed)
    if missing:
        raise DomainReviewError(
            "cannot sign without confirming every checklist item; missing: " + ", ".join(missing)
        )
    unknown = sorted(confirmed - REQUIRED_REVIEW_CHECKLIST)
    if unknown:
        raise DomainReviewError("unknown checklist items: " + ", ".join(unknown))
    verified = _verified_bundle(args.bundle)
    reviewed_at = (
        datetime.fromisoformat(args.reviewed_at)
        if args.reviewed_at
        else datetime.now(timezone.utc)
    )
    authority = load_review_authority(
        args.private_key,
        reviewer_identity=args.reviewer_identity,
        key_id=args.reviewer_key_id,
    )
    attestation = build_domain_review_attestation(
        authority=authority,
        domain_id=verified.domain_id,
        experiment_id=verified.experiment_id,
        operator_identity=verified.operator_identity,
        reviewed_at=reviewed_at,
        bundle=verified.reviewed_bundle,
        checklist=dict.fromkeys(sorted(REQUIRED_REVIEW_CHECKLIST), True),
        note=args.note,
    )
    if args.output.exists():
        raise DomainReviewError(f"refusing to overwrite an existing attestation: {args.output}")
    write_domain_review_attestation(attestation, args.output)
    return {
        "status": "signed",
        "attestation": str(args.output.resolve()),
        "attestation_digest": attestation.attestation_digest,
        "domain_id": attestation.domain_id,
        "reviewer_identity": attestation.reviewer_identity,
        "reviewed_at": attestation.reviewed_at,
    }


def _check(args: argparse.Namespace) -> dict[str, Any]:
    verified = _verified_bundle(args.bundle)
    attestation = load_domain_review_attestation(args.attestation)
    verify_domain_review_attestation(
        attestation,
        trusted_reviewer_keys=load_trusted_reviewer_key(
            args.public_key,
            key_id=args.reviewer_key_id,
        ),
        domain_id=verified.domain_id,
        experiment_id=verified.experiment_id,
        operator_identity=verified.operator_identity,
        bundle=verified.reviewed_bundle,
        last_run_finished_at=verified.last_run_finished_at,
    )
    return {
        "status": "accepted",
        "domain_id": attestation.domain_id,
        "reviewer_identity": attestation.reviewer_identity,
        "attestation_digest": attestation.attestation_digest,
    }


def main() -> None:
    args = _parser().parse_args()
    handlers = {"keygen": _keygen, "verify": _verify, "sign": _sign, "check": _check}
    try:
        result = handlers[args.command](args)
    except (AblationError, OSError, ValueError) as exc:
        print(
            json.dumps(
                {"status": "fail", "error_type": type(exc).__name__, "reason": str(exc)},
                sort_keys=True,
            )
        )
        raise SystemExit(1) from exc
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
