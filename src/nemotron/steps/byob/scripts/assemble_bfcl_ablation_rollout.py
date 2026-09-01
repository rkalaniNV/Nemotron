#!/usr/bin/env python3
"""Assemble three verified domain records into a BFCL rollout decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from nemotron.steps.byob.runtime.mcp.ablation_domain_bundle import load_domain_bundle
from nemotron.steps.byob.runtime.mcp.ablation_review import (
    load_domain_review_attestation,
    load_trusted_reviewer_key,
)
from nemotron.steps.byob.runtime.mcp.ablation_rollout import (
    DomainEvidence,
    RolloutEvidenceError,
    build_rollout,
    load_domain_evidence,
    publish_domain_evidence,
    verify_domain_bundle,
    write_rollout,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--domain-evidence",
        action="append",
        type=Path,
        default=[],
        help="A missing-domain record. Repeat once per domain that has no publishable evidence.",
    )
    parser.add_argument(
        "--domain-bundle",
        action="append",
        type=Path,
        default=[],
        metavar="MANIFEST",
        help=(
            "A complete domain, rebuilt from its raw bundle and the independent review the "
            "manifest names. Repeat once per completed domain. Together with --domain-evidence "
            "this must describe exactly three domains."
        ),
    )
    parser.add_argument(
        "--trusted-reviewer-key",
        action="append",
        default=[],
        metavar="KEY_ID=PUBLIC_KEY_PATH",
        help=(
            "A reviewer key the publishing authority trusts. Repeat once per key. This is "
            "supplied here, never read from an operator's bundle manifest."
        ),
    )
    parser.add_argument(
        "--evidence-kind",
        choices=("descriptive", "causal"),
        required=True,
    )
    parser.add_argument("--decided-by", required=True)
    parser.add_argument("--rationale", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _trusted_reviewer_keys(declarations: list[str]) -> dict[str, Ed25519PublicKey]:
    trusted: dict[str, Ed25519PublicKey] = {}
    for declaration in declarations:
        key_id, separator, path = declaration.partition("=")
        if not separator or not key_id.strip() or not path.strip():
            raise RolloutEvidenceError("--trusted-reviewer-key must be KEY_ID=PUBLIC_KEY_PATH")
        if key_id in trusted:
            raise RolloutEvidenceError(f"reviewer key id {key_id!r} was declared twice")
        trusted.update(load_trusted_reviewer_key(Path(path), key_id=key_id))
    return trusted


def _completed_domain(
    manifest_path: Path,
    trusted_reviewer_keys: dict[str, Ed25519PublicKey],
) -> DomainEvidence:
    if not trusted_reviewer_keys:
        raise RolloutEvidenceError(
            "a completed domain requires at least one --trusted-reviewer-key"
        )
    loaded = load_domain_bundle(manifest_path)
    if loaded.review_attestation_path is None:
        raise RolloutEvidenceError(
            f"bundle {manifest_path} does not name a review attestation, so its domain "
            "cannot be published as complete"
        )
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
    return publish_domain_evidence(
        verified,
        review_attestation=load_domain_review_attestation(loaded.review_attestation_path),
        trusted_reviewer_keys=trusted_reviewer_keys,
    )


def main() -> None:
    args = _parser().parse_args()
    try:
        trusted = _trusted_reviewer_keys(args.trusted_reviewer_key)
        domains: list[DomainEvidence] = [
            load_domain_evidence(path) for path in args.domain_evidence
        ]
        domains.extend(
            _completed_domain(manifest, trusted) for manifest in args.domain_bundle
        )
        if len(domains) != 3:
            raise RolloutEvidenceError(
                "--domain-evidence and --domain-bundle must describe exactly three domains"
            )
        rollout = build_rollout(
            domains,
            evidence_kind=args.evidence_kind,
            decided_by=args.decided_by,
            rationale=args.rationale,
        )
        write_rollout(rollout, args.output)
    except (OSError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "status": "fail",
                    "error_type": type(exc).__name__,
                    "reason": str(exc),
                },
                sort_keys=True,
            )
        )
        raise SystemExit(1) from exc
    print(
        json.dumps(
            {
                "status": "written",
                "output": str(args.output.resolve()),
                "evidence_kind": rollout.decision.evidence_kind,
                "blockers": list(rollout.decision.blockers),
                "domains": {
                    domain.domain_id: domain.status for domain in rollout.domains
                },
                "rollout_digest": rollout.rollout_digest,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
