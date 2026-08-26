#!/usr/bin/env python3
"""Freeze an approved canonical MCP pack into a read-only BFCL release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nemotron.steps.byob.runtime.mcp.release.freeze import (
    FreezeError,
    FreezeInputs,
    freeze_canonical_pack,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "pack",
        "mcp-config",
        "evidence",
        "intake-provenance",
        "gateway-attestation",
        "draft-provenance",
        "validation-report",
        "review-packet",
        "review-approval",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    inputs = FreezeInputs(
        pack_root=args.pack,
        mcp_config_path=args.mcp_config,
        evidence_path=args.evidence,
        intake_provenance_path=args.intake_provenance,
        gateway_attestation_path=args.gateway_attestation,
        draft_provenance_path=args.draft_provenance,
        validation_report_path=args.validation_report,
        review_packet_path=args.review_packet,
        review_approval_path=args.review_approval,
    )
    try:
        release = freeze_canonical_pack(inputs, args.output)
    except (FreezeError, OSError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "status": "fail",
                    "error_type": type(exc).__name__,
                    "reason": str(exc),
                },
                indent=2,
                sort_keys=True,
            )
        )
        raise SystemExit(1) from exc
    print(
        json.dumps(
            {
                "status": "frozen",
                "release": str(release.root),
                "pack": str(release.pack_root),
                "frozen_pack_fingerprint": release.pack_fingerprint,
                "freeze_manifest_digest": release.manifest["record_digest"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
