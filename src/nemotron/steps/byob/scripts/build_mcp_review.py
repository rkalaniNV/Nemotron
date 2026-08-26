#!/usr/bin/env python3
"""Build the deterministic domain-review packet for an MCP-backed BFCL pack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nemotron.steps.byob.runtime.mcp.release.review import (
    ReviewError,
    build_review_packet,
    write_review_packet,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--intake-provenance", type=Path, required=True)
    parser.add_argument("--gateway-attestation", type=Path, required=True)
    parser.add_argument("--draft-provenance", type=Path, required=True)
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--mcp-config", type=Path, required=True)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--held-out", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        packet = build_review_packet(
            args.evidence,
            args.intake_provenance,
            args.gateway_attestation,
            args.draft_provenance,
            args.validation_report,
            args.mcp_config,
            args.pack,
            held_out_path=args.held_out,
        )
        path = write_review_packet(packet, args.output)
    except (ReviewError, OSError, ValueError) as exc:
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
                "status": packet.document["status"],
                "packet_digest": packet.digest,
                "output": str(path),
                "blockers": packet.document["blockers"],
                "risk_ids": [
                    item["id"] for item in packet.document["metadata_risks"]
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if packet.document["status"] != "ready_for_approval":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
