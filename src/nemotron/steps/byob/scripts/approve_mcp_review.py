#!/usr/bin/env python3
"""Record a named reviewer's explicit approval of one MCP review packet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nemotron.steps.byob.runtime.mcp.release.review import (
    REQUIRED_CHECKLIST,
    ReviewError,
    build_review_approval,
    load_review_packet,
    write_review_approval,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--approved-by", required=True)
    parser.add_argument("--reviewed-at", required=True)
    parser.add_argument("--acknowledge-risk", action="append", default=[])
    parser.add_argument("--note")
    parser.add_argument("--output", type=Path, required=True)
    # Separate required flags make each semantic decision visible in shell history. There is no
    # blanket --accept-all because adding a future checklist item must require a new decision.
    for name in sorted(REQUIRED_CHECKLIST):
        parser.add_argument(
            f"--accept-{name.replace('_', '-')}",
            action="store_true",
            required=True,
        )
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        packet = load_review_packet(args.packet)
        checklist = {
            name: bool(getattr(args, f"accept_{name}")) for name in REQUIRED_CHECKLIST
        }
        approval = build_review_approval(
            packet,
            approved_by=args.approved_by,
            reviewed_at=args.reviewed_at,
            checklist=checklist,
            acknowledged_risks=args.acknowledge_risk,
            note=args.note,
        )
        path = write_review_approval(approval, args.output)
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
                "status": "approved",
                "approval_digest": approval.digest,
                "review_packet_digest": packet.digest,
                "output": str(path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
