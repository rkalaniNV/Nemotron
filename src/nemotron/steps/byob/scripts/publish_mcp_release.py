#!/usr/bin/env python3
"""Revalidate a frozen MCP pack and publish through BFCL's existing stage path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nemotron.steps.byob.runtime.mcp.release.handoff import (
    HandoffError,
    handoff_frozen_release,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = handoff_frozen_release(args.release, args.config)
    except (HandoffError, OSError, ValueError) as exc:
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
                "status": "published",
                "validation_report": str(result.validation_report_path),
                "benchmark": str(result.benchmark_path),
                "benchmark_raw": str(result.raw_benchmark_path),
                "run_manifest": str(result.run_manifest_path),
                "frozen_pack_fingerprint": result.release.pack_fingerprint,
                "run_id": result.run_manifest.get("run_id"),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
