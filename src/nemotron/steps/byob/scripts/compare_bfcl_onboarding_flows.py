#!/usr/bin/env python3
"""Build a deterministic manual vs LLM/backend vs LLM/MCP ablation report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nemotron.steps.byob.runtime.mcp.ablation import (
    AblationError,
    build_ablation_report,
    load_ablation_input,
    write_ablation_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = build_ablation_report(load_ablation_input(args.input))
        output = write_ablation_report(report, args.output)
    except (AblationError, OSError, ValueError) as exc:
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
                "status": "pass",
                "output": str(output),
                "report_digest": report["report_digest"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
