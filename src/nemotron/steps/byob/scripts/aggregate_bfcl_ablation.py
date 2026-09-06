#!/usr/bin/env python3
"""Aggregate the BFCL ablation ladder into a release recommendation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from nemotron.steps.byob.runtime.benchmark_families.bfcl.ablation_aggregation import (
    AblationAggregationError,
    AggregationInputs,
    build_ablation_summary,
    write_ablation_summary,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        report = build_ablation_summary(
            AggregationInputs(ablation_input=args.input, output_dir=args.output_dir)
        )
        json_path, markdown_path = write_ablation_summary(report, args.output_dir)
    except (AblationAggregationError, OSError, ValueError) as exc:
        print(f"ablation_aggregation_failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    summary = report["summary"]
    print(
        json.dumps(
            {
                "release_readiness": summary["release_readiness"],
                "report_hash": report["report_hash"],
                "json_report": str(json_path),
                "markdown_report": str(markdown_path),
                "recommendations": summary["recommendations"],
                "unmeasured_families": report["coverage"]["unmeasured_families"],
            },
            sort_keys=True,
        )
    )
    if summary["release_readiness"] == "blocked":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
