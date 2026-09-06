"""Derive ablation-ladder measurements from one BFCL evaluation run.

Read-only. Emits the task success, cost and latency measurements plus the
failure-code distribution that SOV-866 requires the ablation summary to speak
to, with the hash of every artifact it read, so the numbers entering a
content-addressed ladder are reproducible rather than transcribed.

Example:

    python -m nemotron.steps.byob.scripts.extract_bfcl_eval_measurements \
        --run-id gptoss-120b-8k \
        --evaluation-dir release-candidate/sov867-clean-52907cc/evaluations/gptoss-120b-8k \
        --output-dir results/step4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval_measurements import (
    EvalMeasurementError,
    EvalMeasurementInputs,
    build_eval_measurements,
    write_eval_measurements,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="extract-bfcl-eval-measurements",
        description="Derive ladder measurements from a BFCL evaluation run.",
    )
    parser.add_argument("--run-id", required=True, help="Identifier recorded in the report.")
    parser.add_argument(
        "--evaluation-dir",
        type=Path,
        required=True,
        help="Directory holding eval_task_results.parquet, candidate_io_cache.jsonl "
        "and eval_report.json.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Report destination.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    evaluation_dir: Path = args.evaluation_dir
    try:
        inputs = EvalMeasurementInputs(
            run_id=args.run_id,
            task_results=evaluation_dir / "eval_task_results.parquet",
            candidate_io_cache=evaluation_dir / "candidate_io_cache.jsonl",
            eval_report=evaluation_dir / "eval_report.json",
            output_dir=args.output_dir,
        )
        report = build_eval_measurements(inputs)
        json_path, markdown_path = write_eval_measurements(report, inputs.output_dir)
    except EvalMeasurementError as exc:
        print(f"eval measurement extraction failed: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "report_hash": report["report_hash"],
                "run_id": report["run_id"],
                "scored_tasks": report["task_set"]["task_count"],
                "reconciled": report["reconciliation"]["agrees"],
                "json_report": str(json_path),
                "markdown_report": str(markdown_path),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
