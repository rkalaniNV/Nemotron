"""Publish the SOV-862 cross-wording stability readout for a frozen release.

Read-only over the release and the evaluation artifacts. The command answers
what the frozen release can actually support about wording stability and states
plainly what it cannot, rather than approximating the paired design SOV-862
specifies with a confounded unpaired contrast.

Example:

    python -m nemotron.steps.byob.scripts.analyze_bfcl_cross_wording \
        --published releases/banking-vn-gold-v1-1392/benchmark/benchmark.parquet \
        --rendered-conversations \
            releases/banking-vn-gold-v1-1392/stage_cache/rendered_conversations.parquet \
        --primary-run gptoss-120b-8k=evaluations/gptoss-120b-8k/eval_task_results.parquet \
        --replicate-run gptoss-structural-2=evaluations/gptoss-structural-2/eval_task_results.parquet \
        --output-dir results/step4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from nemotron.steps.byob.runtime.benchmark_families.bfcl.cross_wording_analysis import (
    CrossWordingError,
    CrossWordingInputs,
    ScoredRun,
    build_cross_wording_report,
    write_cross_wording_report,
)


def _parse_run(value: str, role: str) -> ScoredRun:
    run_id, separator, raw_path = value.partition("=")
    if not separator or not run_id.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError(
            f"expected RUN_ID=PATH for a {role} run, received {value!r}"
        )
    return ScoredRun(run_id=run_id.strip(), task_results=Path(raw_path.strip()), role=role)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="analyze-bfcl-cross-wording",
        description="Measure BFCL wording stability over a frozen release.",
    )
    parser.add_argument("--published", type=Path, required=True, help="Published benchmark parquet.")
    parser.add_argument(
        "--rendered-conversations",
        type=Path,
        required=True,
        help="Stage-cache rendered conversations parquet carrying the wording provenance.",
    )
    parser.add_argument(
        "--primary-run",
        required=True,
        metavar="RUN_ID=PATH",
        help="Evaluation whose per-task verdicts carry the wording contrast.",
    )
    parser.add_argument(
        "--replicate-run",
        action="append",
        default=[],
        metavar="RUN_ID=PATH",
        help=(
            "Another scored run over the same task set, used to measure the verdict-flip "
            "floor a wording effect has to clear. Repeatable."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Report destination.")
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
        help="Two-sided confidence level for interval estimates (default: 0.95).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        inputs = CrossWordingInputs(
            published=args.published,
            rendered_conversations=args.rendered_conversations,
            primary_run=_parse_run(args.primary_run, "primary"),
            replicate_runs=tuple(_parse_run(item, "replicate") for item in args.replicate_run),
            output_dir=args.output_dir,
            confidence_level=args.confidence_level,
        )
        report = build_cross_wording_report(inputs)
        json_path, markdown_path = write_cross_wording_report(report, inputs.output_dir)
    except (CrossWordingError, argparse.ArgumentTypeError) as exc:
        print(f"cross-wording analysis failed: {exc}", file=sys.stderr)
        return 1

    conclusion = report["stability_conclusion"]
    print(
        json.dumps(
            {
                "report_hash": report["report_hash"],
                "stability_conclusion": conclusion["conclusion"],
                "paired_wording_design": report["paired_wording_design"]["status"],
                "replicate_floor": report["replicate_floor"]["status"],
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
