#!/usr/bin/env python3
"""Collect and assemble real BFCL onboarding ablation observations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from nemotron.steps.byob.runtime.mcp.ablation import AblationError
from nemotron.steps.byob.runtime.mcp.ablation_collection import (
    assemble_ablation_input,
    begin_collection,
    finish_collection,
    mark_review_started,
    stop_collection,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    begin = commands.add_parser("begin")
    begin.add_argument("--state", type=Path, required=True)
    begin.add_argument(
        "--flow",
        choices=("manual", "llm_backend", "llm_mcp"),
        required=True,
    )
    begin.add_argument("--repetition", type=int, required=True)
    begin.add_argument("--sequence", type=int, required=True)

    review = commands.add_parser("review")
    review.add_argument("--state", type=Path, required=True)

    stop = commands.add_parser("stop")
    stop.add_argument("--state", type=Path, required=True)

    finish = commands.add_parser("finish")
    finish.add_argument("--state", type=Path, required=True)
    finish.add_argument("--output", type=Path, required=True)
    finish.add_argument("--run-artifact", type=Path, required=True)
    finish.add_argument("--user-authored-fields", type=int, required=True)
    finish.add_argument("--validation-pass-rate", type=float, required=True)
    finish.add_argument("--tool-coverage", type=float, required=True)
    finish.add_argument("--replay-stability", type=float, required=True)
    finish.add_argument("--benchmark-rows", type=int, required=True)
    finish.add_argument("--excluded-authoring-minutes", type=float, default=0)
    finish.add_argument("--excluded-review-minutes", type=float, default=0)
    finish.add_argument("--evaluation-score", type=float)
    finish.add_argument("--evaluation-score-stderr", type=float)

    assemble = commands.add_parser("assemble")
    assemble.add_argument(
        "--observation",
        type=Path,
        action="append",
        required=True,
        help="Repeat exactly nine times.",
    )
    assemble.add_argument("--output", type=Path, required=True)
    assemble.add_argument("--experiment-id", required=True)
    assemble.add_argument("--domain-artifact-digest", required=True)
    assemble.add_argument("--evaluator-model", required=True)
    assemble.add_argument("--evaluation-config-digest", required=True)
    assemble.add_argument("--held-out-policy-digest", required=True)
    return parser


def _run(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "begin":
        path = begin_collection(
            args.state,
            flow=args.flow,
            repetition=args.repetition,
            sequence=args.sequence,
        )
        return {"status": "authoring", "state": str(path.resolve())}
    if args.command == "review":
        path = mark_review_started(args.state)
        return {"status": "review", "state": str(path.resolve())}
    if args.command == "stop":
        path = stop_collection(args.state)
        return {"status": "stopped", "state": str(path.resolve())}
    if args.command == "finish":
        observation = finish_collection(
            args.state,
            args.output,
            run_artifact=args.run_artifact,
            user_authored_fields=args.user_authored_fields,
            validation_pass_rate=args.validation_pass_rate,
            tool_coverage=args.tool_coverage,
            replay_stability=args.replay_stability,
            benchmark_rows=args.benchmark_rows,
            excluded_authoring_minutes=args.excluded_authoring_minutes,
            excluded_review_minutes=args.excluded_review_minutes,
            evaluation_score=args.evaluation_score,
            evaluation_score_stderr=args.evaluation_score_stderr,
        )
        return {
            "status": "complete",
            "observation": str(args.output.resolve()),
            "run_digest": observation.run_digest,
        }
    source = assemble_ablation_input(
        args.observation,
        args.output,
        experiment_id=args.experiment_id,
        domain_artifact_digest=args.domain_artifact_digest,
        evaluator_model=args.evaluator_model,
        evaluation_config_digest=args.evaluation_config_digest,
        held_out_policy_digest=args.held_out_policy_digest,
    )
    return {
        "status": "assembled",
        "input": str(args.output.resolve()),
        "observations": len(source.observations),
    }


def main() -> None:
    try:
        result = _run(_parser().parse_args())
    except (AblationError, ValueError) as exc:
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
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
