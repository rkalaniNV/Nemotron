#!/usr/bin/env python3
"""A7 — independently audit whether the A0-A6 evidence supports publication.

A7 is deliberately artifact-only: it does not rerun BFCL, call an LLM, or silently
promote model judgements to human ground truth.

    PYTHONPATH=src:. python3 bfcl_ablation/run_a7.py
    PYTHONPATH=src:. python3 bfcl_ablation/run_a7.py \
      --emit-label-template bfcl_ablation/results/A7/human_labels.template.yaml
    PYTHONPATH=src:. python3 bfcl_ablation/run_a7.py --labels completed_labels.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bfcl_ablation import common  # noqa: E402
from bfcl_ablation.quality_gate.artifacts import load_artifacts, load_thresholds  # noqa: E402
from bfcl_ablation.quality_gate.checks import run_quality_gate  # noqa: E402
from bfcl_ablation.quality_gate.labels import (  # noqa: E402
    build_review_expectations,
    build_review_queue,
    label_coverage,
    load_review_file,
    merge_review_labels,
    write_review_file,
)
from bfcl_ablation.quality_gate.report import render  # noqa: E402

DEFAULT_THRESHOLDS = Path(__file__).resolve().parent / "quality_gate" / "defaults.yaml"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_outputs(
    *,
    results_root: Path,
    metrics: dict[str, Any],
    checks: list[dict[str, Any]],
    coverage: dict[str, Any],
    report: str,
) -> dict[str, Path]:
    if results_root.resolve() == common.RESULTS.resolve():
        paths = {
            "metrics": common.dump_result("a7_metrics.json", metrics),
            "checks": common.dump_result("a7_checks.json", checks),
            "label_coverage": common.dump_result("a7_label_coverage.json", coverage),
            "report": common.result_path("a7_report.md"),
        }
        paths["report"].write_text(report, encoding="utf-8")
        return paths

    output = results_root / "A7"
    paths = {
        "metrics": output / "metrics.json",
        "checks": output / "checks.json",
        "label_coverage": output / "label_coverage.json",
        "report": output / "report.md",
    }
    _write_json(paths["metrics"], metrics)
    _write_json(paths["checks"], checks)
    _write_json(paths["label_coverage"], coverage)
    paths["report"].write_text(report, encoding="utf-8")
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=common.RESULTS,
        help="directory containing A0/ through A6/ (default: bfcl_ablation/results)",
    )
    parser.add_argument(
        "--thresholds",
        type=Path,
        default=DEFAULT_THRESHOLDS,
        help="versioned release-policy YAML",
    )
    parser.add_argument("--labels", type=Path, help="completed human-review YAML")
    parser.add_argument(
        "--emit-label-template",
        type=Path,
        help="write a deterministic review queue with empty labels",
    )
    parser.add_argument(
        "--sample-per-template",
        type=int,
        default=3,
        help="A2 prevalence pairs per template in a generated review queue",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero unless integrity and release_readiness both PASS",
    )
    args = parser.parse_args(argv)
    if args.sample_per_template < 1:
        parser.error("--sample-per-template must be at least 1")

    try:
        policy, threshold_provenance = load_thresholds(args.thresholds)
        artifacts, inventory = load_artifacts(args.results_dir)
        queue = build_review_queue(
            artifacts,
            sample_per_template=args.sample_per_template,
        )
        expectations = build_review_expectations(
            artifacts,
            sample_per_template=args.sample_per_template,
        )
        if args.emit_label_template is not None:
            write_review_file(args.emit_label_template, queue)
            print(f"[a7] wrote review template {common.rel(args.emit_label_template)}", flush=True)

        label_issues: list[str] = []
        labels_path: str | None = None
        review = queue
        if args.labels is not None:
            supplied = load_review_file(args.labels)
            review, label_issues = merge_review_labels(queue, supplied)
            labels_path = common.rel(args.labels)
        else:
            label_issues.append("no human label file supplied")
        coverage = label_coverage(review, policy, label_issues, expectations)
    except (OSError, ValueError, ValidationError) as exc:
        print(f"[a7] invalid input: {exc}", file=sys.stderr)
        return 2

    metrics, check_models = run_quality_gate(
        artifacts=artifacts,
        inventory=inventory,
        policy=policy,
        threshold_provenance=threshold_provenance,
        label_coverage=coverage,
        labels_path=labels_path,
    )
    checks = [check.model_dump(mode="json") for check in check_models]
    text = render(metrics, check_models, coverage)
    paths = _write_outputs(
        results_root=args.results_dir,
        metrics=metrics,
        checks=checks,
        coverage=coverage,
        report=text,
    )
    for name, path in paths.items():
        print(f"[a7] wrote {name}: {common.rel(path)}", flush=True)
    print(
        f"[a7] integrity={metrics['rollup']['integrity']} "
        f"study={metrics['rollup']['study_validity']} "
        f"release={metrics['rollup']['release_readiness']} "
        f"decision={metrics['publication_decision']}",
        flush=True,
    )

    if args.strict and (
        metrics["rollup"]["integrity"] != "PASS"
        or metrics["rollup"]["release_readiness"] != "PASS"
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
