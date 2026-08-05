#!/usr/bin/env python3
"""Normalize and validate an oracle pack through a BFCL config."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import yaml

from nemotron.steps.byob.runtime.benchmark_families.bfcl.pipeline import prepare_bfcl


def _run(config_path: Path, output_dir: Path) -> dict:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")
    data["output_dir"] = str(output_dir)
    resolved_config = output_dir / "resolved-config.yaml"
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_config.write_text(yaml.safe_dump(data), encoding="utf-8")
    report_path = prepare_bfcl(resolved_config)
    return json.loads(report_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Path to a BFCL YAML config")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Artifact directory; defaults to a temporary directory",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        with tempfile.TemporaryDirectory() as tmp:
            report = _run(args.config.resolve(), Path(tmp))
    else:
        report = _run(args.config.resolve(), args.output_dir.resolve())

    print(json.dumps(report, indent=2, sort_keys=True))
    if not report.get("gold_eligible"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
