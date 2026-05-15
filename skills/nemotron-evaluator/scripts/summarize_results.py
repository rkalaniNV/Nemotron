#!/usr/bin/env python3
"""Summarize NeMo Evaluator result artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def _load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_result(path: Path) -> Any:
    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return _load_yaml(path)


def _flatten_numbers(obj: Any, prefix: str = "") -> list[tuple[str, int | float]]:
    rows: list[tuple[str, int | float]] = []
    if isinstance(obj, dict):
        if "value" in obj and isinstance(obj["value"], (int, float)):
            rows.append((prefix.rstrip("."), obj["value"]))
        for key, value in obj.items():
            if key == "value":
                continue
            rows.extend(_flatten_numbers(value, f"{prefix}{key}."))
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            rows.extend(_flatten_numbers(value, f"{prefix}{index}."))
    return rows


def _find_results(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(
        {
            *path.rglob("artifacts/results.yml"),
            *path.rglob("results.yml"),
        }
    )


def summarize(path: Path) -> None:
    result_files = _find_results(path)
    if not result_files:
        raise SystemExit(f"No results.yml or artifacts/results.yml found under {path}")

    for result_file in result_files:
        print(f"\n{result_file}")
        data = _load_result(result_file) or {}
        status = data.get("status") or data.get("status_code")
        if status is not None:
            print(f"status: {status}")

        rows = _flatten_numbers(data)
        printed = 0
        for name, value in rows:
            if any(token in name.lower() for token in ("score", "metric", "acc", "correct", "rouge", "chrf", "exact")):
                print(f"{name}: {value}")
                printed += 1
                if printed >= 20:
                    break

        metrics_file = result_file.with_name("eval_factory_metrics.json")
        if metrics_file.exists():
            with metrics_file.open("r", encoding="utf-8") as f:
                metrics = json.load(f)
            response_stats = metrics.get("response_stats", {})
            count = (
                metrics.get("count")
                or metrics.get("request_count")
                or response_stats.get("count")
            )
            successful = response_stats.get("successful_count")
            latency = (
                metrics.get("avg_latency_ms")
                or metrics.get("latency_ms_avg")
                or response_stats.get("avg_latency_ms")
            )
            if count is not None:
                print(f"requests: {count}")
            if successful is not None:
                print(f"successful_requests: {successful}")
            if latency is not None:
                print(f"avg_latency_ms: {latency}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Result directory or artifacts/results.yml")
    args = parser.parse_args()
    summarize(args.path)


if __name__ == "__main__":
    main()
