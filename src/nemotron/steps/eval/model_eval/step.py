#!/usr/bin/env python3
# /// script
# [tool.runspec]
# schema = "1"
# name = "steps/eval/model_eval"
#
# [tool.runspec.run]
# launch = "python"
#
# [tool.runspec.config]
# dir = "./config"
# default = "default"
# format = "omegaconf"
#
# [tool.runspec.resources]
# nodes = 1
# gpus_per_node = 0
# ///

# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run NeMo Evaluator directly or through NeMo Evaluator Launcher."""

from __future__ import annotations

import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import requests
import yaml
from nemo_evaluator.api import evaluate
from nemo_evaluator.api.api_dataclasses import (
    ApiEndpoint,
    ConfigParams,
    EvaluationConfig,
    EvaluationTarget,
)
from omegaconf import OmegaConf

from nemotron.kit.train_script import (
    apply_hydra_overrides,
    load_omegaconf_yaml,
    parse_config_and_overrides,
)

DEFAULT_CONFIG = Path(__file__).parent / "config" / "default.yaml"

_STEP_ONLY_KEYS = {
    "runner",
    "output_dir",
    "benchmarks",
    "params",
    "endpoint_check",
    "dry_run",
    "task_filters",
    "sovereign",
}


def _prepend_python_bin_to_path() -> None:
    """Expose console scripts installed beside the active Python interpreter."""
    python_bin = str(Path(sys.executable).resolve().parent)
    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    if python_bin not in path_parts:
        os.environ["PATH"] = os.pathsep.join([python_bin, *path_parts])


def _load_config() -> dict[str, Any]:
    config_path, overrides = parse_config_and_overrides(default_config=DEFAULT_CONFIG)
    cfg = apply_hydra_overrides(load_omegaconf_yaml(config_path), overrides)
    return OmegaConf.to_container(cfg, resolve=True)


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_dicts(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _endpoint_config(cfg: dict[str, Any]) -> dict[str, Any]:
    if "target" in cfg and cfg["target"].get("api_endpoint"):
        endpoint = copy.deepcopy(cfg["target"]["api_endpoint"])
        endpoint.setdefault("type", endpoint.pop("endpoint_type", None))
        return endpoint

    deployment = cfg.get("deployment", {})
    return {
        "model_id": deployment["model_id"],
        "url": deployment["url"],
        "type": deployment.get("endpoint_type", deployment.get("type", "completions")),
        "api_key_name": deployment.get("api_key_name"),
        "stream": deployment.get("stream"),
        "adapter_config": deployment.get("adapter_config"),
    }


def _api_endpoint(cfg: dict[str, Any]) -> ApiEndpoint:
    endpoint = {k: v for k, v in _endpoint_config(cfg).items() if v is not None}
    return ApiEndpoint(**endpoint)


def _check_endpoint(endpoint: ApiEndpoint, settings: dict[str, Any]) -> None:
    if not settings.get("enabled", True):
        return

    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if endpoint.api_key_name:
        api_key = os.getenv(endpoint.api_key_name)
        if api_key is None:
            raise RuntimeError(f"API key env var '{endpoint.api_key_name}' is not set.")
        headers["Authorization"] = f"Bearer {api_key}"

    payload: dict[str, Any] = {"model": endpoint.model_id, "max_tokens": 1}
    if endpoint.type == "chat":
        payload["messages"] = [{"role": "user", "content": "hello"}]
    elif endpoint.type in ("completions", "completions_logprob"):
        payload["prompt"] = "hello"
    else:
        raise ValueError(f"Unsupported endpoint type: {endpoint.type}")

    retries = int(settings.get("max_retries", 3))
    interval = float(settings.get("retry_interval", 2))
    last_status: int | None = None
    last_body = ""
    for _ in range(retries):
        response = requests.post(
            str(endpoint.url),
            headers=headers,
            json=payload,
            timeout=int(settings.get("request_timeout", 60)),
        )
        last_status = response.status_code
        last_body = response.text[:500]
        if response.status_code == 200:
            return
        time.sleep(interval)

    raise RuntimeError(
        "Endpoint readiness check failed after "
        f"{retries} attempts; last status={last_status}; response={last_body!r}."
    )


def _benchmark_name(item: str | dict[str, Any]) -> str:
    if isinstance(item, str):
        return item
    return str(item.get("name") or item.get("type"))


def _benchmark_params(global_params: dict[str, Any], item: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(item, str):
        return copy.deepcopy(global_params)
    return _merge_dicts(global_params, item.get("params", {}))


def _run_direct(cfg: dict[str, Any]) -> None:
    endpoint = _api_endpoint(cfg)
    target = EvaluationTarget(api_endpoint=endpoint)
    _check_endpoint(endpoint, cfg.get("endpoint_check", {}))

    output_dir = Path(cfg.get("output_dir", "./results"))
    global_params = cfg.get("params", {})
    for benchmark in cfg.get("benchmarks", []):
        name = _benchmark_name(benchmark)
        params = ConfigParams(**_benchmark_params(global_params, benchmark))
        evaluate(
            target_cfg=target,
            eval_cfg=EvaluationConfig(
                type=name,
                params=params,
                output_dir=str(output_dir / name),
            ),
        )
    _print_results_summary(output_dir)


def _task_from_benchmark(item: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(item, str):
        return {"name": item}

    allowed = {
        "name",
        "container",
        "endpoint_type",
        "env_vars",
        "nemo_evaluator_config",
        "dataset_dir",
        "dataset_mount_path",
        "pre_cmd",
        "post_cmd",
    }
    return {key: copy.deepcopy(value) for key, value in item.items() if key in allowed}


def _build_launcher_config(cfg: dict[str, Any]) -> dict[str, Any]:
    launcher_cfg = copy.deepcopy(cfg.get("launcher_config", cfg))
    for key in _STEP_ONLY_KEYS:
        launcher_cfg.pop(key, None)

    endpoint = _endpoint_config(cfg)
    if endpoint:
        launcher_cfg.setdefault("target", {}).setdefault("api_endpoint", {})
        launcher_endpoint = launcher_cfg["target"]["api_endpoint"]
        for key, value in endpoint.items():
            if value is not None:
                launcher_endpoint.setdefault(key, value)

    deployment = cfg.get("deployment", {})
    if deployment.get("url") or deployment.get("model_id"):
        launcher_cfg["deployment"] = {"type": "none"}
    else:
        launcher_cfg.setdefault("deployment", {"type": "none"})

    launcher_cfg.setdefault("execution", {})
    launcher_cfg["execution"].setdefault("type", "local")
    launcher_cfg["execution"].setdefault("mode", "sequential")
    launcher_cfg["execution"].setdefault("output_dir", cfg.get("output_dir", "./results"))

    launcher_cfg.setdefault("evaluation", {})
    evaluation = launcher_cfg["evaluation"]
    evaluation.setdefault("nemo_evaluator_config", {})
    evaluation["nemo_evaluator_config"].setdefault("config", {})
    evaluation["nemo_evaluator_config"]["config"].setdefault(
        "params", copy.deepcopy(cfg.get("params", {}))
    )
    if "tasks" not in evaluation:
        evaluation["tasks"] = [_task_from_benchmark(item) for item in cfg.get("benchmarks", [])]

    return launcher_cfg


def _run_launcher(cfg: dict[str, Any]) -> None:
    from nemo_evaluator_launcher.api.functional import run_eval

    launcher_cfg = _build_launcher_config(cfg)
    invocation_id = run_eval(
        OmegaConf.create(launcher_cfg),
        dry_run=bool(cfg.get("dry_run", False)),
        tasks=cfg.get("task_filters"),
    )
    if invocation_id:
        print(f"launcher_invocation_id: {invocation_id}")
    if not cfg.get("dry_run", False):
        _print_results_summary(Path(launcher_cfg["execution"]["output_dir"]))


def _flatten_numbers(obj: Any, prefix: str = "") -> list[tuple[str, int | float]]:
    rows: list[tuple[str, int | float]] = []
    if isinstance(obj, dict):
        if "value" in obj and isinstance(obj["value"], (int, float)):
            rows.append((prefix.rstrip("."), obj["value"]))
        for key, value in obj.items():
            if key != "value":
                rows.extend(_flatten_numbers(value, f"{prefix}{key}."))
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            rows.extend(_flatten_numbers(value, f"{prefix}{index}."))
    return rows


def _print_results_summary(output_dir: Path) -> None:
    result_files = sorted(
        {
            *output_dir.rglob("artifacts/results.yml"),
            *output_dir.rglob("results.yml"),
        }
    )
    if not result_files:
        return

    for result_file in result_files[-10:]:
        print(f"\nresults: {result_file}")
        with result_file.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        for name, value in _flatten_numbers(data):
            lower = name.lower()
            if any(token in lower for token in ("score", "metric", "acc", "correct", "rouge", "chrf", "exact")):
                print(f"{name}: {value}")

        metrics_file = result_file.with_name("eval_factory_metrics.json")
        if metrics_file.exists():
            with metrics_file.open("r", encoding="utf-8") as f:
                metrics = json.load(f)
            response_stats = metrics.get("response_stats", {})
            summary = {
                "count": metrics.get("count") or metrics.get("request_count") or response_stats.get("count"),
                "successful_count": response_stats.get("successful_count"),
                "avg_latency_ms": metrics.get("avg_latency_ms")
                or metrics.get("latency_ms_avg")
                or response_stats.get("avg_latency_ms"),
            }
            for key, value in summary.items():
                if value is not None:
                    print(f"{key}: {value}")


def main() -> None:
    _prepend_python_bin_to_path()
    cfg = _load_config()
    runner = cfg.get("runner", "direct")
    if runner == "launcher":
        _run_launcher(cfg)
    elif runner == "direct":
        _run_direct(cfg)
    else:
        raise ValueError("runner must be either 'direct' or 'launcher'")


if __name__ == "__main__":
    main()
