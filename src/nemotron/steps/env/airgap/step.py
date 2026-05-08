#!/usr/bin/env python3
# /// script
# [tool.runspec]
# schema = "1"
# name = "steps/env/airgap"
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

# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Generate an airgap workflow lock from a step config.

This local stage is a config-driven wrapper around ``nemotron step airgap
lock-workflow``. Use the CLI subcommands for fetch/build/verify release work;
use this runner when the workflow should appear as a normal cataloged step.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml
from omegaconf import OmegaConf

from nemo_runspec.config.loader import load_config
from nemotron.kit.train_script import parse_config_and_overrides
from nemotron.steps.airgap import (
    AirgapCompiler,
    AirgapIssue,
    AirgapTarget,
    build_delivery_plan,
    lock_to_dict,
    verify_lock,
    write_lock,
)

DEFAULT_CONFIG = Path(__file__).parent / "config" / "default.yaml"


def main() -> None:
    config_path, _ = parse_config_and_overrides(default_config=DEFAULT_CONFIG)
    cfg = OmegaConf.to_container(load_config(Path(config_path)), resolve=True)
    if not isinstance(cfg, dict):
        raise TypeError("Airgap config must be a mapping")

    targets = _string_list(cfg.get("targets"))
    if not targets:
        raise ValueError(
            "Set targets to one or more step_id:config entries, for example "
            "`targets=['prep/sft_packing:tiny','sft/megatron_bridge:tiny']`."
        )

    repo_root = Path(str(cfg.get("repo_root") or Path.cwd())).expanduser().resolve()
    env_file = _optional_path(cfg.get("env_file"))
    output_path = Path(str(cfg.get("output_path") or "airgap.lock.yaml")).expanduser()
    bundle_dir = _optional_path(cfg.get("bundle_dir"))

    compiler = AirgapCompiler(repo_root=repo_root)
    lock = compiler.compile_many(
        [_parse_target(target) for target in targets],
        workflow_name=str(cfg.get("name") or "workflow"),
        profiles=_string_list(cfg.get("profiles")),
        env_file=env_file,
    )
    write_lock(lock, output_path)
    lock_data = lock_to_dict(lock)
    print(f"Wrote {output_path} for {len(targets)} airgap target(s)")

    if bool(cfg.get("show_plan", True)):
        print("\nAirgap plan")
        print(yaml.safe_dump(build_delivery_plan(lock_data), sort_keys=False).rstrip())

    if bool(cfg.get("verify", True)):
        issues = verify_lock(lock_data, strict=bool(cfg.get("strict", False)), bundle_dir=bundle_dir)
        _print_issues(issues)
        if any(issue.severity == "error" for issue in issues):
            raise SystemExit(1)


def _parse_target(value: str) -> AirgapTarget:
    step_id, sep, config_name = value.partition(":")
    step_id = step_id.strip()
    config_name = config_name.strip() if sep else None
    if not step_id:
        raise ValueError(f"Invalid airgap target {value!r}")
    return AirgapTarget(step_id=step_id, config_name=config_name or None)


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list | tuple):
        return [str(item) for item in value if str(item).strip()]
    raise TypeError(f"Expected string or list, got {type(value).__name__}")


def _optional_path(value: Any) -> Path | None:
    if value in (None, ""):
        return None
    return Path(str(value)).expanduser()


def _print_issues(issues: list[AirgapIssue]) -> None:
    if not issues:
        print("\nAirgap verification: no issues")
        return
    print("\nAirgap verification")
    for issue in issues:
        location = f" [{issue.source}]" if issue.source else ""
        print(f"- {issue.severity}: {issue.code}{location}: {issue.message}")


if __name__ == "__main__":
    sys.exit(main())
