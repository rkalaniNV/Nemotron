#!/usr/bin/env python3
# /// script
# [tool.runspec]
# schema = "1"
# docs = "https://raw.githubusercontent.com/NVIDIA-NeMo/Nemotron/main/docs/runspec/v1/spec.md"
# name = "nemotron/steps/translation"
# image = "nvcr.io/nvidia/nemo:25.11.nemotron"
# setup = "NeMo Curator translation dependencies are pre-installed."
#
# [tool.runspec.run]
# launch = "direct"
#
# [tool.runspec.config]
# dir = "../assets"
# default = "default"
# format = "omegaconf"
#
# [tool.runspec.resources]
# nodes = 1
# gpus_per_node = 0
# ///

"""Run the corpus translation skill."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from nemotron.kit.train_script import (
    apply_hydra_overrides,
    load_omegaconf_yaml,
    parse_config_and_overrides,
)
from nemotron.steps.translation.scripts.driver import translate_data

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "assets" / "default.yaml"


def main() -> None:
    """Load config, apply CLI overrides, and run corpus translation."""
    try:
        config_path, cli_overrides = parse_config_and_overrides(default_config=DEFAULT_CONFIG_PATH)
        config = load_omegaconf_yaml(config_path)
        config = apply_hydra_overrides(config, cli_overrides)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        sys.exit(1)

    output_path = translate_data(config)
    logger.info("Translation complete. Output: %s", output_path)


if __name__ == "__main__":
    main()
