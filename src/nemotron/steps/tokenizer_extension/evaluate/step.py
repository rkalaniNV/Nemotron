#!/usr/bin/env python3
# /// script
# [tool.runspec]
# schema = "1"
# name = "steps/tokenizer_extension/evaluate"
#
# [tool.runspec.run]
# launch = "python"
#
# [tool.runspec.config]
# dir = "./config"
# default = "default"
# format = "yaml"
#
# [tool.runspec.resources]
# nodes = 1
# gpus_per_node = 0
# ///
"""Tokenizer fertility evaluation — CPU-only.

Reports corpus-level fertility (sum(tokens)/sum(words)) for a tokenizer on an
eval corpus (HF dataset or local). This is a TOKENIZER-level metric only; model /
downstream evaluation post-CPT is handled by the existing steps/eval catalog.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
# package root too: languages.py / script_ranges.py are shared across steps
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fertility import run_fertility  # noqa: E402


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", "-c", default=str(Path(__file__).parent / "config" / "default.yaml"))
    args, _ = ap.parse_known_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    run_fertility(cfg)


if __name__ == "__main__":
    main()
