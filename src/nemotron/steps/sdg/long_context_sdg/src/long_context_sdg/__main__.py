"""Command-line entry point for the isolated long-context SDG project."""

from __future__ import annotations

import argparse
import json

from .config import load_config
from .evaluation import evaluate_checkpoint
from .exporters import export_records
from .models import offline_judge_models
from .pipeline import generate
from .seeds import prepare_seed_file


def main() -> None:
    parser = argparse.ArgumentParser(prog="long-context-sdg")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "generate", "evaluate", "export"):
        p = sub.add_parser(command)
        p.add_argument("--config", required=True)
        if command == "evaluate":
            p.add_argument(
                "--rejudge",
                action="store_true",
                help="rejudge every deterministically valid record",
            )
            p.add_argument(
                "--no-network",
                action="store_true",
                help="leave pending judge records quarantined",
            )
    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.command == "prepare":
        print(json.dumps({"prepared": prepare_seed_file(cfg)}))
    elif args.command == "generate":
        print(json.dumps({"submitted": generate(cfg)}))
    elif args.command == "evaluate":
        models = None
        if cfg.judge.enabled and not args.no_network:
            try:
                models = offline_judge_models(cfg)
            except Exception as exc:
                print(f"judge unavailable; pending records remain quarantined: {exc}")
        try:
            print(
                json.dumps(
                    evaluate_checkpoint(cfg, judge_models=models, rejudge=args.rejudge),
                    indent=2,
                )
            )
        finally:
            for model in (models or {}).values():
                close = getattr(model, "close", None)
                if callable(close):
                    close()
    else:
        source = cfg.resolve(cfg.paths.canonical)
        count = export_records(
            source, cfg.resolve(cfg.paths.export), output_format=cfg.export.format
        )
        print(json.dumps({"exported": count, "format": cfg.export.format}))


if __name__ == "__main__":
    main()
