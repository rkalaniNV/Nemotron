"""Command-line entry point for the isolated long-context SDG project."""

from __future__ import annotations

import argparse
import json

from .config import load_config
from .conversation_generation import (
    generate_conversations,
    prepare_conversation_seeds,
)
from .evaluation import evaluate_generated
from .exporters import export_records
from .models import evaluation_judge_models


def main() -> None:
    parser = argparse.ArgumentParser(prog="long-context-sdg")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("synthesize", "prepare", "generate", "evaluate", "export"):
        p = sub.add_parser(command)
        p.add_argument("--config", required=True)
        if command == "synthesize":
            p.add_argument(
                "--force",
                action="store_true",
                help="replace an existing seed file produced by another synthesis run",
            )
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
    if args.command == "synthesize":
        from .query_generation.config import load_query_generation_config
        from .query_generation.pipeline import synthesize_queries

        cfg = load_query_generation_config(args.config)
        print(json.dumps(synthesize_queries(cfg, force=args.force), indent=2))
        return

    cfg = load_config(args.config)
    if args.command == "prepare":
        print(json.dumps({"prepared": prepare_conversation_seeds(cfg)}))
    elif args.command == "generate":
        print(json.dumps({"submitted": generate_conversations(cfg)}))
    elif args.command == "evaluate":
        models = None
        if cfg.judge.enabled and not args.no_network:
            try:
                models = evaluation_judge_models(cfg)
            except Exception as exc:
                print(f"judge unavailable; pending records remain quarantined: {exc}")
        try:
            print(
                json.dumps(
                    evaluate_generated(cfg, judge_models=models, rejudge=args.rejudge),
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
        count = export_records(source, cfg.resolve(cfg.paths.export), output_format=cfg.export.format)
        print(json.dumps({"exported": count, "format": cfg.export.format}))


if __name__ == "__main__":
    main()
