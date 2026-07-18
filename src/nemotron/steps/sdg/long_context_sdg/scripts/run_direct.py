"""Run generation directly against OpenAI-compatible models and an HTTP retrieval API."""

from __future__ import annotations

import argparse
import json
import sys
from itertools import islice
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from long_context_sdg.checkpoint import (
    append_record,
    completed_query_ids,
    load_records,
    verify_fingerprint,
)
from long_context_sdg.config import PipelineConfig, load_config
from long_context_sdg.evaluation import evaluate_checkpoint
from long_context_sdg.executors.base import ExecutionServices
from long_context_sdg.exporters import export_records
from long_context_sdg.models import direct_models
from long_context_sdg.retrieval import RetrieverClient
from long_context_sdg.runtime import EpisodeRunner
from long_context_sdg.seeds import enrich_seed, iter_jsonl
from long_context_sdg.tool_registry import ToolRegistry


def _configured_outputs(cfg: PipelineConfig) -> list[Path]:
    output_dir = cfg.resolve(cfg.paths.output_dir)
    return [
        cfg.resolve(cfg.paths.checkpoint),
        cfg.resolve(cfg.paths.canonical),
        cfg.resolve(cfg.paths.export),
        output_dir / "summary.json",
        *(
            output_dir / f"{status}.jsonl"
            for status in ("accepted", "rejected", "quarantine", "generation_failed")
        ),
    ]


def _reset_outputs(cfg: PipelineConfig) -> None:
    for path in _configured_outputs(cfg):
        path.unlink(missing_ok=True)


def _record_limit(cli_value: int | None, cfg: PipelineConfig) -> int | None:
    value = cli_value if cli_value is not None else cfg.run.num_records
    return value or None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="pipeline YAML configuration")
    parser.add_argument(
        "--num-records",
        type=int,
        help="override run.num_records; omit to use the configuration",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="delete only the output files named by this configuration before running",
    )
    args = parser.parse_args()
    if args.num_records is not None and args.num_records < 1:
        parser.error("--num-records must be at least 1")

    config_path = Path(args.config)
    cfg = load_config(
        config_path if config_path.is_absolute() else PROJECT_ROOT / config_path
    )
    if args.fresh:
        _reset_outputs(cfg)

    checkpoint = cfg.resolve(cfg.paths.checkpoint)
    existing = load_records(checkpoint)
    verify_fingerprint(existing, cfg.fingerprint())
    completed = completed_query_ids(
        existing,
        retry_failed=cfg.run.retry_failed,
        retry_quarantine=cfg.run.retry_quarantine,
    )
    limit = _record_limit(args.num_records, cfg)
    raw_seeds = iter_jsonl(cfg.resolve(cfg.paths.seeds))
    if limit is not None:
        raw_seeds = islice(raw_seeds, limit)
    seeds = [enrich_seed(raw, cfg) for raw in raw_seeds]
    query_ids = [seed.query_id for seed in seeds]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("seed input contains duplicate query_id values")

    models = direct_models(cfg)
    retriever = None
    generated = []
    try:
        retriever = RetrieverClient(cfg.retriever)
        services = ExecutionServices(
            retriever=retriever,
            models=models,
        )
        registry = ToolRegistry(cfg.tools, services)
        runner = EpisodeRunner(cfg)
        for index, seed in enumerate(seeds, 1):
            if seed.query_id in completed:
                print(f"[{index}/{len(seeds)}] SKIP {seed.query_id}", flush=True)
                continue
            print(f"[{index}/{len(seeds)}] GENERATE {seed.query_id}", flush=True)
            record = runner.run(
                models,
                seed,
                registry,
                run_id=f"direct-{cfg.fingerprint()[:12]}",
            )
            append_record(checkpoint, record)
            generated.append(record)
            print(
                f"[{index}/{len(seeds)}] {record.status.upper()} "
                f"messages={len(record.messages)} "
                f"retrievals={len(record.retrieval_transcript)}",
                flush=True,
            )

        summary = evaluate_checkpoint(cfg, judge_models=models)
        exported = export_records(
            cfg.resolve(cfg.paths.canonical),
            cfg.resolve(cfg.paths.export),
            output_format=cfg.export.format,
        )
    finally:
        if retriever is not None:
            retriever.close()
        for model in models.values():
            model.close()

    print(
        json.dumps(
            {
                "generated_this_run": len(generated),
                "summary": summary,
                "exported": exported,
                "checkpoint": str(checkpoint),
                "canonical": str(cfg.resolve(cfg.paths.canonical)),
                "export": str(cfg.resolve(cfg.paths.export)),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
