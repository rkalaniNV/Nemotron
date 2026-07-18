"""Data Designer preparation and generation orchestration."""

from __future__ import annotations

import json
from pathlib import Path

from .checkpoint import completed_query_ids, load_records, verify_fingerprint
from .config import PipelineConfig
from .generator_config import LongContextEpisodeConfig
from .schemas import EpisodeSeed
from .seeds import iter_jsonl


def _dd_models(cfg: PipelineConfig):
    import data_designer.config as dd

    return [
        dd.ModelConfig(
            alias=model.alias,
            model=model.model,
            provider=model.provider,
            skip_health_check=model.skip_health_check,
            inference_parameters=dd.ChatCompletionInferenceParams(
                **model.inference_parameters
            ),
        )
        for model in cfg.models
    ]


def _dd_providers(cfg: PipelineConfig):
    import data_designer.config as dd

    return [
        dd.ModelProvider(name=p.name, endpoint=p.endpoint, api_key=p.api_key_env)
        for p in cfg.providers
    ]


def _pending_seed_file(cfg: PipelineConfig) -> tuple[Path, int]:
    checkpoint = cfg.resolve(cfg.paths.checkpoint)
    records = load_records(checkpoint)
    verify_fingerprint(records, cfg.fingerprint())
    completed = completed_query_ids(
        records,
        retry_failed=cfg.run.retry_failed,
        retry_quarantine=cfg.run.retry_quarantine,
    )
    pending = []
    for row in iter_jsonl(cfg.resolve(cfg.paths.enriched_seeds)):
        seed = EpisodeSeed.model_validate_json(row["episode_input"])
        if seed.query_id not in completed:
            pending.append(row)
    if cfg.run.num_records:
        pending = pending[: cfg.run.num_records]
    path = checkpoint.parent / f"_pending_{cfg.fingerprint()[:12]}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in pending:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path, len(pending)


def generate(cfg: PipelineConfig) -> int:
    import data_designer.config as dd
    from data_designer.interface import DataDesigner

    seed_path, count = _pending_seed_file(cfg)
    if count == 0:
        return 0
    builder = dd.DataDesignerConfigBuilder(model_configs=_dd_models(cfg))
    builder.with_seed_dataset(
        dd.LocalFileSeedSource(path=str(seed_path)),
        sampling_strategy=dd.SamplingStrategy.ORDERED,
    )
    pipeline_payload = cfg.model_dump(mode="json", exclude={"config_dir"})
    builder.add_column(
        LongContextEpisodeConfig(
            name="conversation",
            episode_input_column="episode_input",
            pipeline=pipeline_payload,
            checkpoint_path=str(cfg.resolve(cfg.paths.checkpoint)),
            run_id=f"run-{cfg.fingerprint()[:12]}",
        )
    )
    designer = (
        DataDesigner(model_providers=_dd_providers(cfg))
        if cfg.providers
        else DataDesigner()
    )
    if cfg.run.mode == "preview":
        designer.preview(builder, num_records=count)
    else:
        designer.create(builder, num_records=count)
    return count
