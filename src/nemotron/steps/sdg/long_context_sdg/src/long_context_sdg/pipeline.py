"""Data Designer preparation, native resume, and generation orchestration."""

from __future__ import annotations

from .config import PipelineConfig
from .generator_config import LongContextEpisodeConfig
from .records import write_records
from .schemas import CanonicalRecord, EpisodeSeed
from .seeds import iter_jsonl


def _dd_models(cfg: PipelineConfig):
    import data_designer.config as dd

    return [
        dd.ModelConfig(
            alias=model.alias,
            model=model.model,
            provider=model.provider,
            skip_health_check=model.skip_health_check,
            inference_parameters=dd.ChatCompletionInferenceParams(**model.inference_parameters),
        )
        for model in cfg.models
    ]


def _dd_providers(cfg: PipelineConfig):
    import data_designer.config as dd

    return [
        dd.ModelProvider(
            name=provider.name,
            endpoint=provider.endpoint,
            api_key=provider.api_key_env,
        )
        for provider in cfg.providers
    ]


def _seed_order(cfg: PipelineConfig) -> list[str]:
    order = []
    for row in iter_jsonl(cfg.resolve(cfg.paths.enriched_seeds)):
        seed = EpisodeSeed.model_validate_json(row["episode_input"])
        order.append(seed.query_id)
    if len(order) != len(set(order)):
        raise ValueError("enriched seed file contains duplicate query IDs")
    return order


def _materialize_generated(results, cfg: PipelineConfig, expected_order: list[str]) -> None:
    frame = results.load_dataset()
    if "canonical_record" not in frame:
        raise ValueError("Data Designer result does not contain canonical_record")
    records = [CanonicalRecord.model_validate_json(value) for value in frame["canonical_record"]]
    by_query = {record.query_id: record for record in records}
    if len(by_query) != len(records):
        raise ValueError("Data Designer result contains duplicate canonical query IDs")
    missing = [query_id for query_id in expected_order if query_id not in by_query]
    unexpected = sorted(set(by_query) - set(expected_order))
    if missing or unexpected:
        raise ValueError(f"Data Designer result/query mismatch: missing={missing}, unexpected={unexpected}")
    write_records(cfg.resolve(cfg.paths.generated), (by_query[query_id] for query_id in expected_order))


def generate(cfg: PipelineConfig) -> int:
    import data_designer.config as dd
    from data_designer.interface import DataDesigner

    order = _seed_order(cfg)
    count = cfg.run.num_records or len(order)
    if count > len(order):
        raise ValueError(f"run.num_records={count} exceeds the {len(order)} prepared seeds")
    order = order[:count]
    if count == 0:
        return 0

    builder = dd.DataDesignerConfigBuilder(model_configs=_dd_models(cfg))
    builder.with_seed_dataset(
        dd.LocalFileSeedSource(path=str(cfg.resolve(cfg.paths.enriched_seeds))),
        sampling_strategy=dd.SamplingStrategy.ORDERED,
    )
    builder.add_column(
        LongContextEpisodeConfig(
            name="conversation",
            episode_input_column="episode_input",
            pipeline=cfg.generation_payload(),
            run_id=f"run-{cfg.fingerprint()[:12]}",
        )
    )
    providers = _dd_providers(cfg)
    designer = DataDesigner(
        artifact_path=cfg.resolve(cfg.paths.artifacts),
        model_providers=providers or None,
    )
    if cfg.run.mode == "preview":
        designer.preview(builder, num_records=count)
        return count

    results = designer.create(
        builder,
        num_records=count,
        dataset_name=cfg.run.dataset_name,
        resume=dd.ResumeMode(cfg.run.resume),
    )
    _materialize_generated(results, cfg, order)
    return count
