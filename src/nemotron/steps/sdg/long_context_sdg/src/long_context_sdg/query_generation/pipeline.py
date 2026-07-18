"""Data Designer orchestration, resume, reporting, and atomic seed publication."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .candidates import prepare_candidates
from .checkpoint import latest_by_query, load_records, verify_fingerprint
from .config import QueryGenerationPipelineConfig
from .generator_config import SyntheticQueryConfig
from .personas import persona_column_name, persona_key
from .schemas import QueryCandidate, QuerySynthesisRecord
from .validation import query_similarity


def _dd_models(cfg: QueryGenerationPipelineConfig):
    import data_designer.config as dd

    aliases = {
        cfg.query_generation.generator_alias,
        cfg.query_generation.judge_alias,
    }
    return [
        dd.ModelConfig(
            alias=model.alias,
            model=model.model,
            provider=model.provider,
            skip_health_check=model.skip_health_check,
            inference_parameters=dd.ChatCompletionInferenceParams(**model.inference_parameters),
        )
        for model in cfg.models
        if model.alias in aliases
    ]


def _dd_providers(cfg: QueryGenerationPipelineConfig):
    import data_designer.config as dd

    needed = {
        model.provider
        for model in cfg.models
        if model.alias
        in {
            cfg.query_generation.generator_alias,
            cfg.query_generation.judge_alias,
        }
    }
    return [
        dd.ModelProvider(name=provider.name, endpoint=provider.endpoint, api_key=provider.api_key_env)
        for provider in cfg.providers
        if provider.name in needed
    ]


def _persona_columns(cfg: QueryGenerationPipelineConfig) -> dict[str, str]:
    return {
        persona_key(index, locale): persona_column_name(persona_key(index, locale))
        for index, locale in enumerate(cfg.query_generation.persona_locales)
    }


def _add_persona_samplers(builder, cfg: QueryGenerationPipelineConfig, dd) -> dict[str, str]:
    columns = _persona_columns(cfg)
    for index, locale in enumerate(cfg.query_generation.persona_locales):
        key = persona_key(index, locale)
        params = {
            "locale": locale.locale,
            "with_synthetic_personas": True,
            "age_range": locale.age_range,
        }
        for name in ("sex", "city", "select_field_values"):
            value = getattr(locale, name)
            if value is not None:
                params[name] = value
        builder.add_column(
            dd.SamplerColumnConfig(
                name=columns[key],
                drop=True,
                sampler_type=dd.SamplerType.PERSON,
                params=dd.PersonSamplerParams(**params),
            )
        )
    return columns


def _pending_file(
    cfg: QueryGenerationPipelineConfig,
    candidates: list[QueryCandidate],
    records: list[QuerySynthesisRecord],
) -> tuple[Path, int]:
    generation = cfg.query_generation
    latest = latest_by_query(records)
    pending = [
        candidate
        for candidate in candidates
        if not (
            candidate.query_id in latest
            and (
                latest[candidate.query_id].status == "accepted"
                or latest[candidate.query_id].attempt >= generation.max_attempts
            )
        )
    ]
    checkpoint = cfg.resolve(cfg.paths.checkpoint)
    path = checkpoint.parent / f"_pending_queries_{candidates[0].synthesis_fingerprint[:12]}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for candidate in pending:
            handle.write(
                json.dumps(
                    {"candidate_input": candidate.model_dump_json()},
                    ensure_ascii=False,
                )
                + "\n"
            )
    temporary.replace(path)
    return path, len(pending)


def _counter(records: list[QuerySynthesisRecord], field: str) -> dict[str, int]:
    return dict(Counter(str(getattr(record, field)) for record in records))


def _duplicate_pairs(records: list[QuerySynthesisRecord], threshold: float) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[QuerySynthesisRecord]] = defaultdict(list)
    for record in records:
        groups[(record.taxonomy_id, record.language)].append(record)
    duplicates = []
    for group in groups.values():
        for left_index, left in enumerate(group):
            if left.draft is None:
                continue
            for right in group[left_index + 1 :]:
                if right.draft is None:
                    continue
                similarity = query_similarity(left.draft, right.draft)
                if similarity >= threshold:
                    duplicates.append(
                        {
                            "left": left.query_id,
                            "right": right.query_id,
                            "similarity": round(similarity, 4),
                        }
                    )
    return duplicates


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _finalize(
    cfg: QueryGenerationPipelineConfig,
    candidates: list[QueryCandidate],
    fingerprint: str,
    *,
    force: bool,
) -> dict[str, Any]:
    generation = cfg.query_generation
    checkpoint = cfg.resolve(cfg.paths.checkpoint)
    records = load_records(checkpoint)
    verify_fingerprint(records, fingerprint)
    latest = latest_by_query(records)
    terminal = [latest[item.query_id] for item in candidates if item.query_id in latest]
    accepted = sorted(
        (record for record in terminal if record.status == "accepted"),
        key=lambda record: record.candidate_index,
    )
    duplicates = _duplicate_pairs(accepted, generation.evidence.duplicate_similarity)
    report = {
        "synthesis_fingerprint": fingerprint,
        "requested": len(candidates),
        "terminal": len(terminal),
        "accepted": len(accepted),
        "status_counts": dict(Counter(record.status for record in terminal)),
        "attempt_counts": dict(Counter(str(record.attempt) for record in terminal)),
        "taxonomy_counts": _counter(accepted, "taxonomy_id"),
        "archetype_counts": _counter(accepted, "archetype"),
        "language_counts": _counter(accepted, "language"),
        "persona_mode_counts": _counter(accepted, "persona_mode"),
        "duplicates": duplicates,
        "rejection_reasons": dict(
            Counter(error for record in terminal if record.status != "accepted" for error in record.validation_errors)
        ),
    }
    _write_report(cfg.resolve(cfg.paths.report), report)
    if len(accepted) != len(candidates):
        raise RuntimeError(
            f"query synthesis accepted {len(accepted)} of {len(candidates)} candidates; "
            f"see {cfg.resolve(cfg.paths.report)}"
        )
    if duplicates:
        raise RuntimeError(
            f"query synthesis found {len(duplicates)} near-duplicate pair(s); see {cfg.resolve(cfg.paths.report)}"
        )
    if any(record.seed is None for record in accepted):
        raise RuntimeError("accepted query checkpoint record is missing its seed")
    content = "".join(json.dumps(record.seed, ensure_ascii=False) + "\n" for record in accepted)
    destination = cfg.resolve(cfg.paths.seeds)
    if destination.exists():
        existing = destination.read_text(encoding="utf-8")
        if existing == content:
            return report
        if not force:
            raise FileExistsError(f"refusing to replace existing seed file {destination}; use --force")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(destination)
    return report


def synthesize_queries(cfg: QueryGenerationPipelineConfig, *, force: bool = False) -> dict[str, Any]:
    if cfg.run.mode != "create":
        raise ValueError("query synthesis requires run.mode: create")
    candidates, fingerprint = prepare_candidates(cfg)
    checkpoint = cfg.resolve(cfg.paths.checkpoint)
    records = load_records(checkpoint)
    verify_fingerprint(records, fingerprint)
    pending_path, pending_count = _pending_file(cfg, candidates, records)
    if pending_count:
        import data_designer.config as dd
        from data_designer.interface import DataDesigner

        builder = dd.DataDesignerConfigBuilder(model_configs=_dd_models(cfg))
        builder.with_seed_dataset(
            dd.LocalFileSeedSource(path=str(pending_path)),
            sampling_strategy=dd.SamplingStrategy.ORDERED,
        )
        persona_columns = _add_persona_samplers(builder, cfg, dd)
        builder.add_column(
            SyntheticQueryConfig(
                name="synthetic_query",
                model_alias=cfg.query_generation.generator_alias,
                candidate_input_column="candidate_input",
                persona_columns=persona_columns,
                pipeline=cfg.model_dump(mode="json", exclude={"config_dir"}),
                checkpoint_path=str(checkpoint),
                run_id=f"query-{fingerprint[:12]}",
            )
        )
        providers = _dd_providers(cfg)
        designer = DataDesigner(model_providers=providers) if providers else DataDesigner()
        designer.create(builder, num_records=pending_count)
    return _finalize(cfg, candidates, fingerprint, force=force)
