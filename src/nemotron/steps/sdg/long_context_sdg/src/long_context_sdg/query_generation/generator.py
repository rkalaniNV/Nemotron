"""Data Designer generator for one independently checkpointed query candidate."""

from __future__ import annotations

import json
from pathlib import Path

from data_designer.engine.column_generators.generators.base import (
    ColumnGeneratorCellByCell,
    ColumnGeneratorWithModelRegistry,
)

from ..llm import call_structured
from ..retrieval import RetrieverClient
from .checkpoint import append_record, latest_by_query, load_records
from .config import QueryGenerationPipelineConfig
from .evidence import content_hash
from .generator_config import SyntheticQueryConfig
from .personas import persona_config_by_key, project_persona
from .prompts import JUDGE_DIMENSIONS, draft_messages, judge_messages
from .schemas import (
    PersonaProjection,
    QueryCandidate,
    QueryDraft,
    QuerySynthesisJudgment,
    QuerySynthesisRecord,
)
from .validation import validate_draft


def _seed(
    candidate: QueryCandidate,
    persona: PersonaProjection,
    draft: QueryDraft,
    cfg: QueryGenerationPipelineConfig,
) -> dict:
    language_instruction = (
        f"Write every visible user and assistant turn in {candidate.language}. "
        "Keep the synthetic reasoning.think field in English."
    )
    instructions = "\n\n".join(value for value in (language_instruction, draft.seed_instructions.strip()) if value)
    return {
        "query_id": candidate.query_id,
        "query": draft.query.strip(),
        "naive_query": draft.naive_query.strip(),
        "persona": {
            "role": draft.role.strip(),
            "expertise": draft.expertise.strip(),
            "style": draft.style.strip(),
            "description": persona.narrative,
            "language": candidate.language,
            "source_id": persona.source_id,
            "source_dataset": persona.source_dataset,
            "source_revision": persona.source_revision,
            "source_split": persona.source_split,
            "attributes": persona.attributes,
        },
        "instructions": instructions,
        "query_provenance": {
            "synthesis_fingerprint": candidate.synthesis_fingerprint,
            "taxonomy_id": candidate.taxonomy_id,
            "archetype": candidate.archetype,
            "answerability": candidate.answerability,
            "evidence_chunk_ids": [chunk.chunk_id for chunk in candidate.evidence],
            "evidence_hashes": [content_hash(chunk) for chunk in candidate.evidence],
            "evidence_sources": [chunk.source for chunk in candidate.evidence],
            "generator_alias": cfg.query_generation.generator_alias,
            "judge_alias": cfg.query_generation.judge_alias,
            "prompt_version": "query-synthesis-v1",
        },
    }


def _judge_errors(
    judgment: QuerySynthesisJudgment,
    candidate: QueryCandidate,
    minimum: int,
) -> list[str]:
    missing = sorted(set(JUDGE_DIMENSIONS) - set(judgment.scores))
    below = {
        dimension: judgment.scores.get(dimension, 0)
        for dimension in JUDGE_DIMENSIONS
        if judgment.scores.get(dimension, 0) < minimum
    }
    errors = []
    if missing:
        errors.append(f"judge omitted dimensions: {missing}")
    if below:
        errors.append(f"judge scores below {minimum}: {below}")
    if judgment.rating != "success":
        errors.append("judge rating is failure")
    if judgment.answerability != candidate.answerability:
        errors.append(
            f"judge answerability `{judgment.answerability}` does not match target `{candidate.answerability}`"
        )
    return errors


class PersonaQueryGenerator(
    ColumnGeneratorCellByCell[SyntheticQueryConfig],
    ColumnGeneratorWithModelRegistry[SyntheticQueryConfig],
):
    def generate(self, data: dict) -> dict:
        cfg = QueryGenerationPipelineConfig.model_validate(self.config.pipeline)
        generation = cfg.query_generation
        raw = data[self.config.candidate_input_column]
        candidate = (
            QueryCandidate.model_validate_json(raw) if isinstance(raw, str) else QueryCandidate.model_validate(raw)
        )
        checkpoint_path = Path(self.config.checkpoint_path)
        latest = latest_by_query(load_records(checkpoint_path)).get(candidate.query_id)
        start_attempt = latest.attempt + 1 if latest else 1
        if latest and latest.status == "accepted":
            data["synthetic_seed"] = json.dumps(latest.seed, ensure_ascii=False)
            data["query_status"] = "accepted"
            data["query_validation"] = "[]"
            data[self.config.name] = data["synthetic_seed"]
            return data

        models = {
            generation.generator_alias: self.get_model(generation.generator_alias),
            generation.judge_alias: self.get_model(generation.judge_alias),
        }
        retriever = RetrieverClient(cfg.retriever)
        final_record = latest
        previous_errors = list(latest.validation_errors) if latest else []
        persona_configs = persona_config_by_key(generation.persona_locales)
        try:
            for attempt in range(start_attempt, generation.max_attempts + 1):
                try:
                    persona = project_persona(
                        data[self.config.persona_columns[candidate.persona_key]],
                        persona_configs[candidate.persona_key],
                        seed=cfg.run.seed,
                    )
                    draft = call_structured(
                        models,
                        generation.generator_alias,
                        draft_messages(candidate, persona, previous_errors),
                        QueryDraft,
                    )
                    errors = validate_draft(draft, candidate, generation, retriever)
                    judgment = None
                    if not errors:
                        judgment = call_structured(
                            models,
                            generation.judge_alias,
                            judge_messages(candidate, persona, draft),
                            QuerySynthesisJudgment,
                        )
                        errors.extend(_judge_errors(judgment, candidate, generation.min_judge_score))
                    seed = _seed(candidate, persona, draft, cfg) if not errors else None
                    final_record = QuerySynthesisRecord(
                        query_id=candidate.query_id,
                        synthesis_fingerprint=candidate.synthesis_fingerprint,
                        candidate_index=candidate.candidate_index,
                        attempt=attempt,
                        status="accepted" if not errors else "rejected",
                        taxonomy_id=candidate.taxonomy_id,
                        archetype=candidate.archetype,
                        language=candidate.language,
                        persona_mode=candidate.persona_mode,
                        draft=draft,
                        judgment=judgment,
                        validation_errors=errors,
                        seed=seed,
                    )
                except Exception as exc:
                    final_record = QuerySynthesisRecord(
                        query_id=candidate.query_id,
                        synthesis_fingerprint=candidate.synthesis_fingerprint,
                        candidate_index=candidate.candidate_index,
                        attempt=attempt,
                        status="generation_failed",
                        taxonomy_id=candidate.taxonomy_id,
                        archetype=candidate.archetype,
                        language=candidate.language,
                        persona_mode=candidate.persona_mode,
                        validation_errors=[str(exc)],
                    )
                append_record(checkpoint_path, final_record)
                if final_record.status == "accepted":
                    break
                previous_errors = list(final_record.validation_errors)
        finally:
            retriever.close()

        if final_record is None:
            raise RuntimeError("query candidate had no available synthesis attempts")
        data["synthetic_seed"] = json.dumps(final_record.seed, ensure_ascii=False) if final_record.seed else ""
        data["query_status"] = final_record.status
        data["query_validation"] = json.dumps(final_record.validation_errors, ensure_ascii=False)
        data[self.config.name] = data["synthetic_seed"]
        return data
