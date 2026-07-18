from __future__ import annotations

import json
import random
from collections import Counter

import pytest
from long_context_sdg.query_generation.allocation import largest_remainder
from long_context_sdg.query_generation.candidates import prepare_candidates
from long_context_sdg.query_generation.checkpoint import (
    append_record,
    latest_by_query,
    load_records,
    verify_fingerprint,
)
from long_context_sdg.query_generation.config import (
    PersonaLocaleConfig,
    QueryEvidenceConfig,
    QueryGenerationPipelineConfig,
)
from long_context_sdg.query_generation.evidence import (
    build_evidence_pools,
    sample_bundle,
)
from long_context_sdg.query_generation.generator import _judge_errors, _seed
from long_context_sdg.query_generation.personas import project_persona
from long_context_sdg.query_generation.pipeline import (
    _add_persona_samplers,
    _finalize,
)
from long_context_sdg.query_generation.schemas import (
    PersonaProjection,
    QueryCandidate,
    QueryDraft,
    QuerySynthesisJudgment,
    QuerySynthesisRecord,
    QueryTaxonomy,
)
from long_context_sdg.query_generation.taxonomy import load_taxonomy
from long_context_sdg.query_generation.validation import validate_draft
from long_context_sdg.schemas import RetrievalChunk


def _taxonomy_text() -> str:
    return """
version: 'test-1'
topics:
  - id: foundations
    label: Foundations
    weight: 0.45
    children:
      - id: concepts
        label: Concepts
        weight: 0.60
        seed_queries: [concept mechanisms]
      - id: limits
        label: Limits
        weight: 0.40
        seed_queries: [limitations boundaries]
  - id: decisions
    label: Decisions
    weight: 0.55
    children:
      - id: compare
        label: Compare
        weight: 0.45
        seed_queries: [compare alternatives]
      - id: apply
        label: Apply
        weight: 0.55
        seed_queries: [apply in practice]
"""


def make_query_config(tmp_path, *, count: int = 100) -> QueryGenerationPipelineConfig:
    taxonomy = tmp_path / "taxonomy.yaml"
    taxonomy.write_text(_taxonomy_text(), encoding="utf-8")
    cfg = QueryGenerationPipelineConfig.model_validate(
        {
            "paths": {
                "seeds": str(tmp_path / "queries.jsonl"),
                "evidence_manifest": str(tmp_path / "evidence.json"),
                "candidates": str(tmp_path / "candidates.jsonl"),
                "checkpoint": str(tmp_path / "checkpoint.jsonl"),
                "report": str(tmp_path / "report.json"),
            },
            "run": {"mode": "create", "seed": 19},
            "providers": [],
            "models": [
                {"alias": "assistant", "model": "fake", "provider": "external"},
                {"alias": "judge", "model": "fake", "provider": "external"},
            ],
            "retriever": {
                "endpoint": "http://retrieval/query",
                "retries": 1,
                "backoff_seconds": 0,
            },
            "query_generation": {
                "num_queries": count,
                "taxonomy_path": str(taxonomy),
                "persona_locales": [
                    {
                        "locale": "en_IN",
                        "language": "en",
                        "weight": 0.5,
                        "asset_revision": "pinned-revision",
                    },
                    {
                        "locale": "hi_Deva_IN",
                        "language": "hi-Deva",
                        "weight": 0.4,
                        "asset_revision": "pinned-revision",
                    },
                    {
                        "locale": "hi_Latn_IN",
                        "language": "hi-Latn",
                        "weight": 0.1,
                        "asset_revision": "pinned-revision",
                    },
                ],
                "evidence": {
                    "pool_size": 8,
                    "bundle_min": 2,
                    "bundle_max": 3,
                    "min_chunk_chars": 20,
                },
            },
        }
    )
    cfg.config_dir = tmp_path
    return cfg


def _chunk(index: int, *, source: str | None = None) -> RetrievalChunk:
    return RetrievalChunk(
        chunk_id=f"h-{index:020x}",
        content=(f"Evidence item {index} explains mechanisms, constraints, and tradeoffs. " * 4),
        title=f"Document {index}",
        source=source or f"source-{index}",
        score=1.0,
    )


def _persona(index: int = 0, *, language: str = "en") -> PersonaProjection:
    return PersonaProjection(
        source_dataset="example/personas",
        source_revision="pinned-revision",
        source_split=language,
        source_id=f"persona-{index}",
        language=language,
        narrative_field="persona",
        narrative="A curious practitioner evaluating a realistic decision.",
        attributes={"occupation": "practitioner"},
    )


def _candidate(
    index: int = 0,
    *,
    archetype: str = "research",
    language: str = "en",
) -> QueryCandidate:
    persona_key, persona_locale = {
        "en": ("0:en_IN", "en_IN"),
        "hi-Deva": ("1:hi_Deva_IN", "hi_Deva_IN"),
        "hi-Latn": ("2:hi_Latn_IN", "hi_Latn_IN"),
    }[language]
    return QueryCandidate(
        query_id=f"query-{index}",
        synthesis_fingerprint="fingerprint",
        candidate_index=index,
        taxonomy_id="concepts",
        taxonomy_label="Concepts",
        archetype=archetype,
        answerability=("insufficient" if archetype == "insufficient_evidence" else "answerable"),
        persona_mode="situated_need",
        persona_key=persona_key,
        persona_locale=persona_locale,
        language=language,
        evidence=[_chunk(index * 2 + 1), _chunk(index * 2 + 2)],
    )


def test_query_config_is_independent_of_conversation_configuration(tmp_path):
    cfg = make_query_config(tmp_path)
    assert {model.alias for model in cfg.models} == {"assistant", "judge"}
    assert not hasattr(cfg, "planning")
    assert not hasattr(cfg, "tools")
    assert not hasattr(cfg, "context")
    assert set(type(cfg.run).model_fields) == {"mode", "seed"}


def test_exact_archetype_allocation_for_one_hundred_queries():
    assert largest_remainder(
        100,
        {
            "research": 0.65,
            "applied_scenario": 0.15,
            "comparison": 0.10,
            "misconception": 0.05,
            "clarification": 0.03,
            "insufficient_evidence": 0.02,
        },
    ) == {
        "research": 65,
        "applied_scenario": 15,
        "comparison": 10,
        "misconception": 5,
        "clarification": 3,
        "insufficient_evidence": 2,
    }


def test_taxonomy_rejects_duplicate_ids_and_multiplies_parent_weights(tmp_path):
    taxonomy_path = tmp_path / "taxonomy.yaml"
    taxonomy_path.write_text(_taxonomy_text(), encoding="utf-8")
    taxonomy, digest = load_taxonomy(taxonomy_path)
    assert len(digest) == 64
    assert {leaf.id: weight for leaf, weight in taxonomy.weighted_leaves()} == {
        "concepts": pytest.approx(0.27),
        "limits": pytest.approx(0.18),
        "compare": pytest.approx(0.2475),
        "apply": pytest.approx(0.3025),
    }

    payload = taxonomy.model_dump()
    payload["topics"][1]["children"][0]["id"] = "concepts"
    with pytest.raises(ValueError, match="duplicate taxonomy id"):
        QueryTaxonomy.model_validate(payload)


def test_evidence_pools_filter_duplicates_and_comparison_needs_two_sources():
    taxonomy = QueryTaxonomy.model_validate(
        {
            "version": "1",
            "topics": [
                {
                    "id": "topic",
                    "label": "Topic",
                    "seed_queries": ["seed one", "seed two"],
                    "exclusions": ["exclude-me"],
                }
            ],
        }
    )
    duplicate = _chunk(1)

    class Retriever:
        def query(self, text, *, top_k=None):
            return [
                duplicate,
                duplicate.model_copy(update={"chunk_id": "h-duplicate-content"}),
                _chunk(2),
                _chunk(3).model_copy(update={"content": "exclude-me " * 20}),
            ]

    cfg = QueryEvidenceConfig(pool_size=8, bundle_min=2, bundle_max=2, min_chunk_chars=20)
    pool = build_evidence_pools(taxonomy, Retriever(), cfg)["topic"]
    assert [chunk.chunk_id for chunk in pool] == [duplicate.chunk_id, _chunk(2).chunk_id]
    bundle = sample_bundle(pool, archetype="comparison", cfg=cfg, rng=random.Random(1))
    assert len({chunk.source for chunk in bundle}) == 2

    same_source = [chunk.model_copy(update={"source": "one"}) for chunk in pool]
    with pytest.raises(ValueError, match="at least two sources"):
        sample_bundle(
            same_source,
            archetype="comparison",
            cfg=cfg,
            rng=random.Random(1),
        )


def test_candidate_preparation_has_exact_topic_archetype_and_language_marginals(tmp_path, monkeypatch):
    cfg = make_query_config(tmp_path)

    pools = {
        taxonomy_id: [_chunk(index * 10 + offset) for offset in range(1, 6)]
        for index, taxonomy_id in enumerate(("concepts", "limits", "compare", "apply"))
    }
    monkeypatch.setattr(
        "long_context_sdg.query_generation.candidates._load_or_build_pools",
        lambda *args: pools,
    )

    candidates, fingerprint = prepare_candidates(cfg)

    assert len(candidates) == 100 and len(fingerprint) == 64
    assert Counter(item.taxonomy_id for item in candidates) == {
        "concepts": 27,
        "limits": 18,
        "compare": 25,
        "apply": 30,
    }
    assert Counter(item.archetype for item in candidates) == {
        "research": 65,
        "applied_scenario": 15,
        "comparison": 10,
        "misconception": 5,
        "clarification": 3,
        "insufficient_evidence": 2,
    }
    assert Counter(item.language for item in candidates) == {
        "en": 50,
        "hi-Deva": 40,
        "hi-Latn": 10,
    }
    assert len({item.query_id for item in candidates}) == 100


def test_native_data_designer_persona_samplers_cover_each_configured_locale(tmp_path):
    import data_designer.config as dd

    cfg = make_query_config(tmp_path)

    class Builder:
        def __init__(self):
            self.columns = []

        def add_column(self, column):
            self.columns.append(column)

    builder = Builder()
    columns = _add_persona_samplers(builder, cfg, dd)

    assert len(columns) == 3
    assert set(columns) == {"0:en_IN", "1:hi_Deva_IN", "2:hi_Latn_IN"}
    assert len(set(columns.values())) == 3
    assert [column.params.locale for column in builder.columns] == [
        "en_IN",
        "hi_Deva_IN",
        "hi_Latn_IN",
    ]
    assert all(column.sampler_type == dd.SamplerType.PERSON for column in builder.columns)
    assert all(column.drop for column in builder.columns)
    assert all(column.params.with_synthetic_personas for column in builder.columns)
    assert all(list(column.params.age_range) == [18, 114] for column in builder.columns)


def test_managed_persona_projection_is_compact_deterministic_and_provenanced():
    locale = PersonaLocaleConfig(
        locale="hi_Deva_IN",
        language="hi-Deva",
        weight=1,
        asset_revision="reviewed-2026-07",
        narrative_fields={"persona": 1, "professional_persona": 1},
        attribute_fields=["occupation", "state"],
    )
    raw = {
        "uuid": "persona-123",
        "first_name": "Do not project this field",
        "occupation": "teacher",
        "state": "Rajasthan",
        "persona": "A curious adult learner comparing practical options.",
        "professional_persona": "An experienced teacher responsible for a small team.",
    }

    first = project_persona(raw, locale, seed=19)
    second = project_persona(raw, locale, seed=19)

    assert first == second
    assert first.source_dataset == "nemotron-personas/hi_Deva_IN"
    assert first.source_revision == "reviewed-2026-07"
    assert first.source_id == "persona-123"
    assert first.language == "hi-Deva"
    assert first.narrative_field in {"persona", "professional_persona"}
    assert first.attributes == {"occupation": "teacher", "state": "Rajasthan"}
    assert "first_name" not in first.attributes


def test_deterministic_validation_checks_language_leakage_and_retrievability(tmp_path):
    cfg = make_query_config(tmp_path, count=1).query_generation
    candidate = _candidate()
    draft = QueryDraft(
        query="How should a team evaluate reliability tradeoffs in practice?",
        naive_query="How should my team evaluate the tradeoffs?",
        role="team member",
        expertise="intermediate",
        style="concise",
    )

    class Retriever:
        def query(self, text, *, top_k=None):
            return [candidate.evidence[0]]

    assert validate_draft(draft, candidate, cfg, Retriever()) == []

    leaked = draft.model_copy(update={"query": f"Use {candidate.evidence[0].chunk_id} to answer this"})
    errors = validate_draft(leaked, candidate, cfg, Retriever())
    assert any("leaks chunk identifiers" in error for error in errors)

    devanagari = _candidate(language="hi-Deva")
    errors = validate_draft(draft, devanagari, cfg, Retriever())
    assert any("expected Devanagari" in error for error in errors)


def test_judge_gate_seed_provenance_and_checkpoint_resume(tmp_path):
    cfg = make_query_config(tmp_path, count=1)
    candidate = _candidate()
    draft = QueryDraft(
        query="How do the mechanisms affect a practical decision?",
        naive_query="What should I consider before deciding?",
        role="practitioner",
        expertise="intermediate",
        style="natural",
    )
    judgment = QuerySynthesisJudgment(
        scores={
            "topic_fit": 5,
            "persona_realism": 5,
            "language_quality": 5,
            "answerability": 5,
            "retrieval_quality": 5,
            "non_leakage": 5,
        },
        answerability="answerable",
        rating="success",
        explanation="usable",
    )
    assert _judge_errors(judgment, candidate, 4) == []
    seed = _seed(candidate, _persona(), draft, cfg)
    assert seed["query_provenance"]["taxonomy_id"] == candidate.taxonomy_id
    assert seed["persona"]["source_revision"] == "pinned-revision"

    record = QuerySynthesisRecord(
        query_id=candidate.query_id,
        synthesis_fingerprint=candidate.synthesis_fingerprint,
        candidate_index=0,
        attempt=1,
        status="accepted",
        taxonomy_id=candidate.taxonomy_id,
        archetype=candidate.archetype,
        language=candidate.language,
        persona_mode=candidate.persona_mode,
        draft=draft,
        judgment=judgment,
        seed=seed,
    )
    checkpoint = cfg.resolve(cfg.paths.checkpoint)
    append_record(checkpoint, record)
    append_record(
        checkpoint,
        record.model_copy(update={"attempt": 2, "status": "rejected", "seed": None}),
    )
    loaded = load_records(checkpoint)
    verify_fingerprint(loaded, "fingerprint")
    assert latest_by_query(loaded)[candidate.query_id].attempt == 2
    with pytest.raises(ValueError, match="incompatible"):
        verify_fingerprint(loaded, "different")


def test_finalization_publishes_only_complete_unique_seed_set(tmp_path):
    cfg = make_query_config(tmp_path, count=2)
    candidates = [_candidate(0), _candidate(1)]
    checkpoint = cfg.resolve(cfg.paths.checkpoint)
    for candidate in candidates:
        draft = QueryDraft(
            query=f"How should scenario {candidate.candidate_index} be evaluated?",
            naive_query=f"What about scenario {candidate.candidate_index}?",
            role="user",
            expertise="intermediate",
            style="natural",
        )
        append_record(
            checkpoint,
            QuerySynthesisRecord(
                query_id=candidate.query_id,
                synthesis_fingerprint="fingerprint",
                candidate_index=candidate.candidate_index,
                attempt=1,
                status="accepted",
                taxonomy_id=candidate.taxonomy_id,
                archetype=candidate.archetype,
                language=candidate.language,
                persona_mode=candidate.persona_mode,
                draft=draft,
                seed=_seed(candidate, _persona(candidate.candidate_index), draft, cfg),
            ),
        )

    report = _finalize(cfg, candidates, "fingerprint", force=False)
    rows = [json.loads(line) for line in cfg.resolve(cfg.paths.seeds).read_text().splitlines()]
    assert report["accepted"] == 2
    assert [row["query_id"] for row in rows] == ["query-0", "query-1"]
    assert cfg.resolve(cfg.paths.report).exists()
