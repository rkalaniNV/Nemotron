from __future__ import annotations

import json
import random
from collections import Counter

import pytest
from long_context_sdg.query_generation.allocation import largest_remainder
from long_context_sdg.query_generation.candidates import prepare_candidates
from long_context_sdg.query_generation.config import (
    PersonaLocaleConfig,
    QueryArchetypeProfile,
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
from long_context_sdg.query_generation.prompts import draft_messages
from long_context_sdg.query_generation.schemas import (
    EvidenceNeed,
    PersonaProjection,
    QueryCandidate,
    QueryDraft,
    QuerySynthesisJudgment,
    QuerySynthesisRecord,
    QueryTaxonomy,
)
from long_context_sdg.query_generation.taxonomy import load_taxonomy
from long_context_sdg.query_generation.validation import query_similarity, validate_draft
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
                "artifacts": str(tmp_path / "artifacts"),
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
                    "bundle_min": 1,
                    "bundle_max": 4,
                    "max_pair_similarity": 1.0,
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
    archetype: str = "single_lookup",
    language: str = "en",
    surface_form: str = "well_formed",
) -> QueryCandidate:
    persona_key, persona_locale = {
        "en": ("0:en_IN", "en_IN"),
        "hi-Deva": ("1:hi_Deva_IN", "hi_Deva_IN"),
        "hi-Latn": ("2:hi_Latn_IN", "hi_Latn_IN"),
    }[language]
    multi = archetype in {"comparison", "timeline", "multi_constraint", "conflict_resolution", "insufficient_evidence"}
    evidence = (
        [_chunk(index * 3 + 1), _chunk(index * 3 + 2), _chunk(index * 3 + 3)]
        if multi
        else [_chunk(index * 3 + 1)]
    )
    return QueryCandidate(
        query_id=f"query-{index}",
        synthesis_fingerprint="fingerprint",
        candidate_index=index,
        taxonomy_id="concepts",
        taxonomy_label="Concepts",
        archetype=archetype,
        evidence_scope="multi_facet" if multi else "single_facet",
        minimum_evidence_needs=2 if multi else 1,
        answerability=("insufficient" if archetype == "insufficient_evidence" else "answerable"),
        persona_mode="situated_need",
        surface_form=surface_form,
        persona_key=persona_key,
        persona_locale=persona_locale,
        language=language,
        evidence=evidence,
    )


def test_query_config_is_independent_of_conversation_configuration(tmp_path):
    cfg = make_query_config(tmp_path)
    assert {model.alias for model in cfg.models} == {"assistant", "judge"}
    assert not hasattr(cfg, "episode")
    assert not hasattr(cfg, "tools")
    assert not hasattr(cfg, "context")
    assert set(type(cfg.run).model_fields) == {"mode", "seed", "dataset_name", "resume"}
    assert cfg.generation_payload()["run"] == {
        "mode": "create",
        "seed": cfg.run.seed,
        "dataset_name": "embedded",
        "resume": "never",
    }
    assert set(cfg.generation_payload()["paths"].values()) == {"."}


def test_surface_forms_and_evidence_scopes_reject_unsupported_or_impossible_config(tmp_path):
    with pytest.raises(ValueError, match="exactly one evidence need"):
        QueryArchetypeProfile(
            evidence_scope="single_facet",
            bundle_min=1,
            bundle_max=1,
            min_sources=1,
            min_evidence_needs=0,
        )
    with pytest.raises(ValueError, match="zero evidence needs"):
        QueryArchetypeProfile(
            evidence_scope="conversational",
            bundle_min=1,
            bundle_max=1,
            min_sources=1,
            min_evidence_needs=1,
        )

    payload = make_query_config(tmp_path).model_dump(mode="json", exclude={"config_dir"})
    payload["query_generation"]["surface_form_weights"] = {"custom_surface": 1.0}
    payload["query_generation"]["surface_form_profiles"] = {
        "custom_surface": {"minimum_anchor_recall_gap": 0.0}
    }
    with pytest.raises(ValueError, match="surface forms are unsupported"):
        QueryGenerationPipelineConfig.model_validate(payload)


def test_exact_archetype_allocation_for_one_hundred_queries():
    assert largest_remainder(
        100,
        {
            "single_lookup": 0.25,
            "comparison": 0.20,
            "timeline": 0.15,
            "multi_constraint": 0.20,
            "conflict_resolution": 0.10,
            "clarification": 0.05,
            "insufficient_evidence": 0.05,
        },
    ) == {
        "single_lookup": 25,
        "comparison": 20,
        "timeline": 15,
        "multi_constraint": 20,
        "conflict_resolution": 10,
        "clarification": 5,
        "insufficient_evidence": 5,
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

    cfg = QueryEvidenceConfig(
        pool_size=8,
        bundle_min=1,
        bundle_max=2,
        max_pair_similarity=1,
        min_chunk_chars=20,
    )
    profile = QueryArchetypeProfile(
        evidence_scope="multi_facet",
        bundle_min=2,
        bundle_max=2,
        min_sources=2,
        min_evidence_needs=2,
    )
    pool = build_evidence_pools(taxonomy, Retriever(), cfg)["topic"]
    assert [chunk.chunk_id for chunk in pool] == [duplicate.chunk_id, _chunk(2).chunk_id]
    bundle = sample_bundle(pool, archetype="comparison", profile=profile, cfg=cfg, rng=random.Random(1))
    assert len({chunk.source for chunk in bundle}) == 2

    same_source = [chunk.model_copy(update={"source": "one"}) for chunk in pool]
    with pytest.raises(ValueError, match=r"at least 2 source"):
        sample_bundle(
            same_source,
            archetype="comparison",
            profile=profile,
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
        "single_lookup": 25,
        "comparison": 20,
        "timeline": 15,
        "multi_constraint": 20,
        "conflict_resolution": 10,
        "clarification": 5,
        "insufficient_evidence": 5,
    }
    assert Counter(item.language for item in candidates) == {
        "en": 50,
        "hi-Deva": 40,
        "hi-Latn": 10,
    }
    assert Counter(item.surface_form for item in candidates) == {
        "well_formed": 20,
        "underspecified": 20,
        "retrieval_rewrite": 20,
        "noisy_language": 10,
        "keyword_fragment": 10,
        "overbroad": 10,
        "adjacent_intent": 10,
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
        evidence_needs=[
            EvidenceNeed(
                need="Identify the relevant reliability mechanism.",
                retrieval_probe="reliability mechanism tradeoffs",
                supporting_chunk_ids=[candidate.evidence[0].chunk_id],
            )
        ],
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


def test_underspecified_surface_form_must_be_improved_by_canonical_rewrite(tmp_path):
    cfg = make_query_config(tmp_path, count=1).query_generation
    candidate = _candidate(surface_form="underspecified")
    draft = QueryDraft(
        query="How do reliability mechanisms affect this operational decision?",
        naive_query="What about reliability for my case?",
        role="operator",
        expertise="intermediate",
        style="informal",
        evidence_needs=[
            EvidenceNeed(
                need="Understand the reliability mechanism.",
                retrieval_probe="reliability mechanism operational effects",
                supporting_chunk_ids=[candidate.evidence[0].chunk_id],
            )
        ],
    )

    class ImprovedRetriever:
        def query(self, text, *, top_k=None):
            if text in {draft.query, draft.evidence_needs[0].retrieval_probe}:
                return [candidate.evidence[0]]
            return []

    assert validate_draft(draft, candidate, cfg, ImprovedRetriever()) == []

    class NoImprovementRetriever:
        def query(self, text, *, top_k=None):
            return [candidate.evidence[0]]

    errors = validate_draft(draft, candidate, cfg, NoImprovementRetriever())
    assert any("improved anchor recall" in error for error in errors)


def test_multifacet_draft_requires_distinct_independently_retrievable_probes(tmp_path):
    cfg = make_query_config(tmp_path, count=1).query_generation
    candidate = _candidate(archetype="comparison")
    draft = QueryDraft(
        query="How do the two mechanisms differ in constraints and outcomes?",
        naive_query="Can you compare these approaches for me?",
        role="decision maker",
        expertise="intermediate",
        style="analytical",
        evidence_needs=[
            EvidenceNeed(
                need="Establish the first mechanism's constraints.",
                retrieval_probe="first mechanism operational constraints",
                supporting_chunk_ids=[candidate.evidence[0].chunk_id],
            ),
            EvidenceNeed(
                need="Establish the second mechanism's outcomes.",
                retrieval_probe="second mechanism measured outcomes",
                supporting_chunk_ids=[candidate.evidence[1].chunk_id],
            ),
        ],
    )

    class FacetRetriever:
        def query(self, text, *, top_k=None):
            if text == draft.evidence_needs[0].retrieval_probe:
                return [candidate.evidence[0]]
            if text == draft.evidence_needs[1].retrieval_probe:
                return [candidate.evidence[1]]
            return [candidate.evidence[0]]

    assert validate_draft(draft, candidate, cfg, FacetRetriever()) == []
    duplicate = draft.model_copy(
        update={
            "evidence_needs": [
                draft.evidence_needs[0],
                draft.evidence_needs[1].model_copy(
                    update={"retrieval_probe": draft.evidence_needs[0].retrieval_probe}
                ),
            ]
        }
    )
    errors = validate_draft(duplicate, candidate, cfg, FacetRetriever())
    assert any("probes are too similar" in error for error in errors)


def test_surface_form_prompt_supports_noisy_and_adjacent_queries_without_runtime_policy():
    candidate = _candidate(surface_form="noisy_language")
    messages = draft_messages(candidate, _persona())
    system = messages[0]["content"]
    payload = json.loads(messages[1]["content"])

    assert payload["surface_form"] == "noisy_language"
    assert "imperfect/non-native English" in payload["surface_form_guidance"]
    assert "whether the downstream assistant should call a tool" in system
    with pytest.raises(ValueError, match="extra_forbidden"):
        QueryDraft.model_validate(
            {
                "query": "canonical query",
                "naive_query": "naive query",
                "role": "user",
                "expertise": "novice",
                "style": "natural",
                "seed_instructions": "Call retrieve three times",
            }
        )


def test_batch_duplicate_similarity_catches_reordered_queries():
    left = QueryDraft(
        query="Compare federal and California rules for dying declarations",
        naive_query="Compare federal and California rules for dying declarations",
        role="researcher",
        expertise="intermediate",
        style="concise",
    )
    right = QueryDraft(
        query="How do California and federal dying declaration rules compare",
        naive_query="How do California and federal dying declaration rules compare",
        role="researcher",
        expertise="intermediate",
        style="concise",
    )

    assert query_similarity(left, right) >= 0.72


def test_judge_gate_seed_provenance_and_record_round_trip(tmp_path):
    cfg = make_query_config(tmp_path, count=1)
    candidate = _candidate()
    draft = QueryDraft(
        query="How do the mechanisms affect a practical decision?",
        naive_query="What should I consider before deciding?",
        role="practitioner",
        expertise="intermediate",
        style="natural",
        evidence_needs=[
            EvidenceNeed(
                need="Understand the governing mechanism.",
                retrieval_probe="governing mechanism practical effects",
                supporting_chunk_ids=[candidate.evidence[0].chunk_id],
            )
        ],
    )
    judgment = QuerySynthesisJudgment(
        scores={
            "topic_fit": 5,
            "persona_realism": 5,
            "language_quality": 5,
            "answerability": 5,
            "retrieval_quality": 5,
            "evidence_structure": 5,
            "non_leakage": 5,
            "surface_form_fidelity": 5,
            "rewrite_value": 5,
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
        surface_form=candidate.surface_form,
        draft=draft,
        judgment=judgment,
        seed=seed,
    )
    assert QuerySynthesisRecord.model_validate_json(record.model_dump_json()) == record


def test_finalization_publishes_only_complete_unique_seed_set(tmp_path):
    cfg = make_query_config(tmp_path, count=2)
    candidates = [_candidate(0), _candidate(1)]
    records = []
    for candidate in candidates:
        topic = (
            "solar panel degradation efficiency"
            if candidate.candidate_index == 0
            else "database backup restoration topology"
        )
        draft = QueryDraft(
            query=f"How should {topic} be evaluated?",
            naive_query=f"I need help with {topic}.",
            role="user",
            expertise="intermediate",
            style="natural",
            evidence_needs=[
                EvidenceNeed(
                    need=f"Resolve evidence about {topic}.",
                    retrieval_probe=f"{topic} evidence",
                    supporting_chunk_ids=[candidate.evidence[0].chunk_id],
                )
            ],
        )
        records.append(
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
                surface_form=candidate.surface_form,
                draft=draft,
                seed=_seed(candidate, _persona(candidate.candidate_index), draft, cfg),
            ),
        )

    report = _finalize(cfg, candidates, "fingerprint", records, force=False)
    rows = [json.loads(line) for line in cfg.resolve(cfg.paths.seeds).read_text().splitlines()]
    assert report["accepted"] == 2
    assert report["target_counts"] == report["realized_counts"]
    assert report["realized_counts"]["surface_form"] == {"well_formed": 2}
    assert [row["query_id"] for row in rows] == ["query-0", "query-1"]
    assert cfg.resolve(cfg.paths.report).exists()
