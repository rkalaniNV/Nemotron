import pytest
from long_context_sdg.config import PipelineConfig
from long_context_sdg.planning import RESEARCH_INTENTS, plan_episode
from long_context_sdg.prompts import (
    assistant_final_system,
    assistant_system,
    assistant_turn_directive,
)
from long_context_sdg.schemas import TurnPlan
from long_context_sdg.seeds import enrich_seed, prepare_seed_file

from tests.fixtures import make_config


def test_raw_seed_enrichment_is_reproducible(tmp_path):
    cfg = make_config(tmp_path, depth_weights={1: 0, 2: 1, 3: 0})
    raw = {
        "query": "How does this policy work?",
        "instructions": "Répondez en français.",
    }
    first = enrich_seed(raw, cfg)
    second = enrich_seed(raw, cfg)
    assert first == second
    assert 15 <= first.turn_budget <= 22
    assert first.retrieval_depth == 2
    assert "Répondez en français" in first.instructions


def test_rich_seed_overrides_defaults(tmp_path):
    cfg = make_config(tmp_path)
    seed = enrich_seed(
        {
            "query_id": "rich-1",
            "query": "q",
            "naive_query": "n",
            "turn_budget": 17,
            "retrieval_depth": 3,
            "persona": {"role": "analyst", "expertise": "expert", "style": "terse"},
        },
        cfg,
    )
    assert seed.query_id == "rich-1"
    assert seed.naive_query == "n"
    assert seed.turn_budget == 17 and seed.retrieval_depth == 3


def test_config_can_ignore_seed_turn_budget_and_sample_full_range(tmp_path):
    cfg = make_config(tmp_path)
    cfg.planning.turn_budget.min = 6
    cfg.planning.turn_budget.max = 40
    cfg.planning.honor_seed_turn_budget = False

    budgets = {
        enrich_seed(
            {
                "query_id": f"variable-{index:03d}",
                "query": "q",
                "turn_budget": 22,
            },
            cfg,
        ).turn_budget
        for index in range(100)
    }

    assert min(budgets) >= 6
    assert max(budgets) <= 40
    assert len(budgets) >= 25
    assert budgets != {22}


def test_disallowed_memory_seed_is_rejected(tmp_path):
    cfg = make_config(tmp_path)
    with pytest.raises(ValueError, match="disallowed"):
        enrich_seed({"query": "q", "memory_seed": {"secret": "x"}}, cfg)


def test_prepare_only_validates_and_enriches(tmp_path):
    cfg = make_config(tmp_path)
    source = cfg.resolve(cfg.paths.seeds)
    source.write_text('{"query":"first"}\n{"query":"second"}\n', encoding="utf-8")
    assert prepare_seed_file(cfg) == 2
    assert len(cfg.resolve(cfg.paths.enriched_seeds).read_text().splitlines()) == 2


def test_prepare_rejects_duplicate_query_ids_without_replacing_output(tmp_path):
    cfg = make_config(tmp_path)
    source = cfg.resolve(cfg.paths.seeds)
    destination = cfg.resolve(cfg.paths.enriched_seeds)
    destination.write_text("existing\n", encoding="utf-8")
    source.write_text('{"query":"same"}\n{"query":"same"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate query_id"):
        prepare_seed_file(cfg)

    assert destination.read_text(encoding="utf-8") == "existing\n"


def test_config_rejects_unknown_keys_and_impossible_step_budget(tmp_path):
    cfg = make_config(tmp_path)
    payload = cfg.model_dump(mode="json", exclude={"config_dir"})
    payload["unknown_key"] = True
    with pytest.raises(ValueError, match="extra_forbidden"):
        PipelineConfig.model_validate(payload)

    payload.pop("unknown_key")
    payload["planning"]["retrieval_depth_weights"] = {1: 0, 2: 0, 3: 1}
    payload["planning"]["max_steps_per_turn"] = 3
    with pytest.raises(ValueError, match="final-answer step"):
        PipelineConfig.model_validate(payload)


def test_fingerprint_includes_run_seed_but_not_retry_policy(tmp_path):
    cfg = make_config(tmp_path)
    original = cfg.fingerprint()
    cfg.run.retry_failed = not cfg.run.retry_failed
    assert cfg.fingerprint() == original
    cfg.run.seed += 1
    assert cfg.fingerprint() != original


def test_plan_applies_depth_only_to_research_intents(tmp_path):
    cfg = make_config(tmp_path, depth_weights={1: 0, 2: 0, 3: 1})
    seed = enrich_seed({"query": "q", "turn_budget": 15}, cfg)
    plan = plan_episode(seed, cfg.planning, cfg.run.seed)
    assert len(plan.turns) == 15
    assert any(t.intent in RESEARCH_INTENTS for t in plan.turns)
    assert all(
        t.retrieval_depth == 3 for t in plan.turns if t.intent in RESEARCH_INTENTS
    )
    assert all(
        t.retrieval_depth == 0 for t in plan.turns if t.intent not in RESEARCH_INTENTS
    )


def test_first_turn_intents_are_diverse_and_not_always_retrieval(tmp_path):
    cfg = make_config(tmp_path)
    first_intents = set()
    for index in range(100):
        seed = enrich_seed(
            {
                "query_id": f"diverse-{index:03d}",
                "query": "q",
                "turn_budget": 15,
            },
            cfg,
        )
        plan = plan_episode(seed, cfg.planning, cfg.run.seed)
        first_intents.add(plan.turns[0].intent)
        assert any(turn.retrieval_required for turn in plan.turns)

    assert len(first_intents) >= 6
    assert first_intents - RESEARCH_INTENTS
    assert first_intents & RESEARCH_INTENTS


def test_episode_retrieval_fallback_never_forces_turn_one(tmp_path):
    cfg = make_config(tmp_path)
    cfg.planning.first_turn_intents = {"clarify": 1.0}
    cfg.planning.intents = {"user_context": 1.0}
    seed = enrich_seed({"query_id": "fallback", "query": "q", "turn_budget": 15}, cfg)

    plan = plan_episode(seed, cfg.planning, cfg.run.seed)

    assert plan.turns[0].intent == "clarify"
    required_turns = [turn.turn for turn in plan.turns if turn.retrieval_required]
    assert len(required_turns) == 1
    assert required_turns[0] > 1


def test_satisfied_retrieval_directive_requires_a_final_answer():
    plan = TurnPlan(
        turn=4,
        intent="research",
        retrieval_required=True,
        retrieval_depth=2,
    )
    directive = assistant_turn_directive(plan, completed_retrievals=2)
    assert "final answer" in directive
    assert "without another tool call" in directive


def test_clarification_intent_has_explicit_opening_guidance():
    plan = TurnPlan(turn=1, intent="clarify")
    directive = assistant_turn_directive(plan, completed_retrievals=0)
    assert "Ask a focused clarification" in directive
    assert "Retrieval is optional" in directive


def test_assistant_prompts_list_exact_allowed_chunk_ids(tmp_path):
    seed = enrich_seed({"query": "q"}, make_config(tmp_path))
    chunk_ids = ["h-0123456789abcdefabcd", "h-fedcba9876543210fedc"]

    tool_prompt = assistant_system(seed, [], chunk_ids)
    final_prompt = assistant_final_system(seed, chunk_ids)

    assert all(chunk_id in tool_prompt for chunk_id in chunk_ids)
    assert all(chunk_id in final_prompt for chunk_id in chunk_ids)
    assert "[h-0123]" not in final_prompt
