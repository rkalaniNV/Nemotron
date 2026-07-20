import pytest
from long_context_sdg.config import PipelineConfig
from long_context_sdg.episode_control import (
    build_episode_spec,
    retrieval_deadline_event,
)
from long_context_sdg.pipeline import _dd_providers
from long_context_sdg.prompts import (
    assistant_final_system,
    assistant_system,
    assistant_turn_directive,
    user_turn_prompt,
)
from long_context_sdg.schemas import RetrievalPolicyEvent
from long_context_sdg.seeds import enrich_seed, prepare_seed_file
from long_context_sdg.service_config import ProviderConfig, resolve_api_key

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
    cfg.episode.turn_budget.min = 6
    cfg.episode.turn_budget.max = 40
    cfg.episode.honor_seed_turn_budget = False

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
    payload["episode"]["retrieval_depth_weights"] = {1: 0, 2: 0, 3: 1}
    payload["episode"]["max_steps_per_turn"] = 3
    with pytest.raises(ValueError, match="final-answer step"):
        PipelineConfig.model_validate(payload)


def test_config_rejects_removed_intent_scheduler_keys(tmp_path):
    cfg = make_config(tmp_path)
    payload = cfg.model_dump(mode="json", exclude={"config_dir"})
    payload["episode"]["intents"] = {"research": 1.0}
    with pytest.raises(ValueError, match="extra_forbidden"):
        PipelineConfig.model_validate(payload)


def test_config_uses_data_designer_artifacts_instead_of_custom_checkpoint(tmp_path):
    cfg = make_config(tmp_path)

    assert "checkpoint" not in type(cfg.paths).model_fields
    assert cfg.run.resume == "always"
    assert cfg.resolve(cfg.paths.artifacts).name == "artifacts"


def test_provider_key_reference_is_resolved_and_allows_explicit_empty(monkeypatch):
    provider = ProviderConfig(
        name="test",
        endpoint="https://model.example/v1",
        api_key_env="TEST_MODEL_KEY",
    )
    monkeypatch.delenv("TEST_MODEL_KEY", raising=False)
    with pytest.raises(ValueError, match="needs environment variable"):
        resolve_api_key(provider)

    monkeypatch.setenv("TEST_MODEL_KEY", "")
    assert resolve_api_key(provider) == ""
    monkeypatch.setenv("TEST_MODEL_KEY", " secret ")
    assert resolve_api_key(provider) == " secret "


def test_data_designer_provider_receives_native_secret_reference(tmp_path):
    cfg = make_config(tmp_path)
    cfg.providers = [
        ProviderConfig(
            name="test",
            endpoint="https://model.example/v1",
            api_key_env="TEST_MODEL_KEY",
        )
    ]
    provider = _dd_providers(cfg)[0]
    assert provider.api_key == "TEST_MODEL_KEY"


def test_generation_payload_excludes_orchestration_identity(tmp_path):
    cfg = make_config(tmp_path)
    payload = cfg.generation_payload()
    assert payload["run"] == {
        "mode": "create",
        "seed": cfg.run.seed,
        "num_records": 0,
        "dataset_name": "embedded",
        "resume": "never",
    }
    assert set(payload["paths"].values()) == {"."}

    cfg.run.resume = "if_possible"
    cfg.run.dataset_name = "different"
    cfg.run.num_records = 1
    cfg.paths.generated = "elsewhere.jsonl"
    assert cfg.generation_payload() == payload


def test_fingerprint_includes_run_seed_but_not_dd_resume_mode(tmp_path):
    cfg = make_config(tmp_path)
    original = cfg.fingerprint()
    cfg.run.resume = "never"
    assert cfg.fingerprint() == original
    cfg.run.seed += 1
    assert cfg.fingerprint() != original


def test_episode_spec_is_reproducible_without_precomputing_turns(tmp_path):
    cfg = make_config(tmp_path, depth_weights={1: 0, 2: 0, 3: 1})
    seed = enrich_seed({"query_id": "spec", "query": "q", "turn_budget": 15}, cfg)

    first = build_episode_spec(seed, cfg.episode, cfg.run.seed)
    second = build_episode_spec(seed, cfg.episode, cfg.run.seed)

    assert first == second
    assert not hasattr(first, "turns")
    assert not hasattr(first, "opening_intent")
    assert 1 <= first.required_retrieval_calls <= 3
    assert first.max_tool_calls_per_turn == 3


def test_episode_spec_contains_constraints_but_no_semantic_turn_plan(tmp_path):
    cfg = make_config(tmp_path)
    seed = enrich_seed({"query_id": "natural", "query": "q", "turn_budget": 15}, cfg)
    spec = build_episode_spec(seed, cfg.episode, cfg.run.seed)

    assert set(spec.model_dump()) == {
        "query_id",
        "turn_budget",
        "required_retrieval_calls",
        "max_retrieval_calls",
        "max_tool_calls_per_turn",
        "max_tool_calls_per_conversation",
    }


def test_controller_emits_only_sparse_retrieval_deadline_events(tmp_path):
    cfg = make_config(tmp_path)
    cfg.episode.retrieval_calls.min = 2
    cfg.episode.retrieval_calls.max = 2
    seed = enrich_seed({"query_id": "online", "query": "q", "turn_budget": 15}, cfg)
    spec = build_episode_spec(seed, cfg.episode, cfg.run.seed)

    early = retrieval_deadline_event(
        spec,
        seed,
        turn=2,
        successful_retrievals=0,
        retrieval_attempts=0,
        tool_calls=0,
    )
    late = retrieval_deadline_event(
        spec,
        seed,
        turn=14,
        successful_retrievals=0,
        retrieval_attempts=0,
        tool_calls=0,
    )

    assert early is None
    assert late is not None
    assert late.reason == "retrieval_deadline"
    assert late.required_retrievals_this_turn == 1


def test_retrieval_deadline_uses_only_remaining_target(tmp_path):
    cfg = make_config(tmp_path, depth_weights={1: 0, 2: 0, 3: 1})
    cfg.episode.retrieval_calls.min = 2
    cfg.episode.retrieval_calls.max = 2
    seed = enrich_seed({"query_id": "depth", "query": "q", "turn_budget": 15}, cfg)
    spec = build_episode_spec(seed, cfg.episode, cfg.run.seed)

    event = retrieval_deadline_event(
        spec,
        seed,
        turn=15,
        successful_retrievals=1,
        retrieval_attempts=1,
        tool_calls=1,
    )

    assert event is not None
    assert event.required_retrievals_this_turn == 1


def test_satisfied_retrieval_directive_requires_a_final_answer():
    event = RetrievalPolicyEvent(
        turn=4,
        required_retrievals_this_turn=2,
        successful_retrievals_before=0,
        retrieval_attempts_before=0,
        tool_calls_before=0,
        turns_remaining=2,
    )
    directive = assistant_turn_directive(4, event, completed_retrievals=2)
    assert "final answer" in directive
    assert "without another tool call" in directive


def test_natural_turn_prompts_do_not_expose_intent_taxonomy():
    assistant = assistant_turn_directive(1, None, completed_retrievals=0)
    user = user_turn_prompt(turn=2, turns_remaining=10)

    assert "Retrieval is optional" in assistant
    assert "intent" not in assistant.lower()
    assert "intent" not in user.lower()


def test_assistant_prompts_list_exact_allowed_chunk_ids(tmp_path):
    seed = enrich_seed({"query": "q"}, make_config(tmp_path))
    chunk_ids = ["h-0123456789abcdefabcd", "h-fedcba9876543210fedc"]

    tool_prompt = assistant_system(seed, [], chunk_ids)
    final_prompt = assistant_final_system(seed, chunk_ids)

    assert all(chunk_id in tool_prompt for chunk_id in chunk_ids)
    assert all(chunk_id in final_prompt for chunk_id in chunk_ids)
    assert "[h-0123]" not in final_prompt
