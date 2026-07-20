from pathlib import Path

import pytest
from long_context_sdg.config import PipelineConfig, load_config
from long_context_sdg.episode_control import build_episode_spec
from long_context_sdg.pipeline import _dd_providers
from long_context_sdg.prompts import (
    assistant_final_system,
    assistant_system,
    assistant_turn_directive,
    user_turn_prompt,
)
from long_context_sdg.query_generation.config import load_query_generation_config
from long_context_sdg.seeds import enrich_seed, prepare_seed_file
from long_context_sdg.service_config import ProviderConfig, resolve_api_key

from tests.fixtures import make_config

PACKAGE_ROOT = Path(__file__).parents[1]


def test_raw_seed_enrichment_is_reproducible(tmp_path):
    cfg = make_config(tmp_path)
    raw = {
        "query": "How does this policy work?",
        "instructions": "Répondez en français.",
    }
    first = enrich_seed(raw, cfg)
    second = enrich_seed(raw, cfg)
    assert first == second
    assert 15 <= first.turn_budget <= 22
    assert not hasattr(first, "retrieval_depth")
    assert "Répondez en français" in first.instructions


def test_rich_seed_overrides_defaults(tmp_path):
    cfg = make_config(tmp_path)
    seed = enrich_seed(
        {
            "query_id": "rich-1",
            "query": "q",
            "naive_query": "n",
            "turn_budget": 17,
            "persona": {"role": "analyst", "expertise": "expert", "style": "terse"},
        },
        cfg,
    )
    assert seed.query_id == "rich-1"
    assert seed.naive_query == "n"
    assert seed.turn_budget == 17


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


def test_config_rejects_unknown_keys_and_impossible_retrieval_budget(tmp_path):
    cfg = make_config(tmp_path)
    payload = cfg.model_dump(mode="json", exclude={"config_dir"})
    payload["unknown_key"] = True
    with pytest.raises(ValueError, match="extra_forbidden"):
        PipelineConfig.model_validate(payload)

    payload.pop("unknown_key")
    payload["episode"]["max_retrieval_calls_per_turn"] = 3
    payload["episode"]["max_tool_calls_per_turn"] = 2
    with pytest.raises(ValueError, match="max_retrieval_calls_per_turn"):
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
    cfg = make_config(tmp_path)
    seed = enrich_seed({"query_id": "spec", "query": "q", "turn_budget": 15}, cfg)

    first = build_episode_spec(seed, cfg.episode, cfg.run.seed)
    second = build_episode_spec(seed, cfg.episode, cfg.run.seed)

    assert first == second
    assert not hasattr(first, "turns")
    assert not hasattr(first, "opening_intent")
    assert not hasattr(first, "required_retrieval_calls")
    assert first.max_retrieval_calls == 3
    assert first.max_retrieval_calls_per_turn == 1


def test_episode_spec_contains_constraints_but_no_semantic_turn_plan(tmp_path):
    cfg = make_config(tmp_path)
    seed = enrich_seed({"query_id": "natural", "query": "q", "turn_budget": 15}, cfg)
    spec = build_episode_spec(seed, cfg.episode, cfg.run.seed)

    assert set(spec.model_dump()) == {
        "query_id",
        "turn_budget",
        "max_retrieval_calls",
        "max_retrieval_calls_per_turn",
        "max_tool_calls_per_turn",
        "max_tool_calls_per_conversation",
        "query_lexical_similarity_threshold",
        "evidence_lexical_similarity_threshold",
        "min_new_chunk_fraction",
        "max_low_gain_chain",
        "low_gain_followup_similarity_threshold",
    }


def test_legacy_retrieval_floor_is_rejected_but_optional_cap_migrates(tmp_path):
    payload = make_config(tmp_path).model_dump(mode="json", exclude={"config_dir"})
    payload["episode"].pop("max_retrieval_calls")
    payload["episode"]["retrieval_calls"] = {"min": 1, "max": 4}
    with pytest.raises(ValueError, match="no longer supported"):
        PipelineConfig.model_validate(payload)

    payload["episode"]["retrieval_calls"]["min"] = 0
    assert PipelineConfig.model_validate(payload).episode.max_retrieval_calls == 4

    payload = make_config(tmp_path).model_dump(mode="json", exclude={"config_dir"})
    payload["episode"]["retrieval_depth_weights"] = {1: 1.0}
    with pytest.raises(ValueError, match="retrieval_depth_weights is no longer supported"):
        PipelineConfig.model_validate(payload)


def test_shipped_yaml_configs_load_with_strict_models():
    conversation = load_config(PACKAGE_ROOT / "config" / "default.yaml")
    query_generation = load_query_generation_config(
        PACKAGE_ROOT / "config" / "query_generation.example.yaml"
    )

    assert conversation.episode.max_retrieval_calls_per_turn == 1
    assert conversation.episode.retrieval_novelty.max_low_gain_chain == 1
    assert query_generation.query_generation.surface_form_weights["underspecified"] > 0
    assert query_generation.query_generation.evidence.max_probe_similarity == 0.80


def test_legacy_novelty_names_migrate_without_claiming_semantic_similarity(tmp_path):
    payload = make_config(tmp_path).model_dump(mode="json", exclude={"config_dir"})
    novelty = payload["episode"]["retrieval_novelty"]
    novelty["query_similarity_threshold"] = novelty.pop(
        "query_lexical_similarity_threshold"
    )
    novelty["evidence_similarity_threshold"] = novelty.pop(
        "evidence_lexical_similarity_threshold"
    )
    novelty["max_low_gain_calls"] = novelty.pop("max_low_gain_chain")

    migrated = PipelineConfig.model_validate(payload)

    assert migrated.episode.retrieval_novelty.query_lexical_similarity_threshold == 0.80
    assert migrated.episode.retrieval_novelty.evidence_lexical_similarity_threshold == 0.85


def test_natural_turn_prompts_do_not_expose_intent_taxonomy():
    assistant = assistant_turn_directive(1)
    user = user_turn_prompt(turn=2, turns_remaining=10)

    assert "materially improve" in assistant
    assert "quota" not in assistant.lower()
    assert "do not retrieve" not in assistant.lower()
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
