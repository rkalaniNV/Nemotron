import json

import pytest
from long_context_sdg.evaluation import evaluate_generated
from long_context_sdg.executors.base import ExecutionServices
from long_context_sdg.exporters import export_records
from long_context_sdg.records import load_records, write_records
from long_context_sdg.runtime import EpisodeRunner
from long_context_sdg.schemas import CanonicalRecord
from long_context_sdg.seeds import enrich_seed
from long_context_sdg.tool_registry import ToolRegistry

from tests.fixtures import FakeRetriever, fake_models, make_config


def _accepted(tmp_path):
    cfg = make_config(tmp_path)
    seed = enrich_seed({"query": "q", "turn_budget": 15}, cfg)
    models = fake_models()
    registry = ToolRegistry(cfg.tools, ExecutionServices(retriever=FakeRetriever(), models=models))
    return cfg, EpisodeRunner(cfg).run(models, seed, registry, run_id="run-test")


def test_generated_records_are_atomically_written_and_parsed(tmp_path):
    cfg, record = _accepted(tmp_path)
    path = cfg.resolve(cfg.paths.generated)

    assert write_records(path, [record]) == 1
    assert load_records(path) == [record]


def test_evaluation_partitions_reports_observed_behavior_and_exports(tmp_path):
    cfg, record = _accepted(tmp_path)
    write_records(cfg.resolve(cfg.paths.generated), [record])

    summary = evaluate_generated(cfg)

    assert summary["counts"]["accepted"] == 1
    assert summary["turn_budget_summary"] == {
        "count": 1,
        "min": 15,
        "median": 15,
        "mean": 15.0,
        "max": 15,
    }
    assert summary["successful_retrieval_summary"]["count"] == 1
    assert summary["retrieval_call_summary"]["count"] == 1
    assert summary["low_gain_retrieval_summary"]["count"] == 1
    assert summary["rejected_redundant_retrieval_summary"]["count"] == 1
    assert summary["tool_call_summary"]["count"] == 1
    assert "intent_counts" not in summary
    canonical = cfg.resolve(cfg.paths.canonical)
    for output_format in ("messages", "messages_and_tools", "rich"):
        destination = tmp_path / f"{output_format}.jsonl"
        assert export_records(canonical, destination, output_format=output_format) == 1
        row = json.loads(destination.read_text())
        assert "messages" in row
        if output_format == "messages_and_tools":
            assert "tools" in row
        if output_format == "messages":
            assert set(row) == {"messages"}


def test_generation_failure_remains_separate(tmp_path):
    cfg = make_config(tmp_path)
    record = CanonicalRecord(
        run_id="run",
        config_fingerprint=cfg.fingerprint(),
        query_id="q",
        status="generation_failed",
        validation={"ok": False, "errors": ["failed"]},
    )
    write_records(cfg.resolve(cfg.paths.generated), [record])

    summary = evaluate_generated(cfg)

    assert summary["counts"]["generation_failed"] == 1


def test_evaluation_rejects_duplicate_generated_query_ids(tmp_path):
    cfg, accepted = _accepted(tmp_path)
    duplicate = accepted.model_copy(update={"run_id": "duplicate"})
    write_records(cfg.resolve(cfg.paths.generated), [accepted, duplicate])

    with pytest.raises(ValueError, match="duplicate query IDs"):
        evaluate_generated(cfg)


def test_evaluation_rejects_incompatible_generated_fingerprint(tmp_path):
    cfg, accepted = _accepted(tmp_path)
    incompatible = accepted.model_copy(update={"config_fingerprint": "different"})
    write_records(cfg.resolve(cfg.paths.generated), [incompatible])

    with pytest.raises(ValueError, match="incompatible"):
        evaluate_generated(cfg)


def test_visible_unknown_chunk_citation_is_rejected(tmp_path):
    cfg, accepted = _accepted(tmp_path)
    accepted.messages[-1]["content"] += " Unsupported citation: [[550e8400-e29b-41d4-a716-446655440000]]."
    write_records(cfg.resolve(cfg.paths.generated), [accepted])

    summary = evaluate_generated(cfg)

    assert summary["counts"]["rejected"] == 1
    rejected = load_records(cfg.resolve(cfg.paths.canonical))[0]
    assert any("unknown retrieved chunk IDs" in error for error in rejected.validation["errors"])
