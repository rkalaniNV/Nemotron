import json

import pytest
from long_context_sdg.checkpoint import (
    append_record,
    completed_query_ids,
    load_records,
    verify_fingerprint,
)
from long_context_sdg.evaluation import evaluate_checkpoint
from long_context_sdg.executors.base import ExecutionServices
from long_context_sdg.exporters import export_records
from long_context_sdg.runtime import EpisodeRunner
from long_context_sdg.schemas import CanonicalRecord
from long_context_sdg.seeds import enrich_seed
from long_context_sdg.tool_registry import ToolRegistry

from tests.fixtures import FakeRetriever, fake_models, make_config


def _accepted(tmp_path):
    cfg = make_config(tmp_path)
    seed = enrich_seed({"query": "q", "turn_budget": 15, "retrieval_depth": 1}, cfg)
    models = fake_models()
    registry = ToolRegistry(
        cfg.tools, ExecutionServices(retriever=FakeRetriever(), models=models)
    )
    return cfg, EpisodeRunner(cfg).run(models, seed, registry, run_id="run-test")


def test_checkpoint_is_append_only_and_resume_aware(tmp_path):
    cfg, record = _accepted(tmp_path)
    path = cfg.resolve(cfg.paths.checkpoint)
    append_record(path, record)
    loaded = load_records(path)
    assert loaded == [record]
    verify_fingerprint(loaded, cfg.fingerprint())
    assert record.query_id in completed_query_ids(
        loaded, retry_failed=False, retry_quarantine=False
    )
    with pytest.raises(ValueError, match="incompatible"):
        verify_fingerprint(loaded, "different")


def test_evaluation_partitions_and_export_formats(tmp_path):
    cfg, record = _accepted(tmp_path)
    append_record(cfg.resolve(cfg.paths.checkpoint), record)
    summary = evaluate_checkpoint(cfg)
    assert summary["counts"]["accepted"] == 1
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
    append_record(cfg.resolve(cfg.paths.checkpoint), record)
    summary = evaluate_checkpoint(cfg)
    assert summary["counts"]["generation_failed"] == 1


def test_evaluation_uses_latest_attempt_per_query(tmp_path):
    cfg, accepted = _accepted(tmp_path)
    failed = CanonicalRecord(
        run_id="retry",
        config_fingerprint=cfg.fingerprint(),
        query_id=accepted.query_id,
        status="generation_failed",
        validation={"ok": False, "errors": ["retry failed"]},
    )
    checkpoint = cfg.resolve(cfg.paths.checkpoint)
    append_record(checkpoint, accepted)
    append_record(checkpoint, failed)

    summary = evaluate_checkpoint(cfg)

    assert summary["total"] == 1
    assert summary["counts"] == {"generation_failed": 1}
    assert len(load_records(cfg.resolve(cfg.paths.canonical))) == 1
    assert accepted.query_id not in completed_query_ids(
        [accepted, failed], retry_failed=True, retry_quarantine=False
    )
