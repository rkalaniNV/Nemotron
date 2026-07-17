"""Incremental checkpoint writer."""

from __future__ import annotations

import json
import os

from mtsdg.generator import write_checkpoint


def _result(qid="q-1", status=True):
    return {
        "structured_messages": json.dumps([{"role": "user", "content": "hi"}]),
        "episode_metadata": json.dumps({"query_id": qid, "n_messages": 1, "n_retrieved_chunks": 2,
                                        "compaction_events": [{"summary_id": "ctx-001", "covers_turns": [1, 3]}]}),
        "compaction_events": "{}",
        "trajectory_status": status,
        "trajectory_validation": json.dumps({"ok": status, "errors": [], "warnings": []}),
        "trajectory_judgment": json.dumps({"skipped": True}),
    }


def test_write_checkpoint_appends_records(tmp_path):
    path = str(tmp_path / "ckpt.jsonl")
    write_checkpoint(path, _result("q-1", True))
    write_checkpoint(path, _result("q-2", False))
    lines = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    assert len(lines) == 2
    assert lines[0]["query_id"] == "q-1" and lines[0]["trajectory_status"] is True
    assert lines[1]["query_id"] == "q-2" and lines[1]["trajectory_status"] is False
    assert lines[0]["messages"] == [{"role": "user", "content": "hi"}]
    assert lines[0]["compaction_events"][0]["summary_id"] == "ctx-001"


def test_write_checkpoint_noop_on_empty_path(tmp_path):
    # Empty path disables checkpointing (must not raise or create files).
    write_checkpoint("", _result())
    assert not os.listdir(tmp_path)
