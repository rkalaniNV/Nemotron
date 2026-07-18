"""Decoupled deterministic validation, rejudging, partitioning, and reporting."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .checkpoint import load_records, verify_fingerprint
from .config import PipelineConfig
from .llm import call_structured
from .schemas import CanonicalRecord, EpisodePlan, TrajectoryJudgment
from .validation import reconstruct_messages, validate_trajectory


def _judge_record(
    record: CanonicalRecord, cfg: PipelineConfig, models: dict[str, Any]
) -> TrajectoryJudgment:
    prompt = (
        "Rejudge this synthetic trajectory. Return JSON only. Score every requested dimension 1-5.\n"
        f"Effective instructions: {record.metadata.get('instructions', '')}\n"
        f"Dimensions: {json.dumps(cfg.judge.dimensions)}\n"
        f"Tools: {json.dumps(record.tools, ensure_ascii=False)}\n"
        f"Messages: {json.dumps(record.messages, ensure_ascii=False)}\n"
        f"Schema: {json.dumps(TrajectoryJudgment.model_json_schema(), ensure_ascii=False)}"
    )
    return call_structured(
        models,
        "judge",
        [
            {
                "role": "system",
                "content": "You are a strict synthetic-data quality judge.",
            },
            {"role": "user", "content": prompt},
        ],
        TrajectoryJudgment,
    )


def evaluate_record(
    record: CanonicalRecord,
    cfg: PipelineConfig,
    *,
    judge_models: dict[str, Any] | None = None,
    rejudge: bool = False,
) -> CanonicalRecord:
    if record.status == "generation_failed":
        return record
    plan = EpisodePlan.model_validate(record.episode_plan)
    report = validate_trajectory(
        reconstruct_messages(record.model_dump()),
        plan=plan,
        retrieval_transcript=record.retrieval_transcript,
        tool_schemas=record.tools,
        require_final_answer_each_turn=cfg.validation.require_final_answer_each_turn,
    )
    updated = record.model_copy(deep=True)
    updated.validation = report.model_dump()
    if not report.ok:
        updated.status = "rejected"
        return updated
    if not cfg.judge.enabled:
        updated.status = "accepted"
        updated.judgment = {"enabled": False, "skipped": True}
        return updated

    needs_judge = (
        rejudge or record.status == "quarantine" or record.judgment.get("pending")
    )
    if needs_judge:
        if judge_models is None:
            updated.status = "quarantine"
            updated.judgment = {
                "enabled": True,
                "pending": True,
                "error": "judge model unavailable",
            }
            return updated
        try:
            verdict = _judge_record(updated, cfg, judge_models)
        except Exception as exc:
            updated.status = "quarantine"
            updated.judgment = {"enabled": True, "pending": True, "error": str(exc)}
            return updated
        updated.judgment = verdict.model_dump()
    scores = updated.judgment.get("scores") or {}
    missing = sorted(set(cfg.judge.dimensions) - set(scores))
    below = {
        k: scores.get(k, 0)
        for k in cfg.judge.dimensions
        if scores.get(k, 0) < cfg.judge.min_score
    }
    rating = updated.judgment.get("rating")
    updated.status = (
        "accepted" if not missing and not below and rating == "success" else "rejected"
    )
    if updated.status == "rejected":
        updated.judgment["gate_errors"] = {
            "missing_dimensions": missing,
            "below_threshold": below,
        }
    return updated


def _write_jsonl(path: Path, records: Iterable[CanonicalRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(record.model_dump_json() + "\n")
    temporary.replace(path)


def evaluate_checkpoint(
    cfg: PipelineConfig,
    *,
    judge_models: dict[str, Any] | None = None,
    rejudge: bool = False,
) -> dict[str, Any]:
    records = load_records(cfg.resolve(cfg.paths.checkpoint))
    verify_fingerprint(records, cfg.fingerprint())
    # The checkpoint is an append-only attempt log. A retried query supersedes its
    # earlier attempt in canonical/evaluated outputs while history remains durable.
    last_index = {record.query_id: index for index, record in enumerate(records)}
    records = [
        record
        for index, record in enumerate(records)
        if last_index[record.query_id] == index
    ]
    evaluated = [
        evaluate_record(r, cfg, judge_models=judge_models, rejudge=rejudge)
        for r in records
    ]
    canonical = cfg.resolve(cfg.paths.canonical)
    output_dir = cfg.resolve(cfg.paths.output_dir)
    _write_jsonl(canonical, evaluated)
    for status in ("accepted", "rejected", "quarantine", "generation_failed"):
        _write_jsonl(
            output_dir / f"{status}.jsonl", (r for r in evaluated if r.status == status)
        )
    counts = Counter(r.status for r in evaluated)
    summary = {
        "total": len(evaluated),
        "counts": dict(counts),
        "acceptance_rate": round(counts["accepted"] / len(evaluated), 4)
        if evaluated
        else 0.0,
        "retrieval_depths": dict(
            Counter(str(r.metadata.get("retrieval_depth")) for r in evaluated)
        ),
        "turn_budgets": dict(
            Counter(str(r.metadata.get("turn_budget")) for r in evaluated)
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary
