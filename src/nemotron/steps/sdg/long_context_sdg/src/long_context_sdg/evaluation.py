"""Decoupled deterministic validation, rejudging, partitioning, and reporting."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .config import PipelineConfig
from .llm import call_structured
from .records import load_records
from .schemas import CanonicalRecord, EpisodeSpec, TrajectoryJudgment
from .validation import reconstruct_messages, validate_trajectory


def _judge_record(record: CanonicalRecord, cfg: PipelineConfig, models: dict[str, Any]) -> TrajectoryJudgment:
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
    spec = EpisodeSpec.model_validate(record.episode_spec)
    report = validate_trajectory(
        reconstruct_messages(record.model_dump()),
        spec=spec,
        retrieval_transcript=record.retrieval_transcript,
        tool_call_attempts=record.tool_call_attempts,
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

    needs_judge = rejudge or record.status == "quarantine" or record.judgment.get("pending")
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
    below = {k: scores.get(k, 0) for k in cfg.judge.dimensions if scores.get(k, 0) < cfg.judge.min_score}
    rating = updated.judgment.get("rating")
    updated.status = "accepted" if not missing and not below and rating == "success" else "rejected"
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


def _numeric_summary(values: Iterable[int]) -> dict[str, float | int]:
    ordered = sorted(values)
    if not ordered:
        return {"count": 0, "min": 0, "median": 0, "mean": 0.0, "max": 0}
    midpoint = len(ordered) // 2
    median = ordered[midpoint] if len(ordered) % 2 else (ordered[midpoint - 1] + ordered[midpoint]) / 2
    return {
        "count": len(ordered),
        "min": ordered[0],
        "median": median,
        "mean": round(sum(ordered) / len(ordered), 4),
        "max": ordered[-1],
    }


def evaluate_generated(
    cfg: PipelineConfig,
    *,
    judge_models: dict[str, Any] | None = None,
    rejudge: bool = False,
) -> dict[str, Any]:
    records = load_records(cfg.resolve(cfg.paths.generated))
    incompatible = sorted(
        {record.config_fingerprint for record in records if record.config_fingerprint != cfg.fingerprint()}
    )
    if incompatible:
        raise ValueError("generated records use incompatible configuration fingerprint(s): " + ", ".join(incompatible))
    query_ids = [record.query_id for record in records]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("generated records contain duplicate query IDs")
    evaluated = [evaluate_record(r, cfg, judge_models=judge_models, rejudge=rejudge) for r in records]
    canonical = cfg.resolve(cfg.paths.canonical)
    output_dir = cfg.resolve(cfg.paths.output_dir)
    _write_jsonl(canonical, evaluated)
    for status in ("accepted", "rejected", "quarantine", "generation_failed"):
        _write_jsonl(output_dir / f"{status}.jsonl", (r for r in evaluated if r.status == status))
    counts = Counter(r.status for r in evaluated)
    turn_budgets = [
        int(record.metadata["turn_budget"]) for record in evaluated if record.metadata.get("turn_budget") is not None
    ]
    successful_retrievals = [
        int(record.metadata["successful_retrieval_calls"])
        for record in evaluated
        if record.metadata.get("successful_retrieval_calls") is not None
    ]
    tool_calls = [len(record.tool_call_attempts) for record in evaluated if record.episode_spec]
    retrieval_calls = [len(record.retrieval_transcript) for record in evaluated if record.episode_spec]
    low_gain_calls = [
        sum(bool(item.get("low_gain")) for item in record.retrieval_transcript)
        for record in evaluated
        if record.episode_spec
    ]
    rejected_redundant_calls = [
        sum(
            "lexically similar" in str(item.get("error", ""))
            for item in record.metadata.get("rejected_tool_calls", [])
        )
        for record in evaluated
        if record.episode_spec
    ]
    summary = {
        "total": len(evaluated),
        "counts": dict(counts),
        "acceptance_rate": round(counts["accepted"] / len(evaluated), 4) if evaluated else 0.0,
        "turn_budgets": dict(Counter(str(r.metadata.get("turn_budget")) for r in evaluated)),
        "turn_budget_summary": _numeric_summary(turn_budgets),
        "successful_retrieval_summary": _numeric_summary(successful_retrievals),
        "retrieval_call_summary": _numeric_summary(retrieval_calls),
        "low_gain_retrieval_summary": _numeric_summary(low_gain_calls),
        "rejected_redundant_retrieval_summary": _numeric_summary(rejected_redundant_calls),
        "tool_call_summary": _numeric_summary(tool_calls),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
