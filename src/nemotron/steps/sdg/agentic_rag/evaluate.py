#!/usr/bin/env python3
"""Stage 6 — standalone evaluation over a generated trajectory dataset.

Runs two checks over an output JSONL (rows of ``{messages, tools, ...}``):

  1. tool-call validity (always, offline)  — every assistant tool_call is parsed
     and validated against the row's tool schemas (the modular ToolCallVerifier).
     Reports the parseable/valid rate — a hard correctness signal for SFT data.
  2. LLM-as-judge (optional, needs an API key) — rates each conversation
     success/failure with the same trajectory rubric used inline.

Also aggregates grounding + shape stats: gold-retrieval rate (from
``gold_rank_log``), hop distribution, and the conversation-variant mix.

    python evaluate.py --input output/sdg/agentic_rag_pipeline.jsonl
    python evaluate.py --input output/sdg/agentic_rag_pipeline.jsonl --judge \
        --model nvidia/openai/gpt-oss-120b --api-key-env NVIDIA_API_KEY
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agentic_rag.verifiers import ToolCallVerifier


def _tool_call_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    verifier = ToolCallVerifier()
    total, valid, unparseable = 0, 0, 0
    rows_with_bad = 0
    for r in rows:
        tools = r.get("tools", [])
        row_bad = False
        for m in r.get("messages", []):
            if m.get("role") != "assistant":
                continue
            for tc in (m.get("tool_calls") or []):
                total += 1
                try:
                    json.loads(tc.get("function", {}).get("arguments", "{}"))
                except (json.JSONDecodeError, TypeError):
                    unparseable += 1
                    row_bad = True
                    continue
                ok, _ = verifier.verify_single(tc, tools)
                valid += int(ok)
                row_bad = row_bad or not ok
        rows_with_bad += int(row_bad)
    return {"tool_calls": total, "valid": valid, "unparseable": unparseable,
            "valid_rate": round(valid / total, 4) if total else None,
            "rows_with_invalid_calls": rows_with_bad}


def _grounding_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    grounded, ranks = 0, []
    for r in rows:
        log = r.get("gold_rank_log")
        entries = json.loads(log) if isinstance(log, str) else (log or [])
        hit = [e.get("gold_rank") for e in entries if e.get("gold_rank") is not None]
        if hit:
            grounded += 1
            ranks.append(min(hit))
    n = len(rows)
    return {"rows": n,
            "gold_retrieved_rate": round(grounded / n, 4) if n else None,
            "mean_best_gold_rank": round(sum(ranks) / len(ranks), 2) if ranks else None}


def _shape_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    hops = Counter(r.get("hops_taken", 0) for r in rows)
    variants = Counter(r.get("conversation_variant", "?") for r in rows)
    clusters = Counter(r.get("cluster_id", "?") for r in rows)
    return {"hops_distribution": dict(sorted(hops.items())),
            "variant_distribution": dict(variants),
            "n_clusters_represented": len(clusters)}


def _run_judge(rows, model, endpoint, api_key_env) -> Dict[str, Any]:
    sys.path.insert(0, str(Path(__file__).resolve().parent / "data_prep"))
    from question_gen import make_openai_caller
    from agentic_rag.judges import parse_judge_response
    from agentic_rag.messages import format_conversation_history_for_prompt, format_tools_for_prompt
    from agentic_rag.prompts import TRAJECTORY_JUDGE_PROMPT

    caller = make_openai_caller(model, endpoint, api_key_env, params={"temperature": 0.1, "max_tokens": 512})
    passed = 0
    for r in rows:
        prompt = TRAJECTORY_JUDGE_PROMPT.format(
            tools=format_tools_for_prompt(r.get("tools", [])),
            conversation=format_conversation_history_for_prompt(r.get("messages", [])))
        _, _, ok = parse_judge_response(caller("", prompt))
        passed += int(ok)
    return {"judged": len(rows), "passed": passed,
            "pass_rate": round(passed / len(rows), 4) if rows else None}


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 6: evaluate a generated trajectory dataset.")
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--judge", action="store_true", help="also run the LLM-as-judge pass (needs API)")
    ap.add_argument("--model", default="nvidia/openai/gpt-oss-120b")
    ap.add_argument("--endpoint", default="https://inference-api.nvidia.com/v1")
    ap.add_argument("--api-key-env", default="NVIDIA_API_KEY")
    ap.add_argument("--limit", type=int, default=None, help="evaluate at most N rows")
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.input.open(encoding="utf-8") if l.strip()]
    if args.limit:
        rows = rows[: args.limit]

    report = {
        "input": str(args.input),
        "n_rows": len(rows),
        "tool_call_validity": _tool_call_stats(rows),
        "grounding": _grounding_stats(rows),
        "shape": _shape_stats(rows),
    }
    if args.judge:
        report["llm_judge"] = _run_judge(rows, args.model, args.endpoint, args.api_key_env)

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
