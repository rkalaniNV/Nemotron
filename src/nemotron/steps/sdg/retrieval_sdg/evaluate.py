#!/usr/bin/env python3
"""Decoupled judge/filter stage: raw trajectories -> filtered SFT set + summary.

Fully decoupled from generation — re-judge as many times as you like (tweak the
rubric / thresholds) without re-running the pipeline. Two-tier scoring:

  1. OBJECTIVE (deterministic): every tool call parses + validates against its
     schema; the trajectory made at least one retrieval call; and it ends on a
     grounded (tool-free) assistant answer.
  2. RUBRIC (optional, --judge): an LLM scores the subjective dimensions
     (faithfulness / coherence / completeness / tool_use / user_realism) 1-5.
  KEPT if it passes the objective gate and (when judging) every dim >= --min-score.

    python evaluate.py --input output/sdg/retrieval_sdg.raw.jsonl --out output/sdg/retrieval_sdg.jsonl --judge
    python evaluate.py --input raw.jsonl --out kept.jsonl              # objective-only, no LLM
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from retrieval_sdg.conversation.verifiers import ToolCallVerifier

RUBRIC_DIMS = ["faithfulness", "coherence", "completeness", "tool_use", "user_realism"]


def _load(input_file: Optional[Path], raw_dir: Optional[Path]) -> List[Dict[str, Any]]:
    files = [input_file] if input_file else (sorted(raw_dir.glob("*.jsonl")) if raw_dir else [])
    rows: List[Dict[str, Any]] = []
    for fp in files:
        rows += [json.loads(l) for l in fp.open(encoding="utf-8") if l.strip()]
    return rows


def _objective(row: Dict[str, Any], verifier: ToolCallVerifier) -> Dict[str, Any]:
    msgs = row.get("messages", [])
    tools = row.get("tools", [])
    n_calls, invalid, retrievals = 0, 0, 0
    for m in msgs:
        if m.get("role") != "assistant":
            continue
        for tc in (m.get("tool_calls") or []):
            n_calls += 1
            try:
                json.loads(tc.get("function", {}).get("arguments", "{}"))
                ok, _ = verifier.verify_single(tc, tools)
            except (json.JSONDecodeError, TypeError):
                ok = False
            invalid += int(not ok)
    # a retrieval happened if any tool message carries a results payload
    for m in msgs:
        if m.get("role") == "tool" and '"results"' in (m.get("content") or ""):
            retrievals += 1
    last = msgs[-1] if msgs else {}
    ends_on_answer = last.get("role") == "assistant" and not last.get("tool_calls") and bool(last.get("content"))
    ok = (invalid == 0) and (n_calls > 0) and (retrievals > 0) and ends_on_answer
    return {"tool_calls": n_calls, "invalid_calls": invalid, "retrievals": retrievals,
            "ends_on_answer": ends_on_answer, "objective_ok": ok}


def _make_judge(model: str, endpoint: str, api_key_env: str):
    from retrieval_sdg.core.caller import make_openai_caller
    from retrieval_sdg.core.messages import format_conversation_history_for_prompt, format_tools_for_prompt
    from retrieval_sdg.conversation.prompts import JUDGE_SYSTEM_PROMPT, TRAJECTORY_RUBRIC_PROMPT
    caller = make_openai_caller(model, endpoint, api_key_env,
                                params={"temperature": 0.1, "max_tokens": 2000})

    def score(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        prompt = TRAJECTORY_RUBRIC_PROMPT.format(
            tools=format_tools_for_prompt(row.get("tools", [])),
            conversation=format_conversation_history_for_prompt(row.get("messages", [])))
        text = caller(JUDGE_SYSTEM_PROMPT, prompt)
        a, b = text.find("{"), text.rfind("}")
        if a < 0 or b <= a:
            return None
        try:
            d = json.loads(text[a:b + 1])
        except json.JSONDecodeError:
            return None
        return {k: int(d.get(k, 0)) for k in RUBRIC_DIMS} | {"notes": str(d.get("notes", ""))[:200]}

    return score


def _rubric_ok(scores: Optional[Dict[str, Any]], min_score: int) -> bool:
    return bool(scores) and all(scores.get(k, 0) >= min_score for k in RUBRIC_DIMS)


def main() -> None:
    ap = argparse.ArgumentParser(description="Judge raw trajectories -> filtered SFT set.")
    ap.add_argument("--input", type=Path, help="a single raw jsonl")
    ap.add_argument("--raw-dir", type=Path, help="a dir of raw jsonls")
    ap.add_argument("--out", type=Path, required=True, help="filtered training set to write")
    ap.add_argument("--judge", action="store_true", help="also run the LLM rubric (needs API key)")
    ap.add_argument("--min-score", type=int, default=3, help="min rubric score (1-5) on every dimension")
    ap.add_argument("--model", default="nvidia/openai/gpt-oss-120b")  # reliable structured JSON
    ap.add_argument("--endpoint", default="https://inference-api.nvidia.com/v1")
    ap.add_argument("--api-key-env", default="NVIDIA_API_KEY")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    rows = _load(args.input, args.raw_dir)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("[judge] no trajectories found (check --input / --raw-dir).")

    verifier = ToolCallVerifier()
    judge = _make_judge(args.model, args.endpoint, args.api_key_env) if args.judge else None

    kept, scored_rows = [], []
    for r in rows:
        obj = _objective(r, verifier)
        scores = judge(r) if (judge and obj["objective_ok"]) else None
        keep = obj["objective_ok"] and (not args.judge or _rubric_ok(scores, args.min_score))
        scored_rows.append({**r, "eval": {**obj, "rubric": scores, "kept": keep}})
        if keep:
            kept.append({k: v for k, v in r.items() if k != "eval"})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    scored_path = args.out.with_suffix(".scored.jsonl")
    with scored_path.open("w", encoding="utf-8") as f:
        for r in scored_rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    summary = _summary(scored_rows, kept, args)
    args.out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n[judge] kept {len(kept)}/{len(rows)} -> {args.out}")


def _summary(scored: List[Dict[str, Any]], kept: List[Dict[str, Any]], args) -> Dict[str, Any]:
    n = len(scored)
    out: Dict[str, Any] = {
        "total": n,
        "objective_pass": sum(1 for r in scored if r["eval"]["objective_ok"]),
        "kept": len(kept),
        "keep_rate": round(len(kept) / n, 3) if n else None,
        "kept_per_cluster": dict(Counter(r.get("cluster_id", "?") for r in kept)),
        "kept_kinds": dict(Counter(r.get("kind", "?") for r in kept)),
        "kept_hops_distribution": dict(sorted(Counter(r.get("hops_taken", 0) for r in kept).items())),
        "judged": bool(args.judge),
    }
    if args.judge:
        rub = [r["eval"]["rubric"] for r in scored if r["eval"].get("rubric")]
        out["mean_rubric"] = {d: round(sum(s[d] for s in rub) / len(rub), 2) for d in RUBRIC_DIMS} if rub else {}
    return out


if __name__ == "__main__":
    main()
