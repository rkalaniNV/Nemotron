#!/usr/bin/env python3
"""Stage 6 — the JUDGE stage: read raw trajectories, score, write the filtered set.

Fully decoupled from generation. It takes the per-cluster raw trajectory files
(``output/sdg/raw/*.jsonl``) — or any single JSONL — scores each trajectory, and
writes the filtered training set plus a summary. Generate once, re-judge as many
times as you like (tweak the rubric / thresholds without re-running generation).

Two-tier scoring:
  1. OBJECTIVE (deterministic, always): every tool call parses + validates against
     its schema; the trajectory retrieved its gold evidence (gold_rank_log); and
     it ends on a grounded answer.
  2. RUBRIC (optional, --judge): an LLM scores the SUBJECTIVE dimensions
     (faithfulness / coherence / completeness / tool_use / user_realism) 1-5.
  A trajectory is KEPT if it passes the objective gate and (if judging) every
  rubric dimension >= --min-score.

    # after `python pipeline.py …`:
    python evaluate.py --raw-dir output/sdg/raw --out output/sdg/agentic_rag_pipeline.jsonl --judge
    python evaluate.py --input some.jsonl --out kept.jsonl          # objective-only, no LLM
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

RUBRIC_DIMS = ["faithfulness", "coherence", "completeness", "tool_use", "user_realism"]


# ── load ──────────────────────────────────────────────────────────────────────
def _load(clusters_root: Optional[Path], raw_dir: Optional[Path],
          input_file: Optional[Path]) -> List[Dict[str, Any]]:
    if clusters_root:
        files = sorted(clusters_root.glob("*/trajectories.jsonl"))   # per-cluster raw
    elif raw_dir:
        files = sorted(raw_dir.glob("*.jsonl"))
    elif input_file:
        files = [input_file]
    else:
        files = []
    rows: List[Dict[str, Any]] = []
    for fp in files:
        rows += [json.loads(l) for l in fp.open(encoding="utf-8") if l.strip()]
    return rows


# ── objective checks (deterministic) ─────────────────────────────────────────
def _objective(row: Dict[str, Any], verifier: ToolCallVerifier, require_grounded: bool) -> Dict[str, Any]:
    msgs = row.get("messages", [])
    tools = row.get("tools", [])
    n_calls, invalid = 0, 0
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
    log = row.get("gold_rank_log")
    entries = json.loads(log) if isinstance(log, str) else (log or [])
    grounded = any(e.get("gold_rank") is not None for e in entries)
    last = msgs[-1] if msgs else {}
    ends_on_answer = last.get("role") == "assistant" and not last.get("tool_calls") and bool(last.get("content"))
    ok = (invalid == 0) and (n_calls > 0) and ends_on_answer and (grounded or not require_grounded)
    return {"tool_calls": n_calls, "invalid_calls": invalid, "grounded": grounded,
            "ends_on_answer": ends_on_answer, "objective_ok": ok}


# ── rubric judge (optional) ──────────────────────────────────────────────────
def _make_judge(model: str, endpoint: str, api_key_env: str):
    sys.path.insert(0, str(Path(__file__).resolve().parent / "data_prep"))
    from question_gen import make_openai_caller
    from agentic_rag.prompts import TRAJECTORY_RUBRIC_PROMPT, JUDGE_SYSTEM_PROMPT
    from agentic_rag.messages import format_conversation_history_for_prompt, format_tools_for_prompt
    caller = make_openai_caller(model, endpoint, api_key_env, params={"temperature": 0.1, "max_tokens": 2000})

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


# ── driver ────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 6 judge: score raw trajectories -> filtered set.")
    ap.add_argument("--clusters-root", type=Path, help="clusters root; globs <root>/*/trajectories.jsonl")
    ap.add_argument("--raw-dir", type=Path, help="a flat dir of raw jsonls")
    ap.add_argument("--input", type=Path, help="a single raw jsonl")
    ap.add_argument("--out", type=Path, required=True, help="filtered training set to write")
    ap.add_argument("--judge", action="store_true", help="also run the LLM rubric (needs API key)")
    ap.add_argument("--min-score", type=int, default=3, help="min rubric score (1-5) on every dimension")
    ap.add_argument("--no-require-grounded", action="store_true", help="don't require gold retrieval")
    ap.add_argument("--model", default="nvidia/qwen/qwen3.6-35b-a3b")
    ap.add_argument("--endpoint", default="https://inference-api.nvidia.com/v1")
    ap.add_argument("--api-key-env", default="NVIDIA_API_KEY")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    rows = _load(args.clusters_root, args.raw_dir, args.input)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("[judge] no trajectories found (check --clusters-root / --raw-dir / --input).")

    verifier = ToolCallVerifier()
    judge = _make_judge(args.model, args.endpoint, args.api_key_env) if args.judge else None
    require_grounded = not args.no_require_grounded

    kept, scored_rows = [], []
    for r in rows:
        obj = _objective(r, verifier, require_grounded)
        scores = judge(r) if (judge and obj["objective_ok"]) else None   # only pay the LLM for objective passers
        keep = obj["objective_ok"] and (not args.judge or _rubric_ok(scores, args.min_score))
        r_out = {**r, "eval": {**obj, "rubric": scores, "kept": keep}}
        scored_rows.append(r_out)
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
    summary_path = args.out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n[judge] kept {len(kept)}/{len(rows)} -> {args.out}")
    print(f"[judge] scored (all + eval): {scored_path}")
    print(f"[judge] summary: {summary_path}")


def _summary(scored: List[Dict[str, Any]], kept: List[Dict[str, Any]], args) -> Dict[str, Any]:
    n = len(scored)
    obj_pass = sum(1 for r in scored if r["eval"]["objective_ok"])
    per_cluster = Counter(r.get("cluster_id", "?") for r in kept)
    hops = Counter(r.get("hops_taken", 0) for r in kept)
    out: Dict[str, Any] = {
        "total": n, "objective_pass": obj_pass, "kept": len(kept),
        "keep_rate": round(len(kept) / n, 3) if n else None,
        "grounded_rate": round(sum(r["eval"]["grounded"] for r in scored) / n, 3) if n else None,
        "kept_per_cluster": dict(per_cluster),
        "kept_hops_distribution": dict(sorted(hops.items())),
        "judged": bool(args.judge),
    }
    if args.judge:
        rub = [r["eval"]["rubric"] for r in scored if r["eval"].get("rubric")]
        out["mean_rubric"] = {d: round(sum(s[d] for s in rub) / len(rub), 2) for d in RUBRIC_DIMS} if rub else {}
    return out


if __name__ == "__main__":
    main()
