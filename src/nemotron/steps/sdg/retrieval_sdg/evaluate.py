#!/usr/bin/env python3
"""Decoupled judge/filter stage: raw trajectories -> filtered SFT set + summary.

Fully decoupled from generation — re-judge as many times as you like (tweak the
rubric / thresholds) without re-running the pipeline. Two-tier scoring:

  1. OBJECTIVE (deterministic): every tool call parses + validates against its
     schema; the trajectory made at least one retrieval call; and it ends on a
     grounded (tool-free) assistant answer.
  2. DEFECT GATE (optional, --judge): an LLM screens for binary train-harmful
     defects (unsupported claims / no real research / incoherent / request
     unresolved / user out of character) and a soft 1-5 quality score.
  KEPT if it passes the objective gate and (when judging) NO defect fired and
  quality >= --min-quality.

    python evaluate.py --input output/sdg/retrieval_sdg.raw.jsonl --out output/sdg/retrieval_sdg.jsonl --judge
    python evaluate.py --input raw.jsonl --out kept.jsonl              # objective-only, no LLM
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from retrieval_sdg.conversation.verifiers import ToolCallVerifier

# binary train-harmful defects; a row is rejected if ANY fire (see prompts.TRAJECTORY_RUBRIC_PROMPT)
DISQUALIFIERS = ["unsupported_claims", "no_real_research", "incoherent",
                 "request_unresolved", "user_out_of_character"]


def _load(input_file: Optional[Path], raw_dir: Optional[Path]) -> List[Dict[str, Any]]:
    files = [input_file] if input_file else (sorted(raw_dir.glob("*.jsonl")) if raw_dir else [])
    rows: List[Dict[str, Any]] = []
    for fp in files:
        rows += [json.loads(l) for l in fp.open(encoding="utf-8") if l.strip()]
    return rows


def _char_ngrams(text: str, n: int = 12) -> set:
    t = " ".join((text or "").split()).lower()
    return {t[i:i + n] for i in range(len(t) - n + 1)} if len(t) >= n else set()


def _evidence_overlap(row: Dict[str, Any], n: int = 12) -> float:
    """Deterministic, language/domain-agnostic grounding proxy: fraction of each
    answer's character n-grams that also appear in the RETRIEVED chunk text.
    High => the answer is drawn from the evidence; low => it was written from the
    model's own knowledge (drift/hallucination). No regex, no language assumptions."""
    evidence = _char_ngrams(" ".join(m.get("content", "") for m in row.get("messages", [])
                                     if m.get("role") == "tool"), n)
    if not evidence:
        return 0.0
    scores = []
    for m in row.get("messages", []):
        if m.get("role") == "assistant" and not m.get("tool_calls") and (m.get("content") or "").strip():
            ag = _char_ngrams(m["content"], n)
            if ag:
                scores.append(len(ag & evidence) / len(ag))
    return round(min(scores), 3) if scores else 0.0   # weakest answer sets the grounding


# chunk-id token as emitted by the retrieval client's content-hash fallback (h + hex).
# Domain/language-agnostic: it only matches the id SHAPE, never any word in the text.
_ID = re.compile(r"h[0-9a-f]{12}")


def _retrieved_ids(msgs: List[Dict[str, Any]]) -> set:
    ids: set = set()
    for m in msgs:
        if m.get("role") == "tool":
            ids |= set(_ID.findall(m.get("content", "") or ""))
    return ids


def _cited_ids(msgs: List[Dict[str, Any]]) -> set:
    ids: set = set()
    for m in msgs:
        if m.get("role") == "assistant" and not m.get("tool_calls"):
            ids |= set(_ID.findall(m.get("content", "") or ""))
    return ids


def _citation_integrity(row: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic, exact grounding check the LLM judge cannot do reliably: every
    chunk id the assistant CITES must have actually been RETRIEVED in this conversation.
    Any cited-but-never-retrieved id is a fabricated citation (a hard reject)."""
    msgs = row.get("messages", [])
    cited, retrieved = _cited_ids(msgs), _retrieved_ids(msgs)
    fabricated = sorted(cited - retrieved)
    return {"cited_ids": len(cited), "fabricated_ids": len(fabricated),
            "fabricated": fabricated[:10], "citation_ok": not fabricated}


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
    cite = _citation_integrity(row)
    ok = (invalid == 0) and (n_calls > 0) and (retrievals > 0) and ends_on_answer and cite["citation_ok"]
    return {"tool_calls": n_calls, "invalid_calls": invalid, "retrievals": retrievals,
            "ends_on_answer": ends_on_answer, **cite, "objective_ok": ok}


def _judge_from_config(config_path):
    """Resolve the judge caller from pipeline.yaml: the judge_model alias's model +
    its provider's endpoint + the env-var name that holds the key."""
    from omegaconf import OmegaConf
    cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    models = {m.get("alias"): m for m in cfg.get("models", [])}
    provs = {p.get("name"): p for p in cfg.get("providers", [])}
    jm = models.get("judge_model", {})
    prov = provs.get(jm.get("provider"), {})
    return jm.get("model", ""), prov.get("endpoint", ""), prov.get("api_key", "NVIDIA_API_KEY")


def _make_judge(model: str, endpoint: str, api_key_env: str):
    from retrieval_sdg.core.caller import make_openai_caller
    from retrieval_sdg.core.messages import format_history_compact, format_tools_for_prompt
    from retrieval_sdg.conversation.prompts import JUDGE_SYSTEM_PROMPT, TRAJECTORY_RUBRIC_PROMPT
    caller = make_openai_caller(model, endpoint, api_key_env,
                                params={"temperature": 0.1, "max_tokens": 2000})

    def score(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # bound the conversation so it fits the judge model's context (some are 32k);
        # tool outputs are snipped, recent turns kept, so the judge still sees the flow.
        # wide tool snippets so the judge SEES the retrieved evidence (a tight snippet
        # truncates the chunk text/ids and makes faithful answers look unsupported).
        prompt = TRAJECTORY_RUBRIC_PROMPT.format(
            tools=format_tools_for_prompt(row.get("tools", [])),
            conversation=format_history_compact(row.get("messages", []), max_chars=180000, tool_snippet=4000))
        text = caller(JUDGE_SYSTEM_PROMPT, prompt)
        a, b = text.find("{"), text.rfind("}")
        if a < 0 or b <= a:
            return None
        try:
            d = json.loads(text[a:b + 1])
        except json.JSONDecodeError:
            return None
        dq = d.get("disqualifiers", {}) if isinstance(d.get("disqualifiers"), dict) else {}
        return {"disqualifiers": {k: bool(dq.get(k, False)) for k in DISQUALIFIERS},
                "quality": int(d.get("quality", 0) or 0), "notes": str(d.get("notes", ""))[:200]}

    return score


def _rubric_ok(scores: Optional[Dict[str, Any]], min_quality: int) -> bool:
    """Keep only if the judge returned a verdict, NO disqualifier fired, and the soft
    quality score clears the (low, reporting-oriented) floor."""
    if not scores:
        return False
    if any(scores.get("disqualifiers", {}).get(k) for k in DISQUALIFIERS):
        return False
    return scores.get("quality", 0) >= min_quality


def main() -> None:
    ap = argparse.ArgumentParser(description="Judge raw trajectories -> filtered SFT set.")
    ap.add_argument("--input", type=Path, help="a single raw jsonl")
    ap.add_argument("--raw-dir", type=Path, help="a dir of raw jsonls")
    ap.add_argument("--out", type=Path, required=True, help="filtered training set to write")
    ap.add_argument("--judge", action="store_true", help="also run the LLM defect gate (needs API key)")
    ap.add_argument("--min-quality", type=int, default=3,
                    help="soft quality floor (1-5); the gate is the defect flags, this just drops weak-but-clean rows")
    ap.add_argument("--min-overlap", type=float, default=0.0,
                    help="min deterministic answer<->evidence char-ngram overlap (0=report only, don't gate)")
    # judge model/endpoint: taken from --config's judge_model by default (so it matches
    # generation), or set explicitly. Explicit flags win; fall back to NVIDIA hosted.
    ap.add_argument("--config", type=Path, default=None,
                    help="pipeline.yaml — read the judge_model's model/endpoint/key from it")
    ap.add_argument("--model", default=None)          # reliable structured JSON
    ap.add_argument("--endpoint", default=None)
    ap.add_argument("--api-key-env", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    model, endpoint, api_key_env = args.model, args.endpoint, args.api_key_env
    if args.config and not (model and endpoint and api_key_env):
        cfg_model, cfg_endpoint, cfg_key = _judge_from_config(args.config)   # judge_model + its provider
        model = model or cfg_model
        endpoint = endpoint or cfg_endpoint
        api_key_env = api_key_env or cfg_key
    model = model or "nvidia/openai/gpt-oss-120b"
    endpoint = endpoint or "https://inference-api.nvidia.com/v1"
    api_key_env = api_key_env or "NVIDIA_API_KEY"

    rows = _load(args.input, args.raw_dir)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("[judge] no trajectories found (check --input / --raw-dir).")

    verifier = ToolCallVerifier()
    judge = _make_judge(model, endpoint, api_key_env) if args.judge else None

    kept, scored_rows = [], []
    for r in rows:
        obj = _objective(r, verifier)
        overlap = _evidence_overlap(r)                        # deterministic grounding proxy
        scores = judge(r) if (judge and obj["objective_ok"]) else None
        keep = (obj["objective_ok"] and (overlap >= args.min_overlap)
                and (not args.judge or _rubric_ok(scores, args.min_quality)))
        scored_rows.append({**r, "eval": {**obj, "grounding_overlap": overlap, "rubric": scores, "kept": keep}})
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


def _role_seq(msgs: List[Dict[str, Any]]) -> str:
    return "".join((m.get("role") or "?")[0].upper() for m in msgs)   # S/U/A/T


def _ctx_tokens(msgs: List[Dict[str, Any]]) -> int:
    return sum(len(m.get("content", "") or "") + len(m.get("reasoning_content", "") or "") for m in msgs) // 4


def _dist(vals: List[int]) -> Dict[str, Any]:
    v = sorted(vals)
    return {"min": v[0], "p25": v[len(v) // 4], "median": v[len(v) // 2],
            "p75": v[(3 * len(v)) // 4], "max": v[-1], "mean": round(sum(v) / len(v))}


def _summary(scored: List[Dict[str, Any]], kept: List[Dict[str, Any]], args) -> Dict[str, Any]:
    n = len(scored)
    out: Dict[str, Any] = {
        "total": n,
        "objective_pass": sum(1 for r in scored if r["eval"]["objective_ok"]),
        "citation_fabrication_rows": sum(1 for r in scored if not r["eval"].get("citation_ok", True)),
        "cited_ids_total": sum(r["eval"].get("cited_ids", 0) for r in scored),
        "kept": len(kept),
        "keep_rate": round(len(kept) / n, 3) if n else None,
        "kept_hops_distribution": dict(sorted(Counter(r.get("hops_taken", 0) for r in kept).items())),
        "judged": bool(args.judge),
    }
    if kept:
        turns = [sum(1 for m in r["messages"] if m.get("role") == "user") for r in kept]
        out["turns_distribution"] = dict(sorted(Counter(turns).items()))
        out["top_flow_patterns"] = [{"pattern": p, "count": c}
                                    for p, c in Counter(_role_seq(r["messages"]) for r in kept).most_common(5)]
        out["context_length_tokens"] = _dist([_ctx_tokens(r["messages"]) for r in kept])
    ov = sorted(r["eval"]["grounding_overlap"] for r in scored)
    if ov:
        out["grounding_overlap"] = {"mean": round(sum(ov) / len(ov), 3),
                                    "median": ov[len(ov) // 2],
                                    "p10": ov[len(ov) // 10], "min": ov[0], "max": ov[-1]}
    if args.judge:
        rub = [r["eval"]["rubric"] for r in scored if r["eval"].get("rubric")]
        j = len(rub)
        # how often each defect fired among judged rows (lower = cleaner data)
        out["disqualifier_rate"] = ({k: round(sum(bool(s["disqualifiers"].get(k)) for s in rub) / j, 3)
                                     for k in DISQUALIFIERS} if j else {})
        out["clean_rate"] = round(sum(not any(s["disqualifiers"].values()) for s in rub) / j, 3) if j else None
        out["quality_distribution"] = dict(sorted(Counter(s["quality"] for s in rub).items())) if j else {}
        out["mean_quality"] = round(sum(s["quality"] for s in rub) / j, 2) if j else None
    return out


if __name__ == "__main__":
    main()
