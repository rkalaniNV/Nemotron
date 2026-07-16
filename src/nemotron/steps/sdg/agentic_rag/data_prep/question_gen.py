#!/usr/bin/env python3
"""Stage 2 — per-cluster question generation (runs on ONE cluster at a time).

For a single cluster's document set this module produces the two things Stage 3+
needs:

  1. retrieval chunks  (``chunks.jsonl``)  — small size-bounded chunks tagged with
     ``doc_id``; the orchestrator builds the cluster's search index over these.
  2. seed queries      (``queries.jsonl``) — 2-5 questions PER document shard
     (a knob), spanning a difficulty spectrum from half-baked to complex
     multi-step, each carrying gold provenance (which sections answer it) so
     retrieval grounding can be scored downstream.

Sharding: if a document is longer than the question-generator model's usable
window (``shard_max_chars``, derived from its max length), it is split into
shards and each shard seeds its own questions — so long documents are covered,
not truncated.

The LLM is injected as a ``caller(system, user) -> str`` so this module is
runtime-agnostic and unit-testable; ``make_openai_caller`` wires an
OpenAI-compatible endpoint (e.g. the NVIDIA inference API) from config.

Usage (standalone, one cluster):
    python question_gen.py --cluster-dir ../data/clusters/c000 \
        --profile indian_statute --model nvidia/openai/gpt-oss-120b \
        --endpoint https://inference-api.nvidia.com/v1 --api-key-env NVIDIA_API_KEY \
        --n-queries 4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # prefer local package

from chunk_document import PROFILES, ChunkProfile, build_chunks, recursive_split  # noqa: E402

Caller = Callable[[str, str], str]

# The difficulty spectrum the user asked for: half-baked -> crisp -> complex.
DIFFICULTY_LEVELS: List[Dict[str, str]] = [
    {"level": "half_baked",
     "desc": "vague, underspecified, missing a key detail — a good assistant should ASK to clarify first",
     "variant": "multi_turn"},
    {"level": "simple",
     "desc": "a direct single-fact question answerable from one passage",
     "variant": "single_turn"},
    {"level": "crisp",
     "desc": "precise and well-scoped, naming the specific thing asked about",
     "variant": "multi_step"},
    {"level": "complex_multistep",
     "desc": "requires combining several passages / cross-references; forces multiple searches to answer",
     "variant": "multi_step"},
]


@dataclass
class QGenConfig:
    profile: str = "plain"
    chunk_max_chars: int = 2000
    chunk_overlap: int = 0
    shard_max_chars: int = 8000        # generator window (≈ max_tokens*4); shard if longer
    n_queries: int = 5                 # queries PER DOCUMENT (spread across its shards)
    max_docs: Optional[int] = None     # cap docs/cluster for cost control
    ref_field: str = "refs_article"    # metadata key holding cross-references (grounding)


# ── OpenAI-compatible caller (swappable) ─────────────────────────────────────
def make_openai_caller(model: str, endpoint: str, api_key_env: str,
                       params: Optional[Dict] = None) -> Caller:
    """Return a ``caller(system, user) -> str`` backed by an OpenAI-compatible
    endpoint. Kept tiny and dependency-light so Stage 2 runs without the DD
    runtime."""
    from openai import OpenAI
    api_key = os.environ.get(api_key_env, "")
    if not api_key:
        raise SystemExit(f"[question_gen] env var {api_key_env} is not set.")
    client = OpenAI(base_url=endpoint, api_key=api_key)
    # generous max_tokens: reasoning models spend tokens thinking before emitting
    # the JSON, so a low cap yields empty output.
    p = {"temperature": 0.9, "top_p": 0.95, "max_tokens": 6000, **(params or {})}

    def _call(system: str, user: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            **p)
        msg = resp.choices[0].message
        out = (msg.content or "").strip()
        if not out:  # reasoning models can return only reasoning_content
            out = (getattr(msg, "reasoning_content", "") or "").strip()
        return out

    return _call


# ── prompts ──────────────────────────────────────────────────────────────────
QGEN_SYSTEM = """You write realistic user questions that will be answered by an AI research assistant with access to a knowledge-base search tool over a specific document collection.
- Ground every question in the provided source text; a diligent researcher must be able to answer it from that collection.
- Do NOT quote the source verbatim; phrase questions the way a real person would ask them.
- Return ONLY a JSON array, no prose."""

QGEN_USER = """<SOURCE_DOCUMENT id="{doc_id}">
{shard_text}
</SOURCE_DOCUMENT>

<RELATED_SECTIONS_IN_THIS_COLLECTION>
{neighbors}
</RELATED_SECTIONS_IN_THIS_COLLECTION>

Write exactly {n} distinct user questions grounded in the SOURCE_DOCUMENT above, spanning this difficulty spectrum (assign each question one level):
{levels}

For "complex_multistep", prefer questions whose answer requires synthesising the source with the related sections listed above (genuine multi-hop).

Return a JSON array of objects, each:
{{"query": "<the question a real person would type>",
  "level": "<one of: {level_names}>",
  "target_sections": ["<ids of sections this question is grounded in, from the source or related list>"],
  "rationale": "<one line: why answering needs research>"}}"""


# ── chunking (retrieval index inputs) ────────────────────────────────────────
def chunk_cluster(docs: Sequence[Dict], cfg: QGenConfig) -> List[Dict]:
    prof: ChunkProfile = PROFILES[cfg.profile]
    prof = ChunkProfile(**{**prof.__dict__})           # copy so overrides don't leak
    prof.max_chars = cfg.chunk_max_chars
    prof.overlap = cfg.chunk_overlap
    from dataclasses import asdict
    out: List[Dict] = []
    for d in docs:
        doc_id = str(d.get("doc_id", ""))
        chunks = build_chunks(d.get("text", ""), prof, doc_id=doc_id, id_prefix=f"{doc_id}__")
        for c in chunks:
            rec = asdict(c)
            # carry a section_title from the doc's section metadata when present
            if not rec.get("section_title") and d.get("section_title"):
                rec["section_title"] = d["section_title"]
            out.append(rec)
    return out


# ── sharding (generator windows) ─────────────────────────────────────────────
def shard_document(text: str, shard_max_chars: int) -> List[str]:
    if len(text) <= shard_max_chars:
        return [text] if text.strip() else []
    return recursive_split(text, shard_max_chars, ["\n\n", "\n", ". ", " ", ""])


# ── grounding helpers (generic — section ids come from the config-driven
# chunker's profile, NOT any domain-specific regex) ──────────────────────────
def _section_maps(chunks: Sequence[Dict]):
    """From the (profile-chunked) corpus, build:
      cluster_sections: section_id -> short title (valid ids for gold validation)
      doc_sections    : doc_id -> [section_ids it owns]  (document-level gold)"""
    cluster_sections: Dict[str, str] = {}
    doc_sections: Dict[str, List[str]] = {}
    for c in chunks:
        sid = str(c.get("section_id", ""))
        did = str(c.get("doc_id", ""))
        if not sid:
            continue
        cluster_sections.setdefault(sid, (c.get("section_title") or (c.get("text") or "")[:80]))
        lst = doc_sections.setdefault(did, [])
        if sid not in lst:
            lst.append(sid)
    return cluster_sections, doc_sections


def _neighbors_blurb(cluster_sections: Dict[str, str], own: set, limit: int = 6) -> str:
    """A few OTHER sections in this cluster a multi-hop question could reach."""
    others = [s for s in cluster_sections if s not in own][:limit]
    if not others:
        return "(none listed)"
    return "\n".join(f"- section {s}: {cluster_sections[s][:80]}" for s in others)


def _distribute(n: int, k: int, rng: Optional[random.Random] = None) -> List[int]:
    """Spread a per-DOCUMENT quota of n queries across k shards; total == n.

    - k <= n : even split (base per shard), the remainder handed to random shards.
    - k >  n : more shards than queries → pick n random shards, 1 query each
               (so we sample across the document, not just its opening shards).
    """
    if k <= 0:
        return []
    n = max(0, n)
    rng = rng or random.Random(0)
    counts = [0] * k
    idx = list(range(k))
    rng.shuffle(idx)
    if k <= n:
        base, rem = divmod(n, k)
        counts = [base] * k
        for i in idx[:rem]:
            counts[i] += 1
    else:
        for i in idx[:n]:
            counts[i] = 1
    return counts


# ── query generation ─────────────────────────────────────────────────────────
def _levels_for(n: int) -> List[Dict[str, str]]:
    if n >= len(DIFFICULTY_LEVELS):
        base = list(DIFFICULTY_LEVELS)
        while len(base) < n:                       # pad extra slots with complex ones
            base.append(DIFFICULTY_LEVELS[-1])
        return base[:n]
    # n < 4: bias toward the more useful (crisp / complex) end but keep spread
    picks = [DIFFICULTY_LEVELS[0], DIFFICULTY_LEVELS[2], DIFFICULTY_LEVELS[3]]
    return picks[:n]


def _parse_queries(text: str) -> List[Dict]:
    a, b = text.find("["), text.rfind("]")
    if a < 0 or b <= a:
        return []
    try:
        arr = json.loads(text[a:b + 1])
    except json.JSONDecodeError:
        return []
    # keep only well-formed objects: a reasoning-model fallback can yield a bare
    # list of draft strings, which must not crash or pollute the seed queries.
    return [x for x in arr if isinstance(x, dict) and x.get("query")] if isinstance(arr, list) else []


def generate_queries_for_doc(caller: Caller, cluster_id: str, doc: Dict,
                             cluster_sections: Dict[str, str], doc_sections: Dict[str, List[str]],
                             cfg: QGenConfig) -> List[Dict]:
    """Generate ~cfg.n_queries questions PER DOCUMENT (not per shard). If the doc
    is long enough to shard, the quota is spread across shards for coverage."""
    doc_id = str(doc.get("doc_id", ""))
    own = doc_sections.get(doc_id, [])                       # generic document-level gold
    neighbors = _neighbors_blurb(cluster_sections, set(own))
    shards = shard_document(doc.get("text", ""), cfg.shard_max_chars) or [doc.get("text", "")]
    rng = random.Random(int(hashlib.sha256(doc_id.encode()).hexdigest()[:8], 16) or 1)
    per_shard = _distribute(cfg.n_queries, len(shards), rng)

    out: List[Dict] = []
    for si, (shard, cnt) in enumerate(zip(shards, per_shard)):
        if cnt <= 0:
            continue
        levels = _levels_for(cnt)
        level_lines = "\n".join(f"- {lv['level']}: {lv['desc']}" for lv in levels)
        user = QGEN_USER.format(
            doc_id=doc_id, shard_text=shard[:cfg.shard_max_chars], neighbors=neighbors,
            n=cnt, levels=level_lines,
            level_names=", ".join(lv["level"] for lv in DIFFICULTY_LEVELS))
        for qi, item in enumerate(_parse_queries(caller(QGEN_SYSTEM, user))):
            q = (item.get("query") or "").strip()
            if not q:
                continue
            level = str(item.get("level", "")).strip() or "crisp"
            variant = next((lv["variant"] for lv in DIFFICULTY_LEVELS if lv["level"] == level), "multi_step")
            # gold = the source doc's own sections + any related sections it targets
            # (all validated against real section ids in the cluster; no domain regex)
            targets = [str(t) for t in (item.get("target_sections") or []) if str(t) in cluster_sections]
            gold = list(dict.fromkeys(list(own) + targets))[:8] or list(own)
            out.append({
                "query_id": f"{cluster_id}_{doc_id}_s{si}_q{qi}",
                "cluster_id": cluster_id,
                "query": q,
                "level": level,
                "suggested_variant": variant,
                "gold_doc_ids": [doc_id],
                "gold_sections": gold,
                "shard_id": f"{doc_id}_s{si}",
                "rationale": (item.get("rationale") or "").strip(),
            })
            if len(out) >= cfg.n_queries:
                return out[: cfg.n_queries]
    return out[: cfg.n_queries]


# ── driver (one cluster) ─────────────────────────────────────────────────────
def run_cluster(cluster_dir: Path, caller: Optional[Caller], cfg: QGenConfig) -> Dict[str, int]:
    docs = [json.loads(l) for l in (cluster_dir / "docs.jsonl").open(encoding="utf-8") if l.strip()]
    if cfg.max_docs:
        docs = docs[: cfg.max_docs]
    cluster_id = cluster_dir.name

    # chunk with the configured profile → generic section ids per doc/cluster
    chunks = chunk_cluster(docs, cfg)
    cluster_sections, doc_sections = _section_maps(chunks)

    # 1) QUESTION GENERATION (n_queries PER DOCUMENT) -> queries.jsonl
    # resumable: reuse an existing non-empty queries.jsonl (don't re-pay the LLM)
    qpath = cluster_dir / "queries.jsonl"
    n_q = 0
    if caller is not None and qpath.exists() and qpath.stat().st_size > 0:
        n_q = sum(1 for l in qpath.open(encoding="utf-8") if l.strip())
        print(f"[question_gen] {cluster_id}: reusing {n_q} existing queries")
    elif caller is not None:
        with qpath.open("w", encoding="utf-8") as f:
            for d in docs:
                for q in generate_queries_for_doc(caller, cluster_id, d, cluster_sections, doc_sections, cfg):
                    f.write(json.dumps(q, ensure_ascii=False) + "\n")
                    n_q += 1

    # 2) chunks.jsonl (the retrieval-index inputs for Stage 3)
    with (cluster_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"[question_gen] {cluster_id}: {n_q} queries, {len(chunks)} chunks "
          f"({len(docs)} docs) -> {cluster_dir}")
    return {"chunks": len(chunks), "queries": n_q, "docs": len(docs)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 2: per-cluster chunking + question generation.")
    ap.add_argument("--cluster-dir", required=True, type=Path)
    ap.add_argument("--profile", default="plain", choices=sorted(PROFILES))
    ap.add_argument("--chunk-max-chars", type=int, default=2000)
    ap.add_argument("--chunk-overlap", type=int, default=0)
    ap.add_argument("--shard-max-chars", type=int, default=8000)
    ap.add_argument("--n-queries", type=int, default=5, help="queries PER DOCUMENT (spread across shards)")
    ap.add_argument("--max-docs", type=int, default=None)
    ap.add_argument("--no-llm", action="store_true", help="only chunk (skip question generation)")
    ap.add_argument("--model", default="nvidia/openai/gpt-oss-120b")
    ap.add_argument("--endpoint", default="https://inference-api.nvidia.com/v1")
    ap.add_argument("--api-key-env", default="NVIDIA_API_KEY")
    args = ap.parse_args()

    cfg = QGenConfig(profile=args.profile, chunk_max_chars=args.chunk_max_chars,
                     chunk_overlap=args.chunk_overlap, shard_max_chars=args.shard_max_chars,
                     n_queries=args.n_queries, max_docs=args.max_docs)
    caller = None if args.no_llm else make_openai_caller(args.model, args.endpoint, args.api_key_env)
    run_cluster(args.cluster_dir, caller, cfg)


if __name__ == "__main__":
    main()
