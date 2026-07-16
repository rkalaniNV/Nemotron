"""Tool environment: routes each tool call to the right backend.

The core design decision of this pipeline lives here:

  - RETRIEVAL tools (``cfg.retrieval_tools``) are answered by the REAL retriever
    (MiniLM embeddings over the chunk corpus) — grounded, no hallucinated text.
    We log the gold-chunk rank and, if configured, inject a missed gold chunk
    after a couple of failed retries (guided fallback) to protect yield.

  - AUXILIARY tools are answered by an LLM simulator (as in the reference
    pipeline) — fine for tools that don't need factual grounding.

Everything is config-driven; nothing about the domain is hard-coded.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# canonical modular retriever (same module the offline CLI re-exports)
from .retrieval import make_retriever, EmbeddingRetriever, gold_rank, RetrievedChunk, load_corpus
from .llm import call_llm
from .prompts import API_RESPONSE_SIM_SYSTEM_PROMPT, API_RESPONSE_SIM_TURN_PROMPT

API_RESPONSE_MODEL = "api_response_model"

# Process-wide cache so a cluster's index (and MiniLM model) load ONCE, not per
# row. Keyed by the resolved index dir (or an in-memory corpus fingerprint).
_RETRIEVER_CACHE: Dict[str, Any] = {}


class ToolEnvironment:
    """Holds the retriever + config and answers tool calls for one run."""

    def __init__(self, config, corpus: List[Dict[str, Any]], gold_section_ids: List[str],
                 rng: Optional[random.Random] = None, index_dir: Optional[str] = None,
                 gold_doc_ids: Optional[List[str]] = None):
        self.cfg = config
        self.corpus = corpus
        self.gold = [str(g) for g in gold_section_ids]
        self.gold_docs = [str(g) for g in (gold_doc_ids or [])]   # document-level grounding
        self.rng = rng or random.Random(config.salvage_min_hops)  # deterministic-ish
        # per-cluster index dir (overrides cfg.index_dir); None -> in-memory over corpus
        self.index_dir = index_dir or config.index_dir
        self._by_section: Dict[str, List[Dict[str, Any]]] = {}
        for c in corpus:
            self._by_section.setdefault(str(c.get("section_id", "")), []).append(c)
        self._retriever = None
        self._retries: Dict[str, int] = {}  # per gold-section retry counter
        self._seen: set = set()             # chunk ids already returned (per conversation)

    # ── retriever (lazy, cached per index dir across rows) ───────────────────
    @property
    def retriever(self):
        if self._retriever is not None:
            return self._retriever
        key = str(self.index_dir) if self.index_dir else f"_mem_{id(self.corpus)}"
        cached = _RETRIEVER_CACHE.get(key)
        if cached is None:
            if self.index_dir and (Path(self.index_dir) / "meta.json").exists() \
                    and self.cfg.retriever_backend == "embedding":
                cached = EmbeddingRetriever.load(Path(self.index_dir))
            else:
                cached = make_retriever(self.corpus, backend=self.cfg.retriever_backend)
            _RETRIEVER_CACHE[key] = cached
        self._retriever = cached
        return self._retriever

    # ── entry point ──────────────────────────────────────────────────────────
    def respond(self, tool_call: Dict[str, Any], models: Dict[str, Any],
                user_query: str, tools: List[Dict[str, Any]],
                rng: Optional[random.Random] = None) -> Tuple[str, Dict[str, Any]]:
        r = rng or self.rng  # per-call rng keeps concurrent tool execution thread-safe
        name = tool_call.get("function", {}).get("name", "")
        try:
            args = json.loads(tool_call.get("function", {}).get("arguments", "{}"))
        except (json.JSONDecodeError, TypeError):
            args = {}

        # config-driven error injection (teaches recovery)
        if self.cfg.error_injection_rate > 0 and r.random() < self.cfg.error_injection_rate:
            return json.dumps({"error": "temporary backend error, please retry"}), {"injected_error": True}

        if name in self.cfg.retrieval_tools:
            return self._retrieval_response(name, args, r)
        return self._simulated_response(tool_call, models, user_query, tools), {"simulated": True}

    # ── retrieval backend (grounded) ─────────────────────────────────────────
    def _retrieval_response(self, name: str, args: Dict[str, Any],
                            rng: Optional[random.Random] = None) -> Tuple[str, Dict[str, Any]]:
        # direct lookup by id (fetch a specific section/document by its id) —
        # read whatever id-like arg the configured tool provides, generically.
        id_arg = (args.get("id") or args.get("section") or args.get("section_id")
                  or args.get("doc_id") or args.get("article") or _first_scalar(args))
        if id_arg is not None and str(id_arg) in self._by_section:
            chunks = self._by_section[str(id_arg)]
            payload = [{"chunk_id": c["chunk_id"], "section": c.get("section_id"),
                        "doc_id": c.get("doc_id", ""), "title": c.get("section_title", ""),
                        "text": c.get("text", "")} for c in chunks]
            hit = str(id_arg) in set(self.gold) or any(str(c.get("doc_id")) in set(self.gold_docs) for c in chunks)
            meta = {"tool": name, "mode": "direct", "returned": [c["chunk_id"] for c in chunks],
                    "gold_rank": (1 if hit else None)}
            return json.dumps({"results": payload}), self._decorate(payload, meta)

        # search-style semantic retrieval.
        # Stage 3: oversample a pool by similarity, drop chunks already returned
        # this conversation (dedupe → each hop brings NEW text), then RANDOMLY
        # keep top_k (subsample → a single search is lossy, so depth is rewarded).
        query = args.get("query") or args.get("q") or ""
        r = rng or self.rng
        k = self.cfg.top_k
        results = self._pooled_retrieve(query, k, r)
        self._seen.update(c.chunk_id for c in results)
        rank = gold_rank(results, self.gold, self.gold_docs) if self.cfg.log_gold_rank else None

        # guided injection: gold missing after enough retries -> inject it
        injected = False
        key = ",".join(self.gold)
        self._retries[key] = self._retries.get(key, 0) + 1
        if (rank is None and self.cfg.guided_injection
                and self._retries[key] >= self.cfg.guided_injection_after):
            gold_chunk = self._first_gold_chunk()
            if gold_chunk is not None:
                results = [RetrievedChunk(
                    chunk_id=gold_chunk["chunk_id"], text=gold_chunk.get("text", ""), score=1.0,
                    section_id=str(gold_chunk.get("section_id", "")),
                    section_title=gold_chunk.get("section_title", ""))] + results[:-1]
                injected = True
                rank = 1

        payload = [r.to_tool_payload() for r in results]
        meta = {"tool": name, "mode": "search", "query": query,
                "returned": [r.chunk_id for r in results], "gold_rank": rank, "injected": injected}
        return json.dumps({"results": payload}), self._decorate(payload, meta)

    def _pooled_retrieve(self, query: str, k: int, rng: random.Random) -> List[RetrievedChunk]:
        """Retrieve k chunks with optional dedupe (drop already-seen) + subsample."""
        if not (self.cfg.subsample_retrieval or self.cfg.dedupe_retrieval):
            return self.retriever.search(query, k=k)
        pool = self.retriever.search(query, k=k * max(2, self.cfg.oversample_factor))
        if self.cfg.dedupe_retrieval:
            fresh = [c for c in pool if c.chunk_id not in self._seen]
            if fresh:                       # only fall back to repeats if nothing new
                pool = fresh
        if self.cfg.subsample_retrieval and len(pool) > k:
            return [pool[i] for i in sorted(rng.sample(range(len(pool)), k))]
        return pool[:k]

    def _first_gold_chunk(self) -> Optional[Dict[str, Any]]:
        for g in self.gold:
            if g in self._by_section:
                return self._by_section[g][0]
        return None

    @staticmethod
    def _decorate(payload: List[Dict[str, Any]], meta: Dict[str, Any]) -> Dict[str, Any]:
        # top chunk id used by the context manager for compaction references
        meta["_chunk_id"] = payload[0]["chunk_id"] if payload else None
        return meta

    # ── simulated backend (auxiliary tools) ──────────────────────────────────
    def _simulated_response(self, tool_call, models, user_query, tools) -> str:
        name = tool_call.get("function", {}).get("name")
        spec = None
        for td in tools:
            fn = (td["tool"]["function"] if "tool" in td else td.get("function", {}))
            if fn.get("name") == name:
                spec = fn
                break
        prompt = API_RESPONSE_SIM_TURN_PROMPT.format(
            tool_spec=json.dumps(spec or {}, ensure_ascii=False),
            user_query=user_query,
            tool_call=json.dumps(tool_call.get("function", {}), ensure_ascii=False),
        )
        resp = call_llm(models, API_RESPONSE_MODEL,
                        [{"role": "system", "content": API_RESPONSE_SIM_SYSTEM_PROMPT},
                         {"role": "user", "content": prompt}])
        return resp.get("content", "") if isinstance(resp, dict) else ""


def _first_scalar(args: Dict[str, Any]) -> Optional[str]:
    """First string/number arg value — a generic fallback id for direct lookups
    when the tool's id parameter isn't one of the common names."""
    for v in args.values():
        if isinstance(v, (str, int, float)) and not isinstance(v, bool):
            return str(v)
    return None


def sample_tools(all_tools: List[Dict[str, Any]], max_tools: int,
                 must_include: List[str], rng: random.Random) -> List[Dict[str, Any]]:
    """Offer a subset, always keeping the retrieval tools present."""
    def tool_name(td):
        return (td["tool"]["function"] if "tool" in td else td.get("function", {})).get("name")

    keep = [t for t in all_tools if tool_name(t) in set(must_include)]
    rest = [t for t in all_tools if tool_name(t) not in set(must_include)]
    rng.shuffle(rest)
    return keep + rest[: max(0, max_tools - len(keep))]
