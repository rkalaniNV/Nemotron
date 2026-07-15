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
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# make the corpus-side retriever importable (data_prep is a sibling dir)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_prep"))
from retriever import (  # noqa: E402
    make_retriever, EmbeddingRetriever, NIMEmbeddingRetriever, gold_rank, RetrievedChunk)

from .llm import call_llm
from .prompts import API_RESPONSE_SIM_SYSTEM_PROMPT, API_RESPONSE_SIM_TURN_PROMPT

API_RESPONSE_MODEL = "api_response_model"


class ToolEnvironment:
    """Holds the retriever + config and answers tool calls for one run."""

    def __init__(self, config, corpus: List[Dict[str, Any]], gold_section_ids: List[str],
                 rng: Optional[random.Random] = None):
        self.cfg = config
        self.corpus = corpus
        self.gold = [str(g) for g in gold_section_ids]
        self.rng = rng or random.Random(config.salvage_min_hops)  # deterministic-ish
        self._by_section: Dict[str, List[Dict[str, Any]]] = {}
        for c in corpus:
            self._by_section.setdefault(str(c.get("section_id", "")), []).append(c)
        self._retriever = None
        self._retries: Dict[str, int] = {}  # per gold-section retry counter

    # ── retriever (lazy) ─────────────────────────────────────────────────────
    @property
    def retriever(self):
        if self._retriever is None:
            cached = self.cfg.index_dir and (Path(self.cfg.index_dir) / "meta.json").exists()
            loaders = {"embedding": EmbeddingRetriever, "nim": NIMEmbeddingRetriever}
            if cached and self.cfg.retriever_backend in loaders:
                self._retriever = loaders[self.cfg.retriever_backend].load(Path(self.cfg.index_dir))
            else:
                self._retriever = make_retriever(self.corpus, backend=self.cfg.retriever_backend)
        return self._retriever

    # ── entry point ──────────────────────────────────────────────────────────
    def respond(self, tool_call: Dict[str, Any], models: Dict[str, Any],
                user_query: str, tools: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
        name = tool_call.get("function", {}).get("name", "")
        try:
            args = json.loads(tool_call.get("function", {}).get("arguments", "{}"))
        except (json.JSONDecodeError, TypeError):
            args = {}

        # config-driven error injection (teaches recovery)
        if self.cfg.error_injection_rate > 0 and self.rng.random() < self.cfg.error_injection_rate:
            return json.dumps({"error": "temporary backend error, please retry"}), {"injected_error": True}

        if name in self.cfg.retrieval_tools:
            return self._retrieval_response(name, args)
        return self._simulated_response(tool_call, models, user_query, tools), {"simulated": True}

    # ── retrieval backend (grounded) ─────────────────────────────────────────
    def _retrieval_response(self, name: str, args: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        # get_article-style direct lookup by id
        id_arg = args.get("article") or args.get("article_id") or args.get("section_id") or args.get("id")
        if id_arg is not None and str(id_arg) in self._by_section:
            chunks = self._by_section[str(id_arg)]
            payload = [{"chunk_id": c["chunk_id"], "article": c.get("section_id"),
                        "title": c.get("section_title", ""), "text": c.get("text", "")} for c in chunks]
            meta = {"tool": name, "mode": "direct", "returned": [c["chunk_id"] for c in chunks],
                    "gold_rank": gold_rank_from_sections([str(id_arg)], self.gold)}
            return json.dumps({"results": payload}), self._decorate(payload, meta)

        # search-style semantic retrieval
        query = args.get("query") or args.get("q") or ""
        results: List[RetrievedChunk] = self.retriever.search(query, k=self.cfg.top_k)
        rank = gold_rank(results, self.gold) if self.cfg.log_gold_rank else None

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


def gold_rank_from_sections(returned_sections: List[str], gold: List[str]) -> Optional[int]:
    for rank, s in enumerate(returned_sections, 1):
        if s in set(gold):
            return rank
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
