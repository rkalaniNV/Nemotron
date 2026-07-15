#!/usr/bin/env python3
"""Fast, modular NVIDIA-backed embedding + reranking + retrieval utility.

This is the shared retrieval layer for the agentic-RAG SDG pipeline. It is split
into three narrow, swappable pieces so the *clustering* stage (Stage 1) and the
*retrieval tool* (Stage 3/5) can both consume the same embedder while the backend
stays replaceable:

    Embedder   -- embed_passages() / embed_queries()  (NVIDIA NeMo Retriever NIM)
    Reranker   -- rerank(query, passages)             (NVIDIA rerank NIM)
    Retriever  -- one per cluster: semantic search + the SDG sampling policy

Design points (matching the team plan):
  * ONE embedder serves two granularities: whole documents (clustering, Stage 1)
    and chunks (per-cluster index, Stage 1 / retrieval, Stage 3). Only the input
    granularity changes -- `embed_passages` is called with docs OR chunks.
  * Retrieval policy is deliberate: take a semantic candidate pool of
    `candidate_multiplier * n` (default 2n), then RANDOM-SAMPLE down to `n`. The
    random sampling injects diversity so the agent must issue more searches / go
    deeper in multi-step (this is a feature, not a bug -- see the plan).
  * Reranking is OPTIONAL and toggleable, because it pulls the opposite direction
    (precision) from the random sampling (exploration). When on, it sharpens the
    pool *before* the random sample. Left off by default.
  * Everything is per-cluster and independent: build one ClusterIndex per cluster,
    retrieve only within a cluster.

NVIDIA backends (NeMo Retriever), matching this repo's `recipes/embed` + `recipes/rerank`:
  * embedding:  nvidia/llama-3.2-nv-embedqa-1b-v2   POST {base}/v1/embeddings
  * reranking:  nvidia/llama-nemotron-rerank-1b-v2  POST {base}/v1/ranking
Served either hosted (https://integrate.api.nvidia.com/v1, NVIDIA_API_KEY) or via
a self-hosted NIM (http://host:port/v1) -- same wire format, so it's a URL swap.

Offline fallbacks (HashEmbedder / LexicalReranker) are deterministic and need no
network or API key, so the whole pipeline (and this module's tests) runs anywhere.

Speed: HTTP embedding is the bottleneck. `NIMEmbedder` batches inputs and fires
batches concurrently over a thread pool; use a smaller output `dimensions`
(Matryoshka: 384/512/768/1024/2048) for cheaper clustering/search.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np

# ── defaults (hosted build.nvidia.com endpoints, verified live 2026-07) ───────
# NOTE: the older `llama-3.2-nv-embedqa-1b-v2` embedding model and the
# `.../nv-rerankqa-1b-v2/reranking` endpoint both reached end-of-life 2026-05-18
# on the hosted API. Current hosted defaults below; override via env/args for a
# self-hosted NIM (embeddings: {base}/embeddings, rerank: {host}/v1/ranking).
DEFAULT_BASE_URL = os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
DEFAULT_EMBED_MODEL = os.environ.get("NVIDIA_EMBED_MODEL", "nvidia/llama-nemotron-embed-1b-v2")
DEFAULT_RERANK_URL = os.environ.get(
    "NVIDIA_RERANK_URL", "https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking")
DEFAULT_RERANK_MODEL = os.environ.get("NVIDIA_RERANK_MODEL", "nvidia/rerank-qa-mistral-4b")
EMBED_MAX_TOKENS = 8192          # llama-nemotron-embed-1b-v2 context window
# Conservative chars/token for long-doc WINDOWING (real English is ~4; 3 keeps
# each window safely under the 8192-token cap so server-side truncation -- which
# `truncate=END` still guarantees as a backstop -- never silently drops content
# mid-window before mean-pooling).
_WINDOW_CHARS_PER_TOKEN = 3
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _json_loads(raw):
    """Decode a JSON body (bytes or str) -- used by callers and tests."""
    return json.loads(raw)


_RETRYABLE_STATUS = (429, 500, 502, 503, 504)


def _request_with_retry(client, url: str, payload: Dict, max_retries: int, what: str) -> Dict:
    """POST JSON with retry on transient status (429/5xx) and exponential-ish
    backoff-free re-try; fail FAST on other 4xx, surfacing the server's message
    (e.g. an 8192-token overflow) so misconfig is obvious out of the box."""
    import httpx
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            r = client.post(url, json=payload)
            if r.status_code in _RETRYABLE_STATUS:
                raise httpx.HTTPStatusError(f"retryable {r.status_code}", request=r.request, response=r)
            if r.status_code >= 400:
                # non-retryable client error -> stop immediately with the body
                raise RuntimeError(f"NIM {what} HTTP {r.status_code}: {r.text[:300]}")
            return r.json()
        except RuntimeError:
            raise
        except Exception as exc:  # noqa: BLE001 - transient; retry then re-raise
            last_exc = exc
    raise RuntimeError(f"NIM {what} failed after {max_retries} tries: {last_exc}")


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype="float32")
    if mat.ndim == 1:
        mat = mat[None, :]
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return mat / norms


# ══════════════════════════════════════════════════════════════════════════════
# Embedders
# ══════════════════════════════════════════════════════════════════════════════
class Embedder(Protocol):
    """Narrow interface both clustering (docs) and retrieval (chunks) depend on.

    Both methods return an (N, dim) L2-normalised float32 array, so cosine
    similarity is a plain dot product.
    """
    dim: int

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray: ...
    def embed_queries(self, texts: Sequence[str]) -> np.ndarray: ...


# --- wire helpers (pure functions -> unit-testable without a network) ---------
def build_embed_payload(model: str, texts: Sequence[str], input_type: str,
                        truncate: str = "END", dimensions: Optional[int] = None) -> Dict:
    """OpenAI-compatible NeMo Retriever embeddings request body."""
    if input_type not in ("query", "passage"):
        raise ValueError(f"input_type must be 'query' or 'passage', got {input_type!r}")
    payload: Dict[str, object] = {
        "model": model,
        "input": list(texts),
        "input_type": input_type,
        "truncate": truncate,
    }
    if dimensions:
        payload["dimensions"] = dimensions
    return payload


def parse_embed_response(body: Dict) -> List[List[float]]:
    """Extract embeddings from an OpenAI-style response, preserving input order."""
    data = sorted(body["data"], key=lambda d: d.get("index", 0))
    return [d["embedding"] for d in data]


class NIMEmbedder:
    """NVIDIA NeMo Retriever embedding NIM (hosted or self-hosted).

    Concurrency: inputs are split into `batch_size` batches; up to `concurrency`
    batches are POSTed in parallel over a thread pool (network I/O bound, so
    threads give real speedup and stay safe inside or outside an event loop).
    """

    def __init__(self, model: str = DEFAULT_EMBED_MODEL, base_url: str = DEFAULT_BASE_URL,
                 api_key: Optional[str] = None, dim: int = 2048, dimensions: Optional[int] = None,
                 batch_size: int = 64, concurrency: int = 8, timeout: float = 60.0,
                 max_retries: int = 3):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("NVIDIA_API_KEY", "")
        self.dimensions = dimensions
        self.dim = dimensions or dim
        self.batch_size = max(1, batch_size)
        self.concurrency = max(1, concurrency)
        self.timeout = timeout
        self.max_retries = max_retries

    # -- transport -------------------------------------------------------------
    def _post(self, client, texts: Sequence[str], input_type: str) -> List[List[float]]:
        payload = build_embed_payload(self.model, texts, input_type, dimensions=self.dimensions)
        return parse_embed_response(
            _request_with_retry(client, f"{self.base_url}/embeddings", payload,
                                self.max_retries, "embeddings"))

    def _embed(self, texts: Sequence[str], input_type: str) -> np.ndarray:
        import httpx
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        batches = [texts[i:i + self.batch_size] for i in range(0, len(texts), self.batch_size)]
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        with httpx.Client(timeout=self.timeout, headers=headers) as client:
            with ThreadPoolExecutor(max_workers=min(self.concurrency, len(batches))) as pool:
                results = list(pool.map(lambda b: self._post(client, b, input_type), batches))
        vecs = [v for batch in results for v in batch]
        arr = _l2_normalize(np.asarray(vecs, dtype="float32"))
        self.dim = arr.shape[1]  # trust the server's actual dim
        return arr

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray:
        return self._embed(texts, "passage")

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._embed(texts, "query")

    # -- Stage 1 helper: embed WHOLE documents without chunking ----------------
    def embed_documents(self, docs: Sequence[str], max_tokens: int = EMBED_MAX_TOKENS) -> np.ndarray:
        """One vector per document, covering the full text despite the 8192-token
        cap: a doc longer than the window is split into windows, each embedded as
        a passage, and the windows are mean-pooled into a single document vector.
        Short docs take a single call. Keeps clustering's "no chunking" interface
        (input = documents, output = one vector each) while avoiding silent
        truncation of long documents.
        """
        max_chars = max_tokens * _WINDOW_CHARS_PER_TOKEN
        windows: List[str] = []
        spans: List[Tuple[int, int]] = []  # (start, end) window index range per doc
        for d in docs:
            d = d or ""
            start = len(windows)
            if len(d) <= max_chars:
                windows.append(d)
            else:
                windows.extend(d[i:i + max_chars] for i in range(0, len(d), max_chars))
            spans.append((start, len(windows)))
        win_vecs = self.embed_passages(windows) if windows else np.zeros((0, self.dim), "float32")
        out = np.zeros((len(docs), win_vecs.shape[1] if win_vecs.size else self.dim), dtype="float32")
        for i, (s, e) in enumerate(spans):
            if e > s:
                out[i] = win_vecs[s:e].mean(axis=0)
        return _l2_normalize(out)


class HashEmbedder:
    """Deterministic, dependency-free offline embedder (feature hashing).

    Not a real semantic model -- but texts that share tokens get higher cosine,
    which is enough to exercise clustering / retrieval / ranking end-to-end with
    no API key. Same interface as NIMEmbedder, so it is a drop-in for tests and
    local dry-runs.
    """

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype="float32")
        for tok in _TOKEN_RE.findall((text or "").lower()):
            h = hash(tok) % self.dim  # note: process-local hash; fine within one run
            v[h] += 1.0
        return v

    def _embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        return _l2_normalize(np.stack([self._vec(t) for t in texts]))

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray:
        return self._embed(texts)

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._embed(texts)

    def embed_documents(self, docs: Sequence[str], max_tokens: int = EMBED_MAX_TOKENS) -> np.ndarray:
        return self._embed(docs)


def make_embedder(backend: str = "nim", **kwargs) -> Embedder:
    """Factory. `backend='nim'` uses NVIDIA; falls back to the offline hash
    embedder if httpx is missing or no NVIDIA_API_KEY is set (so nothing hard-fails
    in a keyless environment). `backend='hash'` forces offline."""
    if backend == "hash":
        return HashEmbedder(**{k: v for k, v in kwargs.items() if k == "dim"})
    if backend == "nim":
        try:
            import httpx  # noqa: F401
        except ImportError:
            print("[nv_retrieval] httpx not installed; using offline hash embedder.")
            return HashEmbedder()
        if not (kwargs.get("api_key") or os.environ.get("NVIDIA_API_KEY")):
            print("[nv_retrieval] NVIDIA_API_KEY not set; using offline hash embedder. "
                  "Set the key (build.nvidia.com) to use the NeMo Retriever NIM.")
            return HashEmbedder()
        return NIMEmbedder(**kwargs)
    raise ValueError(f"unknown embedder backend: {backend}")


# ══════════════════════════════════════════════════════════════════════════════
# Rerankers
# ══════════════════════════════════════════════════════════════════════════════
class Reranker(Protocol):
    def rerank(self, query: str, passages: Sequence[str],
               top_n: Optional[int] = None) -> List[Tuple[int, float]]:
        """Return [(original_index, score)] sorted by descending relevance."""
        ...


def build_rerank_payload(model: str, query: str, passages: Sequence[str]) -> Dict:
    return {"model": model, "query": {"text": query},
            "passages": [{"text": p} for p in passages]}


def parse_rerank_response(body: Dict) -> List[Tuple[int, float]]:
    """NeMo Retriever rerank returns {'rankings': [{'index': i, 'logit': s}, ...]}
    (already sorted). Tolerate 'score'/'relevance' key variants."""
    rankings = body.get("rankings") or body.get("results") or []
    out: List[Tuple[int, float]] = []
    for r in rankings:
        score = r.get("logit", r.get("score", r.get("relevance_score", 0.0)))
        out.append((int(r["index"]), float(score)))
    return out


class NIMReranker:
    """NVIDIA NeMo Retriever reranking NIM (hosted or self-hosted).

    `url` is the FULL ranking endpoint (it differs from the embeddings host):
      * hosted:       https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking
      * self-hosted:  http://host:port/v1/ranking
    """

    def __init__(self, model: str = DEFAULT_RERANK_MODEL, url: str = DEFAULT_RERANK_URL,
                 api_key: Optional[str] = None, timeout: float = 60.0, max_retries: int = 3):
        self.model = model
        self.url = url
        self.api_key = api_key or os.environ.get("NVIDIA_API_KEY", "")
        self.timeout = timeout
        self.max_retries = max_retries

    def rerank(self, query: str, passages: Sequence[str],
               top_n: Optional[int] = None) -> List[Tuple[int, float]]:
        import httpx
        passages = list(passages)
        if not passages:
            return []
        payload = build_rerank_payload(self.model, query, passages)
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        with httpx.Client(timeout=self.timeout, headers=headers) as client:
            body = _request_with_retry(client, self.url, payload, self.max_retries, "rerank")
        ranked = parse_rerank_response(body)
        return ranked[:top_n] if top_n else ranked


class LexicalReranker:
    """Offline fallback: token-overlap (IDF-free) scoring. Deterministic, no deps."""

    def rerank(self, query: str, passages: Sequence[str],
               top_n: Optional[int] = None) -> List[Tuple[int, float]]:
        q = set(_TOKEN_RE.findall((query or "").lower()))
        scored: List[Tuple[int, float]] = []
        for i, p in enumerate(passages):
            toks = _TOKEN_RE.findall((p or "").lower())
            overlap = sum(1 for t in toks if t in q)
            scored.append((i, overlap / (len(toks) + 1e-9)))
        scored.sort(key=lambda x: -x[1])
        return scored[:top_n] if top_n else scored


def make_reranker(backend: str = "nim", **kwargs) -> Reranker:
    if backend == "lexical":
        return LexicalReranker()
    if backend == "nim":
        try:
            import httpx  # noqa: F401
        except ImportError:
            print("[nv_retrieval] httpx not installed; using offline lexical reranker.")
            return LexicalReranker()
        if not (kwargs.get("api_key") or os.environ.get("NVIDIA_API_KEY")):
            print("[nv_retrieval] NVIDIA_API_KEY not set; using offline lexical reranker.")
            return LexicalReranker()
        return NIMReranker(**kwargs)
    raise ValueError(f"unknown reranker backend: {backend}")


# ══════════════════════════════════════════════════════════════════════════════
# Per-cluster index + retriever (the shape Stage 3/5 calls)
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    metadata: Dict[str, object] = field(default_factory=dict)

    def to_tool_payload(self) -> Dict[str, object]:
        return {"chunk_id": self.chunk_id, "text": self.text,
                "score": round(self.score, 4), **self.metadata}


class ClusterIndex:
    """One independent index over a single cluster's chunks.

    Holds the chunk records plus their (N, dim) L2-normalised embeddings, so a
    query is scored by a single matmul. Built and cached per cluster; nothing
    here is shared across clusters (per the plan's independence requirement).
    """

    def __init__(self, chunks: List[Dict], embedder: Embedder,
                 embeddings: Optional[np.ndarray] = None, cluster_id: str = ""):
        self.chunks = chunks
        self.embedder = embedder
        self.cluster_id = cluster_id
        self.ids = [str(c.get("chunk_id", i)) for i, c in enumerate(chunks)]
        self.texts = [str(c.get("text", "")) for c in chunks]
        self._emb = embeddings

    def build(self) -> "ClusterIndex":
        self._emb = self.embedder.embed_passages(self.texts) if self.texts \
            else np.zeros((0, self.embedder.dim), "float32")
        return self

    def _scores(self, query: str) -> np.ndarray:
        if self._emb is None:
            self.build()
        q = self.embedder.embed_queries([query])[0]
        return self._emb @ q  # cosine (both normalised)

    def semantic_topk(self, query: str, k: int) -> List[Tuple[int, float]]:
        if not self.texts:
            return []
        scores = self._scores(query)
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(i), float(scores[i])) for i in top]

    # -- persistence -----------------------------------------------------------
    def save(self, index_dir: Path) -> None:
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        np.save(index_dir / "embeddings.npy", self._emb)
        with (index_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        (index_dir / "meta.json").write_text(json.dumps({
            "cluster_id": self.cluster_id, "n": len(self.ids),
            "dim": int(self._emb.shape[1]) if self._emb is not None and self._emb.size else self.embedder.dim,
        }))

    @classmethod
    def load(cls, index_dir: Path, embedder: Embedder) -> "ClusterIndex":
        index_dir = Path(index_dir)
        chunks = [json.loads(l) for l in (index_dir / "chunks.jsonl").open(encoding="utf-8") if l.strip()]
        emb = np.load(index_dir / "embeddings.npy")
        meta = json.loads((index_dir / "meta.json").read_text())
        return cls(chunks, embedder, embeddings=emb, cluster_id=meta.get("cluster_id", ""))


class Retriever:
    """Retrieval tool behind the agent's chunk-retriever tool call.

    Policy (the SDG plan):
        1. semantic top `candidate_multiplier * n` (default 2n)   -- coarse recall
        2. [optional] rerank that pool, keep the top `candidate_multiplier * n`
        3. RANDOM-SAMPLE down to `n`                              -- exploration

    Swappable: it only depends on the ClusterIndex (embedder) + optional Reranker,
    so replacing the retrieval offering later means swapping those, not this.
    """

    def __init__(self, index: ClusterIndex, reranker: Optional[Reranker] = None):
        self.index = index
        self.reranker = reranker

    def retrieve(self, query: str, n: int = 4, *, candidate_multiplier: int = 2,
                 rerank: bool = False, seed: Optional[int] = None) -> List[RetrievedChunk]:
        if n <= 0 or not self.index.texts:
            return []
        pool_k = max(candidate_multiplier * n, n)
        # a larger pre-pool when reranking, so rerank has something to reorder
        prepool_k = max(pool_k, 4 * n) if (rerank and self.reranker) else pool_k
        cand = self.index.semantic_topk(query, prepool_k)

        if rerank and self.reranker and cand:
            idxs = [i for i, _ in cand]
            reranked = self.reranker.rerank(query, [self.index.texts[i] for i in idxs])
            cand = [(idxs[j], s) for j, s in reranked]

        cand = cand[:pool_k]                                   # the 2n candidate pool
        rng = random.Random(seed)
        chosen = rng.sample(cand, min(n, len(cand)))           # random-sample down to n
        return [self._wrap(i, s) for i, s in chosen]

    def _wrap(self, idx: int, score: float) -> RetrievedChunk:
        c = self.index.chunks[idx]
        meta = {k: v for k, v in c.items() if k not in ("chunk_id", "text")}
        return RetrievedChunk(chunk_id=self.index.ids[idx], text=self.index.texts[idx],
                              score=score, metadata=meta)


# ── orchestration: build one index per cluster (Stage 1 hand-off) ─────────────
def build_cluster_indexes(clusters: Dict[str, List[Dict]], embedder: Embedder,
                          out_dir: Optional[Path] = None) -> Dict[str, ClusterIndex]:
    """Given {cluster_id: [chunk_records]} (the output of the clustering +
    chunking stage), build and optionally cache one independent index per cluster.

    This is the integration seam with the teammate's clustering module: it hands
    us clusters of chunks; we hand back a ready-to-query Retriever per cluster.
    """
    indexes: Dict[str, ClusterIndex] = {}
    for cid, chunks in clusters.items():
        idx = ClusterIndex(chunks, embedder, cluster_id=cid).build()
        if out_dir is not None:
            idx.save(Path(out_dir) / f"cluster_{cid}")
        indexes[cid] = idx
    return indexes


# ── tiny offline demo ─────────────────────────────────────────────────────────
def _demo() -> None:
    embedder = make_embedder("hash", dim=256)
    chunks = [
        {"chunk_id": "c1", "text": "The Motor Vehicles Act requires a valid driving licence."},
        {"chunk_id": "c2", "text": "Penalties for driving without a licence include fines."},
        {"chunk_id": "c3", "text": "Traffic signage standards define the shape and colour of signs."},
        {"chunk_id": "c4", "text": "Road safety rules cover speed limits and seat belts."},
        {"chunk_id": "c5", "text": "Fundamental rights guarantee equality before the law."},
    ]
    idx = ClusterIndex(chunks, embedder, cluster_id="motor").build()
    retr = Retriever(idx, reranker=make_reranker("lexical"))
    got = retr.retrieve("penalty for driving without a licence", n=2,
                        candidate_multiplier=2, rerank=True, seed=0)
    print("[demo] retrieved:", [(c.chunk_id, round(c.score, 3)) for c in got])


def _selftest() -> int:
    """OOTB check: exercise embed + rerank + retrieve. Hits the live NVIDIA NIM
    when NVIDIA_API_KEY is set, else runs the offline fallbacks. Returns 0 on OK."""
    live = bool(os.environ.get("NVIDIA_API_KEY"))
    print(f"[selftest] mode = {'LIVE NIM' if live else 'offline fallback'}")
    embedder, reranker = make_embedder("nim"), make_reranker("nim")
    print(f"[selftest] embedder={type(embedder).__name__} "
          f"reranker={type(reranker).__name__}")
    chunks = [
        {"chunk_id": "a1", "text": "The Motor Vehicles Act requires a valid driving licence."},
        {"chunk_id": "a2", "text": "Penalties include fines for driving without a licence."},
        {"chunk_id": "a3", "text": "A balanced diet includes proteins, fats and vitamins."},
        {"chunk_id": "a4", "text": "Traffic signage defines the shape and colour of road signs."},
    ]
    idx = ClusterIndex(chunks, embedder, cluster_id="selftest").build()
    print(f"[selftest] index dim = {idx._emb.shape[1]}, chunks = {len(chunks)}")
    retr = Retriever(idx, reranker=reranker)
    q = "what is the fine for driving without a licence"
    print("[selftest] semantic top-3:", idx.semantic_topk(q, 3))
    print("[selftest] rerank order:", reranker.rerank(q, [c["text"] for c in chunks])[:3])
    hits = retr.retrieve(q, n=2, candidate_multiplier=2, rerank=True, seed=1)
    assert len(hits) == 2, "expected 2 retrieved chunks"
    print("[selftest] retrieve(n=2, rerank=on):",
          [(h.chunk_id, round(h.score, 3)) for h in hits])
    print("[selftest] OK")
    return 0


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    _demo()
