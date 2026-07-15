#!/usr/bin/env python3
"""Retriever backend for the agentic-RAG SDG pipeline.

This is the real search function behind the RAG tools (``search_articles`` /
``get_article``). At each retrieval tool-call the assistant issues a query, and
this module returns the top-k most similar corpus chunks — grounding the tool
response in real document text instead of an LLM hallucination.

Primary backend: **sentence-transformers/all-MiniLM-L6-v2** (384-dim, CPU-fast,
strong on short passages — a good fit for our ~140-token chunks). Cosine
similarity via numpy brute force is plenty for corpora up to ~100k chunks; swap
in FAISS only if the corpus grows.

A dependency-free **lexical** fallback is included so the pipeline is runnable
(and testable) on machines without sentence-transformers; results are inferior
but the interface is identical.

Everything here is domain-agnostic: it operates on a JSONL corpus with
``chunk_id`` / ``text`` (+ optional ``section_id`` / ``section_title``) fields.

Usage:
    # build + cache the embedding index
    python retriever.py build --chunks ../data/constitution_chunks.jsonl \
        --index ../data/index --backend embedding
    # smoke-test a query
    python retriever.py query --index ../data/index \
        --q "police held my brother without telling him why" --k 5
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    section_id: str = ""
    section_title: str = ""

    def to_tool_payload(self) -> Dict[str, object]:
        """Shape returned to the assistant as the tool response."""
        return {
            "chunk_id": self.chunk_id,
            "article": self.section_id,
            "title": self.section_title,
            "text": self.text,
            "score": round(self.score, 4),
        }


def load_corpus(path: Path) -> List[Dict[str, object]]:
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


# ── base ─────────────────────────────────────────────────────────────────────
class Retriever:
    """Interface: build an index over corpus records, then search."""

    def __init__(self, corpus: List[Dict[str, object]]):
        self.corpus = corpus
        self.ids = [c["chunk_id"] for c in corpus]
        self.texts = [c.get("text", "") for c in corpus]
        self.sec_ids = [str(c.get("section_id", "")) for c in corpus]
        self.titles = [c.get("section_title", "") for c in corpus]

    def search(self, query: str, k: int = 5) -> List[RetrievedChunk]:  # pragma: no cover
        raise NotImplementedError

    def _wrap(self, idx: int, score: float) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=self.ids[idx], text=self.texts[idx], score=float(score),
            section_id=self.sec_ids[idx], section_title=self.titles[idx],
        )


# ── embedding backend (MiniLM) ───────────────────────────────────────────────
class EmbeddingRetriever(Retriever):
    def __init__(self, corpus, model_name: str = EMBED_MODEL, embeddings=None):
        super().__init__(corpus)
        self.model_name = model_name
        self._model = None
        self._emb = embeddings  # (N, d) L2-normalised float32, or None to build

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # lazy import
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _encode(self, texts: Sequence[str]):
        import numpy as np
        model = self._load_model()
        emb = model.encode(list(texts), batch_size=64, show_progress_bar=False,
                           convert_to_numpy=True, normalize_embeddings=True)
        return emb.astype("float32")

    def build(self) -> "EmbeddingRetriever":
        self._emb = self._encode(self.texts)
        return self

    def save(self, index_dir: Path) -> None:
        import numpy as np
        index_dir.mkdir(parents=True, exist_ok=True)
        np.save(index_dir / "embeddings.npy", self._emb)
        with (index_dir / "corpus.jsonl").open("w", encoding="utf-8") as f:
            for c in self.corpus:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        (index_dir / "meta.json").write_text(json.dumps(
            {"backend": "embedding", "model": self.model_name, "n": len(self.ids)}))

    @classmethod
    def load(cls, index_dir: Path) -> "EmbeddingRetriever":
        import numpy as np
        corpus = load_corpus(index_dir / "corpus.jsonl")
        emb = np.load(index_dir / "embeddings.npy")
        meta = json.loads((index_dir / "meta.json").read_text())
        return cls(corpus, model_name=meta.get("model", EMBED_MODEL), embeddings=emb)

    def search(self, query: str, k: int = 5) -> List[RetrievedChunk]:
        import numpy as np
        if self._emb is None:
            self.build()
        q = self._encode([query])[0]           # normalised
        sims = self._emb @ q                    # cosine (both normalised)
        top = np.argsort(-sims)[:k]
        return [self._wrap(int(i), sims[int(i)]) for i in top]


# ── NVIDIA NeMo Retriever embedding backend (NIM, OOTB) ──────────────────────
class NIMEmbeddingRetriever(Retriever):
    """Same interface as EmbeddingRetriever, but embeddings come from the NVIDIA
    NeMo Retriever embedding NIM (hosted or self-hosted) via `nv_retrieval`.

    Uses the model's asymmetric encoding: passages at index-build time, queries at
    search time. On-disk format matches EmbeddingRetriever (embeddings.npy +
    corpus.jsonl + meta.json) so it drops into the existing tool wiring.
    """

    def __init__(self, corpus, model_name: Optional[str] = None, embeddings=None, embedder=None):
        super().__init__(corpus)
        from nv_retrieval import NIMEmbedder, DEFAULT_EMBED_MODEL  # local import
        self.model_name = model_name or DEFAULT_EMBED_MODEL
        self._embedder = embedder or NIMEmbedder(model=self.model_name)
        self._emb = embeddings

    def build(self) -> "NIMEmbeddingRetriever":
        self._emb = self._embedder.embed_passages(self.texts)
        return self

    def save(self, index_dir: Path) -> None:
        import numpy as np
        index_dir.mkdir(parents=True, exist_ok=True)
        np.save(index_dir / "embeddings.npy", self._emb)
        with (index_dir / "corpus.jsonl").open("w", encoding="utf-8") as f:
            for c in self.corpus:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        (index_dir / "meta.json").write_text(json.dumps(
            {"backend": "nim", "model": self.model_name, "n": len(self.ids)}))

    @classmethod
    def load(cls, index_dir: Path) -> "NIMEmbeddingRetriever":
        import numpy as np
        corpus = load_corpus(index_dir / "corpus.jsonl")
        emb = np.load(index_dir / "embeddings.npy")
        meta = json.loads((index_dir / "meta.json").read_text())
        return cls(corpus, model_name=meta.get("model"), embeddings=emb)

    def search(self, query: str, k: int = 5) -> List[RetrievedChunk]:
        import numpy as np
        if self._emb is None:
            self.build()
        q = self._embedder.embed_queries([query])[0]   # normalised query vector
        sims = self._emb @ q                            # cosine (both normalised)
        top = np.argsort(-sims)[:k]
        return [self._wrap(int(i), sims[int(i)]) for i in top]


# ── lexical fallback (dependency-free) ───────────────────────────────────────
class LexicalRetriever(Retriever):
    """TF-IDF-ish cosine over token counts; no external deps."""

    def __init__(self, corpus):
        super().__init__(corpus)
        self._docs = [self._tok(t) for t in self.texts]
        df: Dict[str, int] = {}
        for toks in self._docs:
            for w in set(toks):
                df[w] = df.get(w, 0) + 1
        n = max(1, len(self._docs))
        self._idf = {w: math.log(1 + n / c) for w, c in df.items()}
        self._vecs = [self._vec(toks) for toks in self._docs]
        self._norms = [math.sqrt(sum(v * v for v in vec.values())) or 1.0 for vec in self._vecs]

    @staticmethod
    def _tok(text: str) -> List[str]:
        return _TOKEN_RE.findall(text.lower())

    def _vec(self, toks: Sequence[str]) -> Dict[str, float]:
        tf: Dict[str, float] = {}
        for w in toks:
            tf[w] = tf.get(w, 0.0) + 1.0
        return {w: c * self._idf.get(w, 0.0) for w, c in tf.items()}

    def search(self, query: str, k: int = 5) -> List[RetrievedChunk]:
        qv = self._vec(self._tok(query))
        qn = math.sqrt(sum(v * v for v in qv.values())) or 1.0
        scored = []
        for i, (dv, dn) in enumerate(zip(self._vecs, self._norms)):
            dot = sum(qv.get(w, 0.0) * v for w, v in dv.items())
            scored.append((i, dot / (qn * dn)))
        scored.sort(key=lambda x: -x[1])
        return [self._wrap(i, s) for i, s in scored[:k]]


# ── factory + grounding signal ───────────────────────────────────────────────
def make_retriever(corpus, backend: str = "nim") -> Retriever:
    if backend == "nim":
        try:
            import httpx  # noqa: F401
            import os
            if os.environ.get("NVIDIA_API_KEY"):
                return NIMEmbeddingRetriever(corpus).build()
            print("[retriever] NVIDIA_API_KEY not set; falling back to lexical. "
                  "Set the key to use the NeMo Retriever embedding NIM.")
        except ImportError:
            print("[retriever] httpx not installed; falling back to lexical.")
        return LexicalRetriever(corpus)
    if backend == "embedding":
        try:
            import sentence_transformers  # noqa: F401
            return EmbeddingRetriever(corpus).build()
        except ImportError:
            print(f"[retriever] sentence-transformers not installed; "
                  f"falling back to lexical. `pip install sentence-transformers` "
                  f"to use {EMBED_MODEL}.")
    return LexicalRetriever(corpus)


def gold_rank(results: Sequence[RetrievedChunk], gold_section_ids: Sequence[str]) -> Optional[int]:
    """1-indexed rank of the first result whose section_id is in the gold set.

    This is the grounding signal: it tells you whether (and how easily) the
    assistant could retrieve its way to the answer chunks. None = gold not in
    the returned window (a candidate for guided injection / quality filtering).
    """
    gold = {str(g) for g in gold_section_ids}
    for rank, r in enumerate(results, start=1):
        if r.section_id in gold:
            return rank
    return None


# ── CLI ──────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="Retriever build / query utility.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="encode a corpus and cache the index")
    b.add_argument("--chunks", required=True, type=Path)
    b.add_argument("--index", required=True, type=Path)
    b.add_argument("--backend", default="nim", choices=["nim", "embedding", "lexical"])

    q = sub.add_parser("query", help="run a smoke-test query")
    q.add_argument("--index", type=Path, help="cached embedding index dir")
    q.add_argument("--chunks", type=Path, help="corpus JSONL (for lexical / no cache)")
    q.add_argument("--backend", default="nim", choices=["nim", "embedding", "lexical"])
    q.add_argument("--q", required=True)
    q.add_argument("--k", type=int, default=5)
    q.add_argument("--gold", nargs="*", default=[], help="gold section ids to score rank")

    args = ap.parse_args()

    if args.cmd == "build":
        corpus = load_corpus(args.chunks)
        if args.backend == "nim":
            NIMEmbeddingRetriever(corpus).build().save(args.index)
            print(f"[retriever] built NIM index ({len(corpus)} chunks) -> {args.index}")
        elif args.backend == "embedding":
            EmbeddingRetriever(corpus).build().save(args.index)
            print(f"[retriever] built embedding index ({len(corpus)} chunks) -> {args.index}")
        else:
            print("[retriever] lexical backend needs no cache; use `query --chunks`.")
        return

    # query
    _loaders = {"embedding": EmbeddingRetriever, "nim": NIMEmbeddingRetriever}
    if args.index and (args.index / "meta.json").exists() and args.backend in _loaders:
        r = _loaders[args.backend].load(args.index)
    else:
        corpus = load_corpus(args.chunks)
        r = make_retriever(corpus, backend=args.backend)

    results = r.search(args.q, k=args.k)
    for rank, rc in enumerate(results, 1):
        print(f"{rank:2d}. [{rc.score:.3f}] art {rc.section_id:>5} — {rc.section_title[:55]}")
    if args.gold:
        print(f"gold_rank({args.gold}) = {gold_rank(results, args.gold)}")


if __name__ == "__main__":
    main()
