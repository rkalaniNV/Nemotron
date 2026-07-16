"""Modular retrieval — the swappable search backend behind the RAG tools.

This is the single source of truth for retrieval. The runtime plugin imports it
as ``agentic_rag.retrieval`` (no sys.path hacks); the offline CLI in
``data_prep/retriever.py`` re-exports it. To swap in a different retrieval
offering later (BM25 service, FAISS, a hosted vector DB), implement the
``Retriever`` interface and register it in ``make_retriever`` — nothing else in
the pipeline needs to change.

Backends shipped:
  - ``EmbeddingRetriever``  MiniLM (all-MiniLM-L6-v2) cosine over numpy. CPU-fast.
  - ``LexicalRetriever``    dependency-free TF-IDF cosine (fallback / tests).

Stage-3 knob — ``subsample``: retrieve ``oversample_factor * k`` candidates by
similarity, then RANDOMLY sample ``k`` of them. This deliberately makes a single
retrieval lossy so the agent must issue more queries / take more hops to gather
all the evidence — i.e. it pushes trajectories to be genuinely multi-step.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from dataclasses import dataclass
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
    doc_id: str = ""

    def to_tool_payload(self) -> Dict[str, object]:
        """Shape returned to the assistant as the tool response (domain-agnostic keys)."""
        payload = {
            "chunk_id": self.chunk_id,
            "section": self.section_id,
            "title": self.section_title,
            "text": self.text,
            "score": round(self.score, 4),
        }
        if self.doc_id:
            payload["doc_id"] = self.doc_id
        return payload


def load_corpus(path: Path) -> List[Dict[str, object]]:
    return [json.loads(l) for l in Path(path).open(encoding="utf-8") if l.strip()]


# ── base ─────────────────────────────────────────────────────────────────────
class Retriever:
    """Interface: build an index over corpus records, then search.

    ``search`` returns the top-k by similarity. ``retrieve`` wraps it with the
    optional Stage-3 oversample-then-random-subsample behaviour and is what the
    tool layer should call.
    """

    def __init__(self, corpus: List[Dict[str, object]]):
        self.corpus = corpus
        self.ids = [c["chunk_id"] for c in corpus]
        self.texts = [c.get("text", "") for c in corpus]
        self.sec_ids = [str(c.get("section_id", "")) for c in corpus]
        self.titles = [c.get("section_title", "") for c in corpus]
        self.doc_ids = [str(c.get("doc_id", "")) for c in corpus]

    def search(self, query: str, k: int = 5) -> List[RetrievedChunk]:  # pragma: no cover
        raise NotImplementedError

    def retrieve(
        self,
        query: str,
        k: int = 5,
        *,
        subsample: bool = False,
        oversample_factor: int = 2,
        rng: Optional[random.Random] = None,
    ) -> List[RetrievedChunk]:
        """Top-k, or (Stage 3) oversample ``k*oversample_factor`` then randomly
        keep ``k`` — order preserved by original similarity so the kept set still
        reads like a ranked result list, just with gaps."""
        if not subsample or oversample_factor <= 1:
            return self.search(query, k=k)
        pool = self.search(query, k=max(k, k * oversample_factor))
        if len(pool) <= k:
            return pool
        rng = rng or random.Random()
        keep_idx = sorted(rng.sample(range(len(pool)), k))
        return [pool[i] for i in keep_idx]

    def _wrap(self, idx: int, score: float) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=self.ids[idx], text=self.texts[idx], score=float(score),
            section_id=self.sec_ids[idx], section_title=self.titles[idx],
            doc_id=self.doc_ids[idx],
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
        model = self._load_model()
        emb = model.encode(list(texts), batch_size=64, show_progress_bar=False,
                           convert_to_numpy=True, normalize_embeddings=True)
        return emb.astype("float32")

    def build(self) -> "EmbeddingRetriever":
        self._emb = self._encode(self.texts)
        return self

    def save(self, index_dir: Path) -> None:
        import numpy as np
        index_dir = Path(index_dir)
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
        index_dir = Path(index_dir)
        corpus = load_corpus(index_dir / "corpus.jsonl")
        emb = np.load(index_dir / "embeddings.npy")
        meta = json.loads((index_dir / "meta.json").read_text())
        return cls(corpus, model_name=meta.get("model", EMBED_MODEL), embeddings=emb)

    def search(self, query: str, k: int = 5) -> List[RetrievedChunk]:
        import numpy as np
        if self._emb is None:
            self.build()
        if not self.ids:
            return []
        q = self._encode([query])[0]           # normalised
        sims = self._emb @ q                    # cosine (both normalised)
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
def make_retriever(corpus, backend: str = "embedding") -> Retriever:
    if backend == "embedding":
        try:
            import sentence_transformers  # noqa: F401
            return EmbeddingRetriever(corpus).build()
        except ImportError:
            print(f"[retriever] sentence-transformers not installed; "
                  f"falling back to lexical. `pip install sentence-transformers` "
                  f"to use {EMBED_MODEL}.")
    return LexicalRetriever(corpus)


def gold_rank(results: Sequence[RetrievedChunk], gold_section_ids: Sequence[str],
              gold_doc_ids: Optional[Sequence[str]] = None) -> Optional[int]:
    """1-indexed rank of the first result that hits the gold — by section id
    (structured corpora) OR by document id (works for ANY corpus). None = gold
    not in the returned window. Document-level match keeps grounding meaningful
    even when the chunker assigns no real section structure.
    """
    gold_s = {str(g) for g in gold_section_ids}
    gold_d = {str(g) for g in (gold_doc_ids or [])}
    for rank, r in enumerate(results, start=1):
        if r.section_id in gold_s or (gold_d and r.doc_id in gold_d):
            return rank
    return None


# ── CLI: build (optionally per-cluster) / query ──────────────────────────────
def _build_one(chunks: Path, index: Path, backend: str) -> int:
    corpus = load_corpus(chunks)
    if backend == "embedding":
        EmbeddingRetriever(corpus).build().save(index)
    else:
        print("[retriever] lexical backend needs no cache; use `query --chunks`.")
    return len(corpus)


def main() -> None:
    ap = argparse.ArgumentParser(description="Retriever build / query utility.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="encode a corpus and cache the index")
    b.add_argument("--chunks", type=Path, help="single corpus JSONL")
    b.add_argument("--index", type=Path, help="output index dir")
    b.add_argument("--clusters-root", type=Path,
                   help="root holding <id>/chunks.jsonl; builds one index per cluster")
    b.add_argument("--backend", default="embedding", choices=["embedding", "lexical"])

    q = sub.add_parser("query", help="run a smoke-test query")
    q.add_argument("--index", type=Path, help="cached embedding index dir")
    q.add_argument("--chunks", type=Path, help="corpus JSONL (for lexical / no cache)")
    q.add_argument("--backend", default="embedding", choices=["embedding", "lexical"])
    q.add_argument("--q", required=True)
    q.add_argument("--k", type=int, default=5)
    q.add_argument("--gold", nargs="*", default=[], help="gold section ids to score rank")

    args = ap.parse_args()

    if args.cmd == "build":
        # per-cluster: build one index under each cluster dir
        if args.clusters_root:
            root = args.clusters_root
            n_built = 0
            for cdir in sorted(p for p in root.iterdir() if p.is_dir()):
                chunks = cdir / "chunks.jsonl"
                if not chunks.exists():
                    continue
                n = _build_one(chunks, cdir / "index", args.backend)
                print(f"[retriever] cluster {cdir.name}: index over {n} chunks -> {cdir/'index'}")
                n_built += 1
            print(f"[retriever] built {n_built} per-cluster indexes under {root}")
            return
        n = _build_one(args.chunks, args.index, args.backend)
        if args.backend == "embedding":
            print(f"[retriever] built embedding index ({n} chunks) -> {args.index}")
        return

    # query
    if args.index and (args.index / "meta.json").exists() and args.backend == "embedding":
        r: Retriever = EmbeddingRetriever.load(args.index)
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
