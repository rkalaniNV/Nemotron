#!/usr/bin/env python3
"""Stage 1 — whole-document clustering (NO chunking).

Embed each *document* as a single unit and cluster the documents into topical
groups. Each cluster becomes a self-contained retrieval world downstream: the
streaming pipeline chunks, indexes, generates questions, and runs trajectories
for ONE cluster at a time, then tears the index down — so peak memory/disk is
bounded to a single cluster instead of the whole corpus.

Two artifacts:
  data/clusters/manifest.jsonl      one row PER CLUSTER: {cluster_id, doc_ids,...}
                                    -> the global, persistent audit/traceability
                                       ledger of which docs share a cluster.
  data/clusters/<cluster_id>/docs.jsonl   the documents in that cluster (input to
                                          Stage 2 chunk+index+question-gen).

Document unit (``--doc-unit``):
  file     : each input file (or each row of an --input JSONL) is one document.
  section  : split a single long document into sections (via a chunker profile's
             section regex) and treat each section as a document — lets the
             shipped single-file Constitution example cluster by article/topic.

Clustering (``--algo``): hdbscan (default, discovers k) | kmeans | agglomerative.

Usage:
    # single statute -> cluster its articles by topic
    python cluster_documents.py --input ../data/constitution_of_india.txt \
        --doc-unit section --profile indian_statute \
        --output-root ../data/clusters --algo hdbscan --min-cluster-size 4

    # a folder of documents
    python cluster_documents.py --input ../data/corpus_dir \
        --doc-unit file --output-root ../data/clusters --algo kmeans --k 12
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Prefer the LOCAL sibling package over any (possibly stale) editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# reuse the chunker's structural sectionizer for --doc-unit section
from chunk_document import PROFILES, _sectionize, _slice_body  # type: ignore  # noqa: E402

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]{2,}")
_STOP = set(
    "the a an and or of to in for on with by as at is are be this that these those "
    "shall may not no any such which who whom whose it its their his her they them "
    "from under upon into within without other than also more most only where when "
    "person state law right rights article section clause part".split()
)


@dataclass
class Document:
    doc_id: str
    text: str
    source: str = ""
    meta: Dict[str, object] = field(default_factory=dict)


# ── document loading (the three input shapes) ────────────────────────────────
def load_documents(input_path: Path, doc_unit: str, profile: str) -> List[Document]:
    if input_path.is_dir():
        docs = []
        for p in sorted(input_path.rglob("*")):
            if p.is_file() and p.suffix.lower() in (".txt", ".md"):
                docs.append(Document(doc_id=p.stem, text=p.read_text(encoding="utf-8", errors="replace"),
                                     source=str(p)))
        return docs

    if input_path.suffix.lower() in (".jsonl", ".json"):
        docs = []
        for i, line in enumerate(input_path.open(encoding="utf-8")):
            if not line.strip():
                continue
            r = json.loads(line)
            docs.append(Document(doc_id=str(r.get("doc_id", r.get("id", i))),
                                 text=r.get("text", ""), source=str(input_path),
                                 meta={k: v for k, v in r.items() if k not in ("doc_id", "id", "text")}))
        return docs

    # single text file
    raw = input_path.read_text(encoding="utf-8", errors="replace")
    if doc_unit == "section":
        prof = PROFILES[profile]
        body = _slice_body(raw, prof)
        docs = []
        for sec_id, sec_title, headings, sec_text in _sectionize(body, prof):
            docs.append(Document(doc_id=str(sec_id), text=sec_text, source=str(input_path),
                                 meta={"section_title": sec_title, "headings": headings}))
        return docs
    # doc-unit=file on a single file -> one document
    return [Document(doc_id=input_path.stem, text=raw, source=str(input_path))]


# ── embedding (whole doc, no chunking) ───────────────────────────────────────
def embed_documents(docs: Sequence[Document], model_name: str, max_chars: int,
                    window_chars: int = 1200):
    """One vector per WHOLE document via mean-pooling.

    MiniLM only attends to ~256 tokens, so embedding a long document's leading
    slice would represent it by just its opening. Instead we split each document
    into ``window_chars`` windows (up to ``max_chars`` total), embed all windows,
    and mean-pool them into one document vector reflecting the whole content.
    Short documents collapse to a single window (unchanged). This windowing is
    embedding-internal only — it is not corpus chunking and does not affect
    retrieval."""
    import numpy as np
    from agentic_rag.retrieval import EMBED_MODEL  # shared default
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name or EMBED_MODEL)

    windows: List[str] = []
    spans: List[tuple] = []  # (start, end) window range per document
    for d in docs:
        text = d.text[:max_chars]
        start = len(windows)
        for i in range(0, max(1, len(text)), window_chars):
            windows.append(text[i:i + window_chars] or " ")
        spans.append((start, len(windows)))

    win_emb = model.encode(windows, batch_size=64, show_progress_bar=False,
                           convert_to_numpy=True, normalize_embeddings=True)
    doc_vecs = []
    for a, b in spans:
        v = win_emb[a:b].mean(axis=0)
        doc_vecs.append(v / (np.linalg.norm(v) or 1.0))
    return np.stack(doc_vecs).astype("float32")


# ── clustering backends (swappable via --algo) ───────────────────────────────
def cluster_embeddings(emb, algo: str, k: Optional[int], min_cluster_size: int) -> List[int]:
    import numpy as np
    n = len(emb)
    if n <= 1:
        return [0] * n
    if algo == "kmeans":
        from sklearn.cluster import KMeans
        kk = _resolve_k(k, n)
        labels = KMeans(n_clusters=kk, n_init=10, random_state=7).fit_predict(emb)
        return labels.tolist()
    if algo == "agglomerative":
        from sklearn.cluster import AgglomerativeClustering
        if k:
            labels = AgglomerativeClustering(n_clusters=min(k, n)).fit_predict(emb)
        else:  # discover k via a cosine distance threshold on normalised vectors
            labels = AgglomerativeClustering(
                n_clusters=None, distance_threshold=0.5,
                metric="cosine", linkage="average").fit_predict(emb)
        return labels.tolist()
    # default: HDBSCAN (discovers k; labels noise as -1)
    from sklearn.cluster import HDBSCAN
    mcs = max(2, min(min_cluster_size, n))
    labels = HDBSCAN(min_cluster_size=mcs, metric="euclidean").fit_predict(emb)
    return _absorb_noise(np.asarray(emb), labels).tolist()


def _resolve_k(k: Optional[int], n: int) -> int:
    if k:
        return max(1, min(k, n))
    return max(2, min(n, round((n / 2) ** 0.5)))  # heuristic when k unspecified


def _absorb_noise(emb, labels):
    """HDBSCAN marks outliers as -1. Reassign each outlier to the nearest cluster
    centroid so every document is processed; if ALL points are noise, fall back
    to one singleton cluster per document."""
    import numpy as np
    labels = np.asarray(labels)
    clustered = labels >= 0
    if not clustered.any():
        return np.arange(len(labels))
    cluster_ids = sorted(set(labels[clustered].tolist()))
    centroids = np.stack([emb[labels == c].mean(axis=0) for c in cluster_ids])
    out = labels.copy()
    for i in np.where(~clustered)[0]:
        d = np.linalg.norm(centroids - emb[i], axis=1)
        out[i] = cluster_ids[int(d.argmin())]
    return out


# ── labels for the audit ledger ──────────────────────────────────────────────
def _cluster_label(docs: Sequence[Document], top: int = 5) -> str:
    counts: Counter = Counter()
    for d in docs:
        for w in _WORD_RE.findall(d.text.lower()):
            if w not in _STOP:
                counts[w] += 1
    return ", ".join(w for w, _ in counts.most_common(top))


# ── driver ───────────────────────────────────────────────────────────────────
def run(docs: List[Document], out_root: Path, model_name: str, algo: str,
        k: Optional[int], min_cluster_size: int, max_chars: int) -> Dict[str, List[str]]:
    if not docs:
        raise SystemExit("[cluster] no documents found for the given --input/--doc-unit.")
    emb = embed_documents(docs, model_name, max_chars)
    labels = cluster_embeddings(emb, algo, k, min_cluster_size)

    # group docs by cluster; re-id clusters to a stable 0..C-1 range
    groups: Dict[int, List[int]] = {}
    for idx, lab in enumerate(labels):
        groups.setdefault(int(lab), []).append(idx)
    remap = {old: new for new, old in enumerate(sorted(groups))}

    out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / "manifest.jsonl"
    cluster_to_docs: Dict[str, List[str]] = {}
    with manifest_path.open("w", encoding="utf-8") as mf:
        for old, member_idx in sorted(groups.items()):
            cid = f"c{remap[old]:03d}"
            members = [docs[i] for i in member_idx]
            cdir = out_root / cid
            cdir.mkdir(parents=True, exist_ok=True)
            with (cdir / "docs.jsonl").open("w", encoding="utf-8") as df:
                for d in members:
                    df.write(json.dumps({"doc_id": d.doc_id, "text": d.text,
                                         "source": d.source, **d.meta}, ensure_ascii=False) + "\n")
            doc_ids = [d.doc_id for d in members]
            cluster_to_docs[cid] = doc_ids
            mf.write(json.dumps({
                "cluster_id": cid, "n_docs": len(members), "doc_ids": doc_ids,
                "label": _cluster_label(members), "method": algo,
            }, ensure_ascii=False) + "\n")

    print(f"[cluster] {len(docs)} docs -> {len(cluster_to_docs)} clusters ({algo}) -> {out_root}")
    print(f"[cluster] audit ledger: {manifest_path}")
    for cid, ids in list(cluster_to_docs.items())[:8]:
        print(f"    {cid}: {len(ids)} docs")
    if len(cluster_to_docs) > 8:
        print(f"    … (+{len(cluster_to_docs) - 8} more)")
    return cluster_to_docs


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 1: whole-document clustering (no chunking).")
    ap.add_argument("--input", required=True, type=Path,
                    help="a text file, a JSONL of {doc_id,text}, or a directory of .txt/.md")
    ap.add_argument("--doc-unit", default="file", choices=["file", "section"],
                    help="section = split one file into sections (each a document)")
    ap.add_argument("--profile", default="plain", choices=sorted(PROFILES),
                    help="chunker profile used only for --doc-unit section splitting")
    ap.add_argument("--output-root", required=True, type=Path)
    ap.add_argument("--algo", default="hdbscan", choices=["hdbscan", "kmeans", "agglomerative"])
    ap.add_argument("--k", type=int, default=None, help="cluster count (kmeans/agglomerative)")
    ap.add_argument("--min-cluster-size", type=int, default=5, help="HDBSCAN min cluster size")
    ap.add_argument("--embedding-model", default="", help="override the sentence-transformers model")
    ap.add_argument("--max-chars", type=int, default=40000,
                    help="max chars per document fed to the (windowed, mean-pooled) embedder")
    args = ap.parse_args()

    docs = load_documents(args.input, args.doc_unit, args.profile)
    run(docs, args.output_root, args.embedding_model, args.algo, args.k,
        args.min_cluster_size, args.max_chars)


if __name__ == "__main__":
    main()
