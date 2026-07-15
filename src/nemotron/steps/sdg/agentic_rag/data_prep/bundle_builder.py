#!/usr/bin/env python3
"""Evidence-set (bundle) builder for the agentic-RAG SDG pipeline.

Turns a flat chunk corpus into **multi-hop evidence sets**: small groups of
linked chunks that a single query can only be answered by synthesising across.
These bundles are the seeds for document-level query generation — this is how we
get genuine multi-hop depth from *just documents*, with no manual graph.

Config-driven modes (all domain-agnostic; the corpus supplies the signal):
  - ``entity_link``  : anchor + chunks it cross-references (uses metadata refs).
                       Best for legal text: each ref is a real dependency hop.
  - ``semantic``     : anchor + nearest neighbours by embedding similarity.
                       Needs a retriever; domain-agnostic fallback.
  - ``same_section`` : anchor + other sub-chunks of the same section (shallow).

Output: bundles JSONL, one evidence set per row, ready for query generation.

Usage:
    python bundle_builder.py --chunks ../data/constitution_chunks.jsonl \
        --output ../data/bundles.jsonl --mode entity_link \
        --size 3 --num 200 --seed 7
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence


@dataclass
class BundleConfig:
    """Every knob for bundle construction — populated from the outer NDD config."""
    mode: str = "entity_link"          # entity_link | semantic | same_section
    size: int = 3                      # target chunks per evidence set
    num_bundles: int = 200             # how many bundles to emit
    ref_field: str = "refs_article"    # metadata key holding cross-references
    section_field: str = "section_id"  # which field the refs point to
    min_size: int = 2                  # drop bundles smaller than this (need multi-hop)
    seed: int = 7
    semantic_index: Optional[str] = None   # embedding index dir (semantic mode)


@dataclass
class Bundle:
    bundle_id: str
    mode: str
    anchor_id: str
    member_ids: List[str]
    member_sections: List[str]
    size: int = 0
    hop_articles: List[str] = field(default_factory=list)


def load_corpus(path: Path) -> List[Dict[str, object]]:
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


# ── index helpers ────────────────────────────────────────────────────────────
def _by_section(corpus: List[Dict]) -> Dict[str, List[Dict]]:
    idx: Dict[str, List[Dict]] = {}
    for c in corpus:
        idx.setdefault(str(c.get("section_id", "")), []).append(c)
    return idx


def _refs_of(chunk: Dict, cfg: BundleConfig) -> List[str]:
    return [str(r) for r in (chunk.get("metadata", {}) or {}).get(cfg.ref_field, [])]


# ── modes ────────────────────────────────────────────────────────────────────
def _bundle_entity_link(anchor: Dict, corpus, by_sec, cfg: BundleConfig) -> Optional[Bundle]:
    """Anchor + one chunk from each cross-referenced section (a citation hop)."""
    refs = _refs_of(anchor, cfg)
    members = [anchor]
    hop_articles: List[str] = []
    for ref in refs:
        if ref == str(anchor.get(cfg.section_field)):
            continue
        cand = by_sec.get(ref)
        if cand:
            members.append(cand[0])          # representative chunk of that section
            hop_articles.append(ref)
        if len(members) >= cfg.size:
            break
    if len(members) < cfg.min_size:
        return None
    return _mk_bundle("entity_link", anchor, members, hop_articles)


def _bundle_same_section(anchor: Dict, corpus, by_sec, cfg: BundleConfig) -> Optional[Bundle]:
    sec = str(anchor.get(cfg.section_field))
    members = by_sec.get(sec, [anchor])[: cfg.size]
    if len(members) < cfg.min_size:
        return None
    return _mk_bundle("same_section", anchor, members, [sec])


def _bundle_semantic(anchor: Dict, retriever, cfg: BundleConfig) -> Optional[Bundle]:
    results = retriever.search(anchor.get("text", ""), k=cfg.size + 1)
    members, hop = [anchor], []
    for r in results:
        if r.chunk_id == anchor["chunk_id"]:
            continue
        members.append({"chunk_id": r.chunk_id, "section_id": r.section_id})
        hop.append(r.section_id)
        if len(members) >= cfg.size:
            break
    if len(members) < cfg.min_size:
        return None
    return _mk_bundle("semantic", anchor, members, hop)


def _mk_bundle(mode, anchor, members, hop_articles) -> Bundle:
    ids = [m["chunk_id"] for m in members]
    secs = [str(m.get("section_id", "")) for m in members]
    return Bundle(
        bundle_id=f"bnd_{anchor['chunk_id']}",
        mode=mode,
        anchor_id=anchor["chunk_id"],
        member_ids=ids,
        member_sections=secs,
        size=len(ids),
        hop_articles=hop_articles,
    )


# ── driver ───────────────────────────────────────────────────────────────────
def build_bundles(corpus: List[Dict], cfg: BundleConfig) -> List[Bundle]:
    rng = random.Random(cfg.seed)
    by_sec = _by_section(corpus)

    retriever = None
    if cfg.mode == "semantic":
        from retriever import EmbeddingRetriever, make_retriever  # local import
        if cfg.semantic_index and (Path(cfg.semantic_index) / "meta.json").exists():
            retriever = EmbeddingRetriever.load(Path(cfg.semantic_index))
        else:
            retriever = make_retriever(corpus, backend="embedding")

    # anchors: shuffle for diversity, prefer chunks that actually have links
    anchors = list(corpus)
    if cfg.mode == "entity_link":
        anchors = [c for c in anchors if _refs_of(c, cfg)] or anchors
    rng.shuffle(anchors)

    bundles: List[Bundle] = []
    seen_anchor: set = set()
    for anchor in anchors:
        if len(bundles) >= cfg.num_bundles:
            break
        if anchor["chunk_id"] in seen_anchor:
            continue
        seen_anchor.add(anchor["chunk_id"])
        if cfg.mode == "entity_link":
            b = _bundle_entity_link(anchor, corpus, by_sec, cfg)
        elif cfg.mode == "same_section":
            b = _bundle_same_section(anchor, corpus, by_sec, cfg)
        elif cfg.mode == "semantic":
            b = _bundle_semantic(anchor, retriever, cfg)
        else:
            raise ValueError(f"unknown bundle mode: {cfg.mode}")
        if b:
            bundles.append(b)
    return bundles


def main() -> None:
    ap = argparse.ArgumentParser(description="Build multi-hop evidence-set bundles.")
    ap.add_argument("--chunks", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--mode", default="entity_link", choices=["entity_link", "semantic", "same_section"])
    ap.add_argument("--size", type=int, default=3)
    ap.add_argument("--num", type=int, default=200)
    ap.add_argument("--min-size", type=int, default=2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--semantic-index", type=Path, default=None)
    args = ap.parse_args()

    cfg = BundleConfig(
        mode=args.mode, size=args.size, num_bundles=args.num,
        min_size=args.min_size, seed=args.seed,
        semantic_index=str(args.semantic_index) if args.semantic_index else None,
    )
    corpus = load_corpus(args.chunks)
    bundles = build_bundles(corpus, cfg)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for b in bundles:
            f.write(json.dumps(asdict(b), ensure_ascii=False) + "\n")

    sizes = [b.size for b in bundles]
    avg = sum(sizes) / len(sizes) if sizes else 0
    print(f"[{cfg.mode}] wrote {len(bundles)} bundles -> {args.output}")
    print(f"avg bundle size: {avg:.1f} | multi-hop (size>=2): {sum(s >= 2 for s in sizes)}")


if __name__ == "__main__":
    main()
