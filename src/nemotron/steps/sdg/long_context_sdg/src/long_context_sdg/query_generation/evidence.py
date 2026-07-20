"""Taxonomy-stratified evidence-pool construction and bundle sampling."""

from __future__ import annotations

import hashlib
import random
from collections import Counter, defaultdict, deque
from typing import Any

from ..retrieval import RetrieverClient
from ..schemas import RetrievalChunk
from .config import QueryArchetypeProfile, QueryEvidenceConfig
from .schemas import QueryTaxonomy, TaxonomyNode


def content_hash(chunk: RetrievalChunk) -> str:
    return hashlib.sha256(chunk.content.encode()).hexdigest()


def _metadata_matches(chunk: RetrievalChunk, filters: dict[str, Any]) -> bool:
    values = {
        **chunk.metadata,
        "title": chunk.title,
        "source": chunk.source,
        "date": chunk.date,
    }
    for key, expected in filters.items():
        actual = values.get(key)
        if isinstance(expected, list):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True


def _eligible(chunk: RetrievalChunk, node: TaxonomyNode, cfg: QueryEvidenceConfig) -> bool:
    if len(chunk.content.strip()) < cfg.min_chunk_chars:
        return False
    haystack = " ".join((chunk.title, chunk.source, chunk.content)).casefold()
    if any(term.casefold() in haystack for term in node.exclusions):
        return False
    return _metadata_matches(chunk, node.metadata_filters)


def _source_diverse(chunks: list[RetrievalChunk], limit: int) -> list[RetrievalChunk]:
    by_source: dict[str, deque[RetrievalChunk]] = defaultdict(deque)
    for chunk in chunks:
        by_source[chunk.source or chunk.chunk_id].append(chunk)
    ordered_sources = list(by_source)
    selected: list[RetrievalChunk] = []
    while ordered_sources and len(selected) < limit:
        next_sources = []
        for source in ordered_sources:
            selected.append(by_source[source].popleft())
            if by_source[source]:
                next_sources.append(source)
            if len(selected) == limit:
                break
        ordered_sources = next_sources
    return selected


def build_evidence_pools(
    taxonomy: QueryTaxonomy,
    retriever: RetrieverClient,
    cfg: QueryEvidenceConfig,
) -> dict[str, list[RetrievalChunk]]:
    pools: dict[str, list[RetrievalChunk]] = {}
    for node in taxonomy.leaves():
        chunks: list[RetrievalChunk] = []
        seen_ids: set[str] = set()
        seen_hashes: set[str] = set()
        for query in node.seed_queries:
            for chunk in retriever.query(query, top_k=cfg.pool_size):
                digest = content_hash(chunk)
                if chunk.chunk_id in seen_ids or digest in seen_hashes or not _eligible(chunk, node, cfg):
                    continue
                seen_ids.add(chunk.chunk_id)
                seen_hashes.add(digest)
                chunks.append(chunk)
        pool = _source_diverse(chunks, cfg.pool_size)
        if len(pool) < cfg.bundle_min:
            raise ValueError(
                f"taxonomy leaf `{node.id}` has only {len(pool)} eligible chunks; needs at least {cfg.bundle_min}"
            )
        pools[node.id] = pool
    return pools


def sample_bundle(
    pool: list[RetrievalChunk],
    *,
    archetype: str,
    profile: QueryArchetypeProfile,
    cfg: QueryEvidenceConfig,
    rng: random.Random,
) -> list[RetrievalChunk]:
    del archetype
    minimum = max(profile.bundle_min, profile.min_sources)
    count = rng.randint(minimum, min(profile.bundle_max, len(pool)))
    shuffled = list(pool)
    rng.shuffle(shuffled)
    selected: list[RetrievalChunk] = []
    per_source: Counter[str] = Counter()

    def similarity(left: str, right: str) -> float:
        left_tokens = set(left.casefold().split())
        right_tokens = set(right.casefold().split())
        union = left_tokens | right_tokens
        return len(left_tokens & right_tokens) / len(union) if union else 0.0

    def add(chunk: RetrievalChunk) -> bool:
        source = chunk.source or chunk.chunk_id
        if per_source[source] >= cfg.max_per_source:
            return False
        if any(similarity(chunk.content, prior.content) > cfg.max_pair_similarity for prior in selected):
            return False
        selected.append(chunk)
        per_source[source] += 1
        return True

    source_order = list(dict.fromkeys(chunk.source or chunk.chunk_id for chunk in shuffled))
    rng.shuffle(source_order)
    for source in source_order:
        if len(per_source) >= profile.min_sources:
            break
        for chunk in shuffled:
            if (chunk.source or chunk.chunk_id) == source and add(chunk):
                break
    if len(selected) < count:
        for chunk in shuffled:
            if chunk in selected:
                continue
            add(chunk)
            if len(selected) == count:
                break
    if len(selected) < count:
        raise ValueError("evidence pool cannot satisfy source and semantic-diversity constraints")
    sources = {chunk.source or chunk.chunk_id for chunk in selected}
    if len(sources) < profile.min_sources:
        raise ValueError(
            f"archetype needs evidence from at least {profile.min_sources} source(s); found {len(sources)}"
        )
    return selected
