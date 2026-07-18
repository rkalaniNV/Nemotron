"""Deterministic candidate preparation before Data Designer model execution."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

from ..retrieval import RetrieverClient
from ..schemas import RetrievalChunk
from .allocation import scheduled_labels
from .config import QueryGenerationPipelineConfig
from .evidence import build_evidence_pools, sample_bundle
from .personas import persona_config_by_key, persona_weights
from .schemas import QueryCandidate, QueryTaxonomy, TaxonomyNode
from .taxonomy import load_taxonomy


def synthesis_fingerprint(cfg: QueryGenerationPipelineConfig, taxonomy_hash: str) -> str:
    aliases = {
        cfg.query_generation.generator_alias,
        cfg.query_generation.judge_alias,
    }
    payload = {
        "query_generation": cfg.query_generation.model_dump(mode="json"),
        "taxonomy_hash": taxonomy_hash,
        "retriever": cfg.retriever.model_dump(mode="json"),
        "models": [model.model_dump(mode="json") for model in cfg.models if model.alias in aliases],
        "run_seed": cfg.run.seed,
        "prompt_version": "query-synthesis-v1",
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_or_build_pools(
    cfg: QueryGenerationPipelineConfig,
    taxonomy: QueryTaxonomy,
    taxonomy_hash: str,
    fingerprint: str,
) -> dict[str, list[RetrievalChunk]]:
    path = cfg.resolve(cfg.paths.evidence_manifest)
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("synthesis_fingerprint") == fingerprint and payload.get("taxonomy_hash") == taxonomy_hash:
            return {
                key: [RetrievalChunk.model_validate(chunk) for chunk in chunks]
                for key, chunks in payload.get("pools", {}).items()
            }
    retriever = RetrieverClient(cfg.retriever)
    try:
        pools = build_evidence_pools(taxonomy, retriever, cfg.query_generation.evidence)
    finally:
        retriever.close()
    _write_json(
        path,
        {
            "synthesis_fingerprint": fingerprint,
            "taxonomy_hash": taxonomy_hash,
            "pools": {key: [chunk.model_dump(mode="json") for chunk in chunks] for key, chunks in pools.items()},
        },
    )
    return pools


def load_candidates(path: Path) -> list[QueryCandidate]:
    candidates: list[QueryCandidate] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                candidates.append(QueryCandidate.model_validate_json(row["candidate_input"]))
            except Exception as exc:
                raise ValueError(f"invalid query candidate at {path}:{line_number}: {exc}") from exc
    return candidates


def _existing_candidates(path: Path, fingerprint: str, count: int) -> list[QueryCandidate] | None:
    if not path.exists():
        return None
    candidates = load_candidates(path)
    if len(candidates) == count and all(item.synthesis_fingerprint == fingerprint for item in candidates):
        return candidates
    return None


def prepare_candidates(
    cfg: QueryGenerationPipelineConfig,
) -> tuple[list[QueryCandidate], str]:
    generation = cfg.query_generation
    taxonomy, taxonomy_hash = load_taxonomy(cfg.resolve(generation.taxonomy_path))
    fingerprint = synthesis_fingerprint(cfg, taxonomy_hash)
    candidate_path = cfg.resolve(cfg.paths.candidates)
    existing = _existing_candidates(candidate_path, fingerprint, generation.num_queries)
    if existing is not None:
        return existing, fingerprint

    weighted_leaves = taxonomy.weighted_leaves()
    leaves = [leaf for leaf, _ in weighted_leaves]
    leaf_by_id: dict[str, TaxonomyNode] = {leaf.id: leaf for leaf in leaves}
    rng = random.Random(int(fingerprint[:16], 16))
    topic_labels = scheduled_labels(
        generation.num_queries,
        {leaf.id: weight for leaf, weight in weighted_leaves},
        rng,
    )
    archetypes = scheduled_labels(generation.num_queries, generation.archetype_weights, rng)
    persona_modes = scheduled_labels(generation.num_queries, generation.persona_mode_weights, rng)
    scheduled_personas = scheduled_labels(
        generation.num_queries,
        persona_weights(generation.persona_locales),
        rng,
    )
    persona_configs = persona_config_by_key(generation.persona_locales)
    pools = _load_or_build_pools(cfg, taxonomy, taxonomy_hash, fingerprint)

    candidates: list[QueryCandidate] = []
    for index in range(generation.num_queries):
        node = leaf_by_id[topic_labels[index]]
        bundle_rng = random.Random(f"{fingerprint}|bundle|{index}")
        bundle = sample_bundle(
            pools[node.id],
            archetype=archetypes[index],
            cfg=generation.evidence,
            rng=bundle_rng,
        )
        query_id = "qg-" + hashlib.sha256(f"{fingerprint}|{index}".encode()).hexdigest()[:20]
        candidates.append(
            QueryCandidate(
                query_id=query_id,
                synthesis_fingerprint=fingerprint,
                candidate_index=index,
                taxonomy_id=node.id,
                taxonomy_label=node.label,
                taxonomy_description=node.description,
                taxonomy_required_terms=node.required_terms,
                archetype=archetypes[index],
                answerability=("insufficient" if archetypes[index] == "insufficient_evidence" else "answerable"),
                persona_mode=persona_modes[index],
                persona_key=scheduled_personas[index],
                persona_locale=persona_configs[scheduled_personas[index]].locale,
                language=persona_configs[scheduled_personas[index]].language,
                evidence=bundle,
            )
        )

    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = candidate_path.with_suffix(candidate_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for candidate in candidates:
            handle.write(
                json.dumps(
                    {"candidate_input": candidate.model_dump_json()},
                    ensure_ascii=False,
                )
                + "\n"
            )
    temporary.replace(candidate_path)
    return candidates, fingerprint
