"""Validate and enrich raw or rich seed JSONL without corpus curation."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .config import PipelineConfig
from .schemas import ALLOWED_MEMORY_KEYS, EpisodeSeed, Persona


def _stable_id(record: dict[str, Any]) -> str:
    raw = json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return "q-" + hashlib.sha256(raw.encode()).hexdigest()[:16]


def _rng(run_seed: int, query_id: str) -> random.Random:
    digest = hashlib.sha256(f"{run_seed}|{query_id}".encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


def _weighted_depth(rng: random.Random, weights: dict[int, float]) -> int:
    depths = sorted(weights)
    return rng.choices(depths, weights=[weights[d] for d in depths], k=1)[0]


def enrich_seed(record: dict[str, Any], cfg: PipelineConfig) -> EpisodeSeed:
    query = str(record.get("query", "")).strip()
    if not query:
        raise ValueError("seed query must be a non-empty string")
    qid = str(record.get("query_id") or _stable_id(record))
    rng = _rng(cfg.run.seed, qid)
    turn_budget = record.get("turn_budget") if cfg.episode.honor_seed_turn_budget else None
    if turn_budget is None:
        turn_budget = rng.randint(cfg.episode.turn_budget.min, cfg.episode.turn_budget.max)
    depth = record.get("retrieval_depth")
    if depth is None:
        depth = _weighted_depth(rng, cfg.episode.retrieval_depth_weights)
    seed_instructions = str(record.get("instructions", "")).strip()
    instructions = "\n\n".join(x for x in (cfg.instructions.strip(), seed_instructions) if x)
    memory = dict(record.get("memory_seed") or {})
    invalid = sorted(set(memory) - ALLOWED_MEMORY_KEYS)
    if invalid:
        raise ValueError(f"memory_seed contains disallowed keys: {invalid}")
    return EpisodeSeed(
        query_id=qid,
        query=query,
        naive_query=str(record.get("naive_query") or query).strip(),
        persona=Persona.model_validate(record.get("persona") or {}),
        instructions=instructions,
        turn_budget=turn_budget,
        retrieval_depth=depth,
        memory_seed=memory,
        query_provenance=record.get("query_provenance"),
    )


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            yield value


def prepare_seed_file(cfg: PipelineConfig) -> int:
    src = cfg.resolve(cfg.paths.seeds)
    dst = cfg.resolve(cfg.paths.enriched_seeds)
    dst.parent.mkdir(parents=True, exist_ok=True)
    temporary = dst.with_suffix(dst.suffix + ".tmp")
    count = 0
    query_ids: set[str] = set()
    try:
        with temporary.open("w", encoding="utf-8") as out:
            for record in iter_jsonl(src):
                seed = enrich_seed(record, cfg)
                if seed.query_id in query_ids:
                    raise ValueError(f"duplicate query_id `{seed.query_id}`")
                query_ids.add(seed.query_id)
                out.write(json.dumps({"episode_input": seed.model_dump_json()}, ensure_ascii=False) + "\n")
                count += 1
        temporary.replace(dst)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return count
