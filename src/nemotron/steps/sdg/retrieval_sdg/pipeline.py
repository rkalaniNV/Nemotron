#!/usr/bin/env python3
"""Orchestrator — query_prep -> conversation generation -> raw trajectories.

    Stage A (query_prep, offline, MiniLM only):
        queries.jsonl (query source) -> dedup -> cluster -> sample -> seeds.jsonl
    Stage B (generate, Data Designer):
        seeds.jsonl -> per-row multi-turn trajectory (HTTP retrieval + inline judge)
                    -> output/sdg/*.raw.jsonl

Judging/filtering into the final SFT set is a SEPARATE, re-runnable stage
(``evaluate.py``) so you can re-judge without regenerating.

    python pipeline.py --config config/pipeline.yaml --stage query_prep
    python pipeline.py --config config/pipeline.yaml --stage generate
    python pipeline.py --config config/pipeline.yaml --stage all --limit 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from retrieval_sdg.query_prep import cluster_queries, dedup, sample  # noqa: E402


def _resolve(base: Path, p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (base / pp).resolve()


def _build_model_clients(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    prov = {p["name"]: p for p in cfg.get("providers", [])}
    for m in cfg.get("models", []):
        p = prov.get(m.get("provider"), {})
        out[m["alias"]] = {"model": m["model"], "base_url": p.get("endpoint", ""),
                           "api_key_env": p.get("api_key", "NVIDIA_API_KEY"),
                           "params": dict(m.get("inference_parameters", {}))}
    return out


# ── Stage A: query_prep ───────────────────────────────────────────────────────
def run_query_prep(cfg: Dict[str, Any], base: Path, limit: Optional[int]) -> Path:
    qp = cfg.get("query_prep", {})
    field = qp.get("query_field", "query")
    inp = _resolve(base, cfg["input_path"])
    seeds_path = _resolve(base, cfg["seeds_path"])

    rows = [json.loads(l) for l in inp.open(encoding="utf-8") if l.strip()]
    rows = [r if isinstance(r, dict) else {field: r} for r in rows]
    print(f"[query_prep] loaded {len(rows)} queries from {inp}")

    dd_cfg = qp.get("dedup", {})
    kept = dedup(rows, threshold=dd_cfg.get("threshold", 0.92),
                 model_name=dd_cfg.get("embed_model", ""), query_field=field)
    print(f"[query_prep] deduped -> {len(kept)}")

    cl = qp.get("clustering", {})
    labels = cluster_queries(kept, algo=cl.get("algo", "kmeans"), k=cl.get("k"),
                             min_cluster_size=cl.get("min_cluster_size", 5),
                             model_name=dd_cfg.get("embed_model", ""), query_field=field)
    for r, lab in zip(kept, labels):
        r["cluster_id"] = f"c{lab:03d}"
    n_clusters = len(set(labels))
    print(f"[query_prep] clustered -> {n_clusters} clusters")

    sm = qp.get("sample", {})
    n_target = limit or sm.get("n_target")
    picked = sample(kept, n_target, seed=sm.get("seed", 7))
    print(f"[query_prep] sampled -> {len(picked)}")

    # trajectory shape is sampled per-row by the conversation planner (no classify pass).
    tools_json = json.dumps(cfg["tools"], ensure_ascii=False)
    seeds_path.parent.mkdir(parents=True, exist_ok=True)
    with seeds_path.open("w", encoding="utf-8") as f:
        for r in picked:
            f.write(json.dumps({"query": str(r.get(field, "")), "cluster_id": r.get("cluster_id", ""),
                                "tools": tools_json}, ensure_ascii=False) + "\n")
    print(f"[query_prep] wrote {len(picked)} seeds -> {seeds_path}")
    return seeds_path


# ── Stage B: conversation generation (Data Designer) ──────────────────────────
def build_config_builder(cfg: Dict[str, Any], seed_path: Path):
    import data_designer.config as dd
    from retrieval_sdg.conversation.config import ConversationSimulatorConfig

    model_configs = [
        dd.ModelConfig(alias=m["alias"], model=m["model"], provider=m.get("provider", "nvidia"),
                       skip_health_check=True,
                       inference_parameters=dd.ChatCompletionInferenceParams(
                           **dict(m.get("inference_parameters", {}))))
        for m in cfg["models"]
    ]

    cb = dd.DataDesignerConfigBuilder(model_configs=model_configs)
    cb.with_seed_dataset(dd.LocalFileSeedSource(path=str(seed_path)),
                         sampling_strategy=dd.SamplingStrategy.SHUFFLE)

    r = cfg.get("retrieval", {})
    knobs: Dict[str, Any] = dict(cfg.get("engine", {}))
    knobs.update(retrieval_endpoint=r.get("endpoint", ""), retrieval_tools=r.get("tools", ["search"]),
                 retrieval_field_map=r.get("field_map", {}), retrieval_timeout=r.get("timeout", 30),
                 retrieval_headers=r.get("headers", {}), top_k=r.get("top_k", 4),
                 oversample_factor=r.get("oversample_factor", 2))
    knobs["model_clients"] = _build_model_clients(cfg)
    cb.add_column(ConversationSimulatorConfig(name=cfg.get("column_name", "conversation_messages"), **knobs))
    return cb


def run_generate(cfg: Dict[str, Any], base: Path, seed_path: Path, limit: Optional[int]) -> Path:
    import data_designer.config as dd
    from data_designer.interface import DataDesigner

    cb = build_config_builder(cfg, seed_path)
    providers = [dd.ModelProvider(**p) for p in cfg.get("providers", [])]
    client = DataDesigner(model_providers=providers) if providers else DataDesigner()

    n_seeds = sum(1 for l in seed_path.open(encoding="utf-8") if l.strip())
    n = min(limit, n_seeds) if limit else n_seeds
    result = client.create(cb, num_records=n)
    records = _records(result)

    out = _resolve(base, cfg["output_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    tools = cfg["tools"]
    meta = cfg.get("metadata_fields", ["kind", "difficulty", "cluster_id", "hops_taken",
                                       "conversation_status", "trajectory_judgment", "retrieval_log"])
    n_written = 0
    with out.open("w", encoding="utf-8") as f:
        for rec in records:
            src = rec.get("conversation_messages")
            try:
                messages = json.loads(src) if isinstance(src, str) else src
            except (json.JSONDecodeError, TypeError):
                continue
            if not messages:
                continue
            row = {"messages": messages, "tools": tools}
            for mf in meta:
                if mf in rec:
                    row[mf] = rec[mf]
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            n_written += 1
    print(f"[generate] {len(records)} generated, {n_written} trajectories -> {out}")
    print("  next — judge & filter into the final SFT set:")
    print(f"    python evaluate.py --input {out} --out output/sdg/retrieval_sdg.jsonl --judge")
    return out


def _records(result) -> List[Dict[str, Any]]:
    if hasattr(result, "load_dataset"):
        ds = result.load_dataset()
        return ds.to_dict("records") if hasattr(ds, "to_dict") else list(ds)
    for attr in ("dataset", "records", "data"):
        obj = getattr(result, attr, None)
        if obj is not None:
            return obj.to_dict("records") if hasattr(obj, "to_dict") else list(obj)
    return list(result)


# ── driver ────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="retrieval_sdg pipeline (query_prep + conversation gen).")
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--stage", choices=["query_prep", "generate", "all"], default="all")
    ap.add_argument("--limit", type=int, default=None, help="cap sampled seeds / generated rows")
    args = ap.parse_args()

    from omegaconf import OmegaConf
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    base = args.config.resolve().parent

    seeds_path = _resolve(base, cfg["seeds_path"])
    if args.stage in ("query_prep", "all"):
        seeds_path = run_query_prep(cfg, base, args.limit)
    if args.stage in ("generate", "all"):
        if not seeds_path.exists():
            raise SystemExit(f"[pipeline] no seeds at {seeds_path}; run --stage query_prep first.")
        run_generate(cfg, base, seeds_path, args.limit)


if __name__ == "__main__":
    main()
