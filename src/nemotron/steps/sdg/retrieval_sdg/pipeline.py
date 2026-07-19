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


# engine-block keys that are I/O, not ConversationSimulatorConfig knobs (popped before splat)
_ENGINE_IO = {"input", "output", "column_name", "metadata_fields", "resume", "artifact_path"}


def _resolve(base: Path, p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (base / pp).resolve()


def exp_paths(cfg: Dict[str, Any], base: Path) -> Dict[str, Path]:
    """Every run artifact lives under <exp_root>/<exp_name>/. Paths derive from exp_name
    (no per-stage path config). The corpus (query_gen.chunks_path) stays external."""
    root = _resolve(base, str(cfg.get("exp_root", "../experiments")))
    exp = root / str(cfg.get("exp_name") or "default")
    out = exp / "output"
    return {"exp": exp, "output": out, "artifacts": exp / "artifacts",
            "queries": out / "queries.jsonl", "seeds": out / "seeds.jsonl",
            "raw": out / "raw.jsonl", "sft": out / "sft.jsonl", "summary": out / "summary.json"}


def _build_model_clients(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    prov = {p["name"]: p for p in cfg.get("providers", [])}
    for m in cfg.get("models", []):
        p = prov.get(m.get("provider"), {})
        out[m["alias"]] = {"model": m["model"], "base_url": p.get("endpoint", ""),
                           "api_key_env": p.get("api_key", "NVIDIA_API_KEY"),
                           "params": dict(m.get("inference_parameters", {}))}
    return out


# ── Stage 0: query_gen (optional; corpus -> synthesized seed queries) ─────────
def run_query_gen(cfg: Dict[str, Any], base: Path, limit: Optional[int]) -> Path:
    qg = cfg.get("query_gen", {})
    source = qg.get("source", "lancedb")
    if source == "jsonl" and not qg.get("chunks_path"):
        raise SystemExit("[query_gen] source: jsonl needs query_gen.chunks_path (the corpus JSONL).")
    if source == "lancedb" and not qg.get("lancedb", {}).get("uri"):
        raise SystemExit("[query_gen] source: lancedb needs query_gen.lancedb.uri + table.")
    chunks_path = str(_resolve(base, qg["chunks_path"])) if qg.get("chunks_path") else ""
    out_path = exp_paths(cfg, base)["queries"]
    models = _build_model_clients(cfg)

    client = None
    r = cfg.get("retrieval", {})
    if qg.get("validate", True) and r.get("endpoint"):
        from retrieval_sdg.retrieval.client import HttpRetrievalClient
        client = HttpRetrievalClient(r["endpoint"], oversample_factor=r.get("oversample_factor", 2),
                                     timeout=r.get("timeout", 30), field_map=r.get("field_map", {}),
                                     headers=r.get("headers") or None,
                                     max_retries=r.get("max_retries", 2))
    elif qg.get("validate", True):
        print("[query_gen] WARNING: validate requested but no retrieval.endpoint — "
              "queries will NOT be answerability-checked.")

    cl = qg.get("clustering", {})
    from retrieval_sdg.query_gen import run_query_gen as _run
    seeds = _run(models,
                 source=source,
                 lancedb_cfg=qg.get("lancedb"),
                 chunks_path=chunks_path,
                 coarse_cfg=qg.get("coarse"),
                 model_alias=qg.get("model_alias", "user_model"),
                 field_map=qg.get("field_map"),
                 n_queries=int(limit or qg.get("n_queries", 400)),
                 queries_per_cluster=int(qg.get("queries_per_cluster", 4)),
                 chunks_per_cluster=qg.get("chunks_per_cluster"),
                 pool_size=qg.get("pool_size"),          # None => derived from n_queries
                 candidate_headroom=float(qg.get("candidate_headroom", 1.5)),
                 max_rounds=int(qg.get("max_rounds", 3)),
                 embed_cfg=qg.get("embedding"),
                 cluster_algo=cl.get("algo", "kmeans"), n_clusters=cl.get("k"),
                 min_cluster_size=int(cl.get("min_cluster_size", 5)),
                 kind_weights=qg.get("kind_weights"),
                 multi_hop_chunks=int(qg.get("multi_hop_chunks", 3)),
                 cross_doc=bool(qg.get("cross_doc", True)),
                 client=client, top_k=int(r.get("top_k", 4)),
                 validate=qg.get("validate", True) and client is not None,
                 min_coverage=float(qg.get("min_coverage", 0.35)),
                 seed=int(qg.get("seed", 7)), max_workers=int(qg.get("max_workers", 8)),
                 dry_run=bool(qg.get("_dry_run", False)))
    if qg.get("_dry_run"):
        return out_path                                    # diagnostics only; nothing written
    if not seeds:                                          # never clobber input_path with an empty file
        raise SystemExit(f"[query_gen] produced 0 queries; refusing to overwrite {out_path}. "
                         "Check the source, endpoint, and validation settings.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for s in seeds:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"[query_gen] wrote {len(seeds)} queries -> {out_path}")
    return out_path


# ── Stage A: query_prep ───────────────────────────────────────────────────────
def run_query_prep(cfg: Dict[str, Any], base: Path, limit: Optional[int]) -> Path:
    qp = cfg.get("query_prep", {})
    field = qp.get("query_field", "query")
    p = exp_paths(cfg, base)
    inp, seeds_path = p["queries"], p["seeds"]           # reads query_gen's queries; missing => error
    if not inp.exists():
        raise SystemExit(f"[query_prep] queries not found: {inp}\n"
                         "  run --stage query_gen first (or drop your queries there).")

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

    # persona is sampled at generation time (DD Person sampler); trajectory shape by the planner.
    tools_json = json.dumps(cfg["tools"], ensure_ascii=False)
    seeds_path.parent.mkdir(parents=True, exist_ok=True)
    with seeds_path.open("w", encoding="utf-8") as f:
        for r in picked:
            f.write(json.dumps({"query": str(r.get(field, "")), "cluster_id": r.get("cluster_id", ""),
                                "kind": r.get("kind", ""),   # -> planner opening kind (vague => clarify)
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

    # one persona per row (constant per trajectory) via DD's native Person sampler.
    # Locale must be downloaded once: `data-designer download personas --locale <locale>`.
    pcfg = cfg.get("persona", {})
    if pcfg.get("enabled", True):
        cb.add_column(dd.SamplerColumnConfig(
            name="persona",                              # matches ConversationSimulatorConfig.persona_column
            sampler_type=dd.SamplerType.PERSON,
            params=dd.PersonSamplerParams(
                locale=pcfg.get("locale", "en_IN"),
                with_synthetic_personas=bool(pcfg.get("with_synthetic_personas", True)))))

    r = cfg.get("retrieval", {})
    eng = cfg.get("engine", {})
    knobs: Dict[str, Any] = {k: v for k, v in eng.items() if k not in _ENGINE_IO}   # I/O keys aren't config knobs
    knobs.update(retrieval_endpoint=r.get("endpoint", ""), retrieval_tools=r.get("tools", ["search"]),
                 retrieval_field_map=r.get("field_map", {}), retrieval_timeout=r.get("timeout", 30),
                 retrieval_max_retries=r.get("max_retries", 2),
                 retrieval_headers=r.get("headers", {}), top_k=r.get("top_k", 4),
                 oversample_factor=r.get("oversample_factor", 2))
    knobs["model_clients"] = _build_model_clients(cfg)
    name = eng.get("column_name", cfg.get("column_name", "conversation_messages"))
    cb.add_column(ConversationSimulatorConfig(name=name, **knobs))
    return cb


def run_generate(cfg: Dict[str, Any], base: Path, seed_path: Path, limit: Optional[int]) -> Path:
    import data_designer.config as dd
    from data_designer.config.run_config import ResumeMode
    from data_designer.interface import DataDesigner

    eng = cfg.get("engine", {})
    cb = build_config_builder(cfg, seed_path)
    providers = [dd.ModelProvider(**p) for p in cfg.get("providers", [])]

    # resume from DD's per-row-group checkpoint (stable artifact_path + dataset_name).
    # if_possible: resume when the config matches, else restart. buffer_size left default
    # on both runs so it lines up.
    resume = ResumeMode(str(eng.get("resume", "never")).lower())
    artifact_path = exp_paths(cfg, base)["artifacts"] / "generate"
    artifact_path.mkdir(parents=True, exist_ok=True)
    client = DataDesigner(model_providers=providers, artifact_path=artifact_path) if providers \
        else DataDesigner(artifact_path=artifact_path)

    n_seeds = sum(1 for l in seed_path.open(encoding="utf-8") if l.strip())
    n = min(limit, n_seeds) if limit else n_seeds
    if resume != ResumeMode.NEVER:
        print(f"[generate] resume={resume.value} artifacts={artifact_path} (re-run this stage to resume)")
    result = client.create(cb, num_records=n, dataset_name="retrieval_sdg", resume=resume)
    records = _records(result)

    out = exp_paths(cfg, base)["raw"]
    out.parent.mkdir(parents=True, exist_ok=True)
    tools = cfg["tools"]
    meta = eng.get("metadata_fields", cfg.get("metadata_fields",
                   ["kind", "difficulty", "cluster_id", "hops_taken",
                    "conversation_status", "trajectory_judgment", "retrieval_log"]))
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
    print("    python evaluate.py --config config/pipeline.yaml --judge")
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
    ap.add_argument("--stage", choices=["query_gen", "query_prep", "generate", "all"], default="all")
    ap.add_argument("--limit", type=int, default=None, help="cap generated queries / sampled seeds / rows")
    ap.add_argument("--dry-run", action="store_true",
                    help="query_gen only: embed+cluster+sample and print diagnostics, no LLM/tokens, no write")
    ap.add_argument("--resume", choices=["never", "if_possible", "always"], default=None,
                    help="generate stage: resume an interrupted run from its last completed "
                         "row-group checkpoint (overrides engine.resume in the config)")
    args = ap.parse_args()

    from omegaconf import OmegaConf
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    base = args.config.resolve().parent
    if args.resume is not None:                         # CLI flag overrides engine.resume
        cfg.setdefault("engine", {})["resume"] = args.resume

    # all outputs live under experiments/<exp_name>/. Warn if it already exists (reuse
    # overwrites; change exp_name to keep both).
    P = exp_paths(cfg, base)
    if not args.dry_run:
        if P["output"].exists() and any(P["output"].iterdir()):
            print(f"[pipeline] ⚠️  experiment '{cfg.get('exp_name') or 'default'}' exists at {P['exp']} "
                  "— outputs will be overwritten (change exp_name to keep both).")
        P["output"].mkdir(parents=True, exist_ok=True)

    # Stage 0 (query_gen) is opt-in: explicit --stage query_gen, or "all" when a corpus
    # source (lancedb uri or chunks_path) is configured.
    _qg = cfg.get("query_gen", {})
    _has_source = _qg.get("lancedb", {}).get("uri") or _qg.get("chunks_path")
    if args.stage == "query_gen" or (args.stage == "all" and _has_source):
        if args.dry_run:
            cfg.setdefault("query_gen", {})["_dry_run"] = True
        run_query_gen(cfg, base, args.limit)            # writes query_gen.output; --limit caps it under `all`
        if args.stage == "query_gen" or args.dry_run:   # dry-run is diagnostics-only; never fall into generate
            return

    # `n_queries` is the single count knob: when query_gen feeds query_prep, it is
    # authoritative — query_prep dedups but must not silently re-cap below it.
    if _has_source:
        cfg.setdefault("query_prep", {}).setdefault("sample", {})["n_target"] = int(_qg.get("n_queries", 400))

    seeds_path = P["seeds"]                              # --stage generate reads seeds from the exp folder
    if args.stage in ("query_prep", "all"):
        seeds_path = run_query_prep(cfg, base, args.limit)
    if args.stage in ("generate", "all"):
        if not seeds_path.exists():
            raise SystemExit(f"[pipeline] no seeds at {seeds_path}; run --stage query_prep first.")
        run_generate(cfg, base, seeds_path, args.limit)


if __name__ == "__main__":
    main()
