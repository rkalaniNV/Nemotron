#!/usr/bin/env python3
"""Streaming orchestrator — runs Stages 1→6 ONE cluster at a time.

    Stage 1 (once):  cluster whole docs -> global manifest + per-cluster docs
    then FOR EACH cluster, sequentially:
      Stage 2:  chunk -> build the cluster's index -> generate seed questions
      Stage 3-6 (Data Designer, plugin column):  run trajectories scoped to THIS
                cluster's index, append them (tagged cluster_id) to the output
      teardown: delete the cluster's index (unless --keep-indexes)

Only the global cluster manifest persists across clusters, so peak memory/disk
is bounded to a single cluster instead of the whole corpus. A per-cluster
``.done`` marker makes the run resumable: rerun to continue where it stopped.

    python pipeline.py --config config/agentic_pipeline.yaml            # full run
    python pipeline.py --config config/agentic_pipeline.yaml --limit-clusters 2
    python pipeline.py --config config/agentic_pipeline.yaml --skip-clustering  # reuse clusters
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                 # prefer local package
sys.path.insert(0, str(HERE / "data_prep"))   # offline stage modules

import step as ddstep  # DD wiring helpers (reused; no domain logic)


# ── Stage 1 ───────────────────────────────────────────────────────────────────
def run_clustering(cl: Dict[str, Any], base: Path) -> Path:
    from cluster_documents import load_documents, run
    inp = _resolve(base, cl["input"])
    out_root = _resolve(base, cl["output_root"])
    docs = load_documents(inp, cl.get("doc_unit", "file"), cl.get("profile", "plain"))
    run(docs, out_root, cl.get("embedding_model", ""), cl.get("algo", "hdbscan"),
        cl.get("k"), cl.get("min_cluster_size", 5), cl.get("max_chars", 4000))
    return out_root


# ── Stage 2 (per cluster) ─────────────────────────────────────────────────────
def run_stage2(cluster_dir: Path, qg: Dict[str, Any], caller) -> Dict[str, int]:
    from question_gen import QGenConfig, run_cluster
    cfg = QGenConfig(
        profile=qg.get("profile", "plain"),
        chunk_max_chars=qg.get("chunk_max_chars", 2000),
        chunk_overlap=qg.get("chunk_overlap", 0),
        shard_max_chars=qg.get("shard_max_chars", 8000),
        n_queries=qg.get("n_queries", 4),
        max_docs=qg.get("max_docs"),
    )
    return run_cluster(cluster_dir, caller, cfg)


def build_index(cluster_dir: Path, backend: str) -> Path:
    from agentic_rag.retrieval import EmbeddingRetriever, load_corpus
    index_dir = cluster_dir / "index"
    corpus = load_corpus(cluster_dir / "chunks.jsonl")
    EmbeddingRetriever(corpus).build().save(index_dir)
    return index_dir


def make_stage2_caller(cfg: Dict[str, Any], qg: Dict[str, Any]):
    """Wire the OpenAI-compatible question-writer from the config's models."""
    from question_gen import make_openai_caller
    alias = qg.get("generator_alias", "assistant_model")
    model = next((m for m in cfg["models"] if m["alias"] == alias), cfg["models"][0])
    prov = next((p for p in cfg.get("providers", []) if p["name"] == model.get("provider")), None)
    endpoint = (prov or {}).get("endpoint", "https://inference-api.nvidia.com/v1")
    api_key_env = (prov or {}).get("api_key", "NVIDIA_API_KEY")
    return make_openai_caller(model["model"], endpoint, api_key_env,
                              params=model.get("inference_parameters", {}))


# ── Stages 3-6 (per cluster, Data Designer) ───────────────────────────────────
def _sample_queries(queries: List[Dict[str, Any]], n: int, cluster_id: str) -> List[Dict[str, Any]]:
    """Pick ``n`` diverse queries: round-robin across difficulty levels so the
    sample spans half-baked → complex rather than clumping. Deterministic per
    cluster (stable hash) for reproducibility."""
    import hashlib
    from collections import defaultdict
    seed = int(hashlib.sha256(cluster_id.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for q in queries:
        buckets[q.get("level", "?")].append(q)
    for b in buckets.values():
        rng.shuffle(b)
    order = sorted(buckets)                       # stable level order
    out: List[Dict[str, Any]] = []
    while len(out) < n and any(buckets[k] for k in order):
        for k in order:
            if buckets[k] and len(out) < n:
                out.append(buckets[k].pop())
    return out


def run_dd_for_cluster(cfg, base, cluster_dir, clusters_root, client, use_persona_sampler,
                       per_cluster_records: Optional[int]) -> List[Dict[str, Any]]:
    queries = cluster_dir / "queries.jsonl"
    if not queries.exists() or queries.stat().st_size == 0:
        print(f"[pipeline] {cluster_dir.name}: no queries, skipping DD.")
        return []
    tools_json = ddstep.cfg_tools(cfg)

    # subsample a few DIVERSE queries per cluster (keep queries.jsonl intact for audit)
    all_q = [json.loads(l) for l in queries.open(encoding="utf-8") if l.strip()]
    qg = cfg.get("question_gen", {})
    k = qg.get("queries_per_cluster")
    frac = qg.get("queries_fraction")
    if frac:                                   # take a fraction of generated queries
        k = max(1, round(len(all_q) * float(frac)))
    if k and len(all_q) > k:
        all_q = _sample_queries(all_q, k, cluster_dir.name)
        queries = cluster_dir / "_sampled_queries.jsonl"
        queries.write_text("\n".join(json.dumps(q, ensure_ascii=False) for q in all_q), encoding="utf-8")
        print(f"[pipeline] {cluster_dir.name}: sampled {len(all_q)} of the generated queries")

    seed_path = cluster_dir / "_seed.jsonl"
    n_rows = ddstep.enrich_query_seed(cfg, queries, seed_path,
                                      include_persona=not use_persona_sampler, tools_json=tools_json)
    if n_rows == 0:
        return []

    # point the simulator at the per-cluster index root (resolved in step.py)
    _plugin(cfg)["index_base_dir"] = str(clusters_root)
    cb = ddstep.build_config_builder(cfg, seed_path, base, use_persona_sampler)
    n = per_cluster_records or n_rows
    result = client.create(cb, num_records=n)
    return ddstep._records(result)


def _plugin(cfg):
    return ddstep._plugin_spec(cfg)


# ── driver ────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="Streaming agentic-RAG SDG pipeline (Stages 1-6).")
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--skip-clustering", action="store_true", help="reuse an existing clusters root")
    ap.add_argument("--keep-indexes", action="store_true", help="do not delete per-cluster indexes")
    ap.add_argument("--limit-clusters", type=int, default=None, help="process at most N clusters")
    ap.add_argument("--force", action="store_true", help="reprocess clusters even if marked .done")
    ap.add_argument("--no-llm", action="store_true", help="Stage 1-2 chunk/index only (no question-gen, no DD)")
    args = ap.parse_args()

    from omegaconf import OmegaConf
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    base = args.config.resolve().parent

    cl = cfg["clustering"]
    clusters_root = _resolve(base, cl["output_root"])
    if not args.skip_clustering:
        run_clustering(cl, base)
    if not (clusters_root / "manifest.jsonl").exists():
        raise SystemExit(f"[pipeline] no manifest at {clusters_root}; run without --skip-clustering.")

    # persona + DD client (built once, reused per cluster)
    locale = cfg.get("persona", {}).get("locale", "en_IN")
    use_persona_sampler = ddstep._persona_asset_available(locale)
    caller = None if args.no_llm else make_stage2_caller(cfg, cfg["question_gen"])

    client = None
    if not args.no_llm:
        import data_designer.config as dd
        from data_designer.interface import DataDesigner
        providers = [dd.ModelProvider(**p) for p in cfg.get("providers", [])]
        client = DataDesigner(model_providers=providers) if providers else DataDesigner()

    out = Path(cfg["output_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    per_cluster_records = cfg.get("question_gen", {}).get("per_cluster_records")

    cluster_dirs = sorted(p for p in clusters_root.iterdir() if p.is_dir() and p.name.startswith("c"))
    if args.limit_clusters:
        cluster_dirs = cluster_dirs[: args.limit_clusters]

    totals = {"clusters": 0, "generated": 0}
    for cdir in cluster_dirs:
        done_marker = cdir / ".done"
        if done_marker.exists() and not args.force:
            print(f"[pipeline] {cdir.name}: already done, skipping.")
            continue
        print(f"\n[pipeline] ===== cluster {cdir.name} =====")

        run_stage2(cdir, cfg["question_gen"], caller)
        index_dir = build_index(cdir, cfg["question_gen"].get("backend", "embedding"))

        if args.no_llm:
            print(f"[pipeline] {cdir.name}: chunk+index only (--no-llm).")
            continue

        records = run_dd_for_cluster(cfg, base, cdir, clusters_root, client,
                                     use_persona_sampler, per_cluster_records)
        # per-cluster RAW: every trajectory (SFT schema), written INTO the cluster
        # folder. Judging is a separate stage (evaluate.py) that reads these.
        cluster_raw = cdir / "trajectories.jsonl"
        n = ddstep.project_and_write(records, cfg, cluster_raw, keep_all=True)
        totals["clusters"] += 1
        totals["generated"] += n
        print(f"[pipeline] {cdir.name}: {n} trajectories -> {cluster_raw}")

        if not args.keep_indexes:
            shutil.rmtree(index_dir, ignore_errors=True)  # bound peak disk
        done_marker.write_text(json.dumps({"generated": n}))

    print(f"\n[pipeline] done: {totals['clusters']} clusters, {totals['generated']} trajectories")
    print(f"  per-cluster raw: {clusters_root}/<cluster>/trajectories.jsonl")
    print("  next — run the judge stage to consolidate + filter into output/sdg:")
    print(f"    python evaluate.py --clusters-root {clusters_root} --out {out} --judge")


def _resolve(base: Path, p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (base / pp).resolve()


if __name__ == "__main__":
    main()
