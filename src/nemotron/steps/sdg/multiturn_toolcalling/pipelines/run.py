"""Live pipeline: queries.jsonl -> 20-25 turn tool-calling trajectories.

For each query the assistant LLM drives the live retrieve -> assess -> rewrite ->
answer loop against the real NeMo Retriever, the user LLM improvises follow-ups,
memory is read/written on request, and the context is auto-compacted at the token
threshold (never a tool call). Data Designer orchestrates: one row per query,
one generator column emitting ``structured_messages``.

Env:
  LLM_API_KEY   (required)  provider/proxy key for the generation LLM
  LLM_BASE_URL  (default https://inference-api.nvidia.com)
  LLM_MODEL     (default azure/openai/gpt-5.5)
  RETRIEVER_URL (default http://localhost:8000)
  SIM_THRESHOLD (default 32000) token threshold that triggers auto-compaction
  NUM_QUERIES   (default all)

Usage:
  python pipelines/run.py --queries data/queries.jsonl --out output/const_sft.jsonl
"""

from __future__ import annotations

import argparse
import json
import os

import pandas as pd

import data_designer.config as dd
from data_designer.interface import DataDesigner

from mtsdg.accept import filter_accepted
from mtsdg.generator_config import EpisodeSimulatorConfig
from mtsdg.model_configs import custom_openai_model_configs, custom_openai_provider
from mtsdg.retriever import RetrieverClient
from mtsdg.schemas import QuerySeed

BASE_URL = os.environ.get("LLM_BASE_URL", "https://inference-api.nvidia.com")
MODEL = os.environ.get("LLM_MODEL", "azure/openai/gpt-5.5")
RETRIEVER_URL = os.environ.get("RETRIEVER_URL", "http://localhost:8000")
SIM_THRESHOLD = int(os.environ.get("SIM_THRESHOLD", "32000"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default="data/queries.jsonl")
    ap.add_argument("--out", default="output/const_sft.jsonl")
    ap.add_argument("--num-queries", type=int, default=int(os.environ.get("NUM_QUERIES", "0")))
    ap.add_argument("--sim-threshold", type=int, default=SIM_THRESHOLD)
    args = ap.parse_args()

    if not os.environ.get("LLM_API_KEY", "").strip():
        print("LLM_API_KEY is empty — assuming a no-auth endpoint (no Authorization header sent).")
    os.makedirs("output", exist_ok=True)

    queries = [QuerySeed.model_validate_json(l) for l in open(args.queries, encoding="utf-8") if l.strip()]
    if args.num_queries:
        queries = queries[: args.num_queries]

    print(f"retriever health: {RetrieverClient(RETRIEVER_URL).health()}  url={RETRIEVER_URL}")

    # Seed dataset: one row per query, the QuerySeed serialized into episode_input.
    seed_path = "output/_episode_inputs.jsonl"
    pd.DataFrame([{"episode_input": q.model_dump_json()} for q in queries]).to_json(
        seed_path, orient="records", lines=True, force_ascii=False
    )

    # Incremental checkpoint: each episode is dumped here the moment it finishes,
    # so a crash/kill/long run still leaves usable output.
    ckpt_path = os.environ.get("CHECKPOINT_PATH", "output/checkpoint.jsonl")
    open(ckpt_path, "w").close()  # reset for this run
    print(f"checkpoint (incremental): {ckpt_path}")

    max_parallel = int(os.environ.get("MAX_PARALLEL", "2"))
    print(f"model={MODEL}  max_parallel_requests={max_parallel}")
    model_configs = custom_openai_model_configs(model=MODEL, max_parallel_requests=max_parallel)
    provider = custom_openai_provider(endpoint=BASE_URL)
    designer = DataDesigner(model_providers=[provider])

    builder = dd.DataDesignerConfigBuilder(model_configs=model_configs)
    builder.with_seed_dataset(dd.LocalFileSeedSource(path=seed_path), sampling_strategy=dd.SamplingStrategy.ORDERED)
    builder.add_column(EpisodeSimulatorConfig(
        name="conversation",
        episode_input_column="episode_input",
        retriever_url=RETRIEVER_URL,
        retrieve_top_k=int(os.environ.get("RETRIEVE_TOP_K", "3")),
        context_token_threshold=args.sim_threshold,
        max_reasoning_tokens=int(os.environ.get("MAX_REASONING_TOKENS", "400")),
        majority_vote_n=int(os.environ.get("MAJORITY_VOTE_N", "1")),
        run_inline_judge=os.environ.get("INLINE_JUDGE", "0") == "1",
        run_trajectory_judge=os.environ.get("TRAJ_JUDGE", "1") == "1",
        checkpoint_path=ckpt_path,
    ))

    print(f"\n=== Generating {len(queries)} trajectories (Data Designer) ===")
    print(f"  (tail -f {ckpt_path} to watch records land as they finish)")
    rows = designer.preview(builder, num_records=len(queries)).dataset.to_dict(orient="records")

    for row in rows:
        meta = json.loads(row["episode_metadata"])
        msgs = json.loads(row["structured_messages"])
        print(f"  {meta.get('query_id')}: n_messages={len(msgs)} "
              f"retrieved={meta.get('n_retrieved_chunks')} "
              f"compactions={len(meta.get('compaction_events', []))} "
              f"triggers={meta.get('compaction_triggers')} status={row['trajectory_status']}")

    split = filter_accepted(rows)
    with open(args.out, "w", encoding="utf-8") as fh:
        for row in split["accepted"]:
            fh.write(json.dumps({"messages": json.loads(row["structured_messages"])}, ensure_ascii=False) + "\n")
    with open("output/trajectories_full.json", "w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=2, default=str)
    print(f"\nAccepted {len(split['accepted'])}/{len(rows)} trajectories -> {args.out}")


if __name__ == "__main__":
    main()
