# retrieval_sdg

Generate multi-turn, retrieval-grounded tool-calling conversations for supervised
fine-tuning (SFT). Each output row contains OpenAI-style `messages`, tool schemas, and
generation metadata.

![Code-derived flow diagram showing query preparation, Data Designer generation, the per-turn research loop, and decoupled evaluation.](assets/retrieval-sdg-flow.png)

## What it does

1. Deduplicates, clusters, and samples seed queries.
2. Plans conversation length, turn type, and research depth.
3. Simulates a user and a tool-using research assistant.
4. Routes retrieval calls to an HTTP service and simulates other tools with an LLM.
5. Compresses model context while retaining the full saved trajectory.
6. Validates, scores, and filters trajectories into SFT-ready JSONL.

The implementation is domain-agnostic. Corpus access, tools, models, endpoints, and
generation behavior are configured in `config/pipeline.yaml`.

Everything for one run lives under `experiments/<exp_name>/output/`, keyed by the
top-level `exp_name`. Reusing a name overwrites; change it to keep runs side by side.

## Requirements

- Python 3.10+
- Data Designer
- A JSONL query source
- OpenAI-compatible model endpoints
- An HTTP retrieval endpoint for real generation (the included deployment is
  documented in [`retriever/README.md`](retriever/README.md))
- API keys exposed through the environment variables named in the config

The checked-in retrieval address is deployment-specific; replace it before running
outside that network.

To start or rebuild the NeMo Retriever service currently used by the checked-in
configuration, follow [`retriever/README.md`](retriever/README.md). That deployment
serves an already-extracted JSONL corpus; it does not extract PDFs.

## Quick start

Run from this directory:

```bash
pip install -e .

# Optional: synthesize seed queries from the corpus (writes queries.jsonl)
python pipeline.py --config config/pipeline.yaml --stage query_gen

# Prepare seeds: normalize, deduplicate, embed, cluster, sample (writes seeds.jsonl)
python pipeline.py --config config/pipeline.yaml --stage query_prep

# Generate a small smoke-test batch (writes raw.jsonl)
python pipeline.py --config config/pipeline.yaml --stage generate --limit 5

# Validate and filter; --config resolves the judge and the experiment paths
python evaluate.py --config config/pipeline.yaml --judge
```

`--stage all` runs query preparation and generation. `--limit N` caps sampled or
generated rows. Paths are derived from `exp_name`, so the commands above read and write
`experiments/<exp_name>/output/`; pass `--input`/`--out` to `evaluate.py` to point at
explicit files instead.

## Configure

Update `config/pipeline.yaml` before generation:

| Section | Key settings |
|---|---|
| Top level | `exp_name`, `exp_root` (all outputs go to `experiments/<exp_name>/output/`) |
| `query_gen` | optional corpus-grounded seed synthesis: source, chunks/lancedb, `n_queries`, embedding |
| `query_prep` | dedup threshold, embedding model, clustering, target sample size |
| `persona` | DD-native Person sampler: enable, locale, synthetic personas |
| `providers` | provider name, endpoint, `api_key_env` (the env-var **name**, not the key) |
| `models` | four required aliases and inference parameters |
| `retrieval` | endpoint, tool names, `top_k`, oversampling, timeout, headers, field map |
| `tools` | OpenAI-format function schemas |
| `engine` | turns, hops, steps, context compression, judging, persona behavior |

The required model aliases are:

| Alias | Responsibility |
|---|---|
| `assistant_model` | Research, tool calls, and final answers |
| `user_model` | Openings, follow-ups, and clarification replies |
| `aux_response_model` | Responses for tools without real backends |
| `judge_model` | Optional query and trajectory checks |

Tool names listed in `retrieval.tools` are routed to the HTTP client. Every other tool
is simulated by `aux_response_model`. The default config provides `search`,
`memory_read`, and `memory_write`.

### Retrieval API mapping

`retrieval.field_map` adapts service-specific request and response schemas. The client:

- requests `top_k * oversample_factor` results;
- deterministically samples back to `top_k` per seed row;
- preserves result order after sampling;
- assigns a stable content-hash id when a chunk has no id.

`top_k` is controlled by config, not by model-generated arguments.

## Runtime flow

For each seed, `ConversationSimulatorGenerator` creates one shared message list. On each
user turn it:

1. builds a bounded assistant view and attaches tool schemas;
2. requests one or more research steps, forcing tool use until `min_hops` when enabled;
3. sanitizes and schema-validates each tool call;
4. executes retrieval calls or simulates non-retrieval tools;
5. appends tool results and updates the research scratchpad;
6. stops on a tool-free answer, repeated retrieval stalls, or `max_steps`;
7. generates the next user follow-up and repeats.

The saved trajectory retains full tool results and `reasoning_content`. Later model
calls receive a compacted view: the system prompt and current turn remain visible,
older tool results may become references or snippets, and prior reasoning is not
replayed.

Tool-calling uses direct OpenAI-compatible clients because some Data Designer facades
do not preserve tool-call names correctly.

## Evaluation

Evaluation is separate from generation, so raw trajectories can be re-scored without
regenerating them.

The deterministic gate requires:

- at least one tool call;
- no schema-invalid recorded tool calls;
- at least one retrieval-shaped result;
- a final tool-free assistant answer;
- no cited chunk id that was never retrieved (citation integrity — the hard grounding
  guarantee).

Because the objective gate requires a real retrieval call, degenerate zero-hop rows
cannot reach the SFT set — so generation leaves `force_first_tool: false` (an A/B run
showed the query, not the flag, drives first-turn tool use) and lets the gate enforce
the guarantee.

Evaluation also reports answer/evidence character n-gram overlap as a diagnostic. It is
a weak exact-substring proxy (paraphrased answers score low; the observed median is
~0.11), so it is **report-only by default**. If you gate on it, set `--min-overlap` low
(~0.02–0.03, to catch only true zero-grounding) and calibrate per corpus — a larger
value discards well-grounded rows.

With `--judge`, an LLM adds a defect gate plus a 1-5 quality score. Any configured
disqualifier rejects the row; clean rows must also meet `--min-quality`.

The SFT rows keep assistant `reasoning_content` by default: the chain-of-thought is a
training target — the whole point of the SDG — and the assistant model is a config knob,
so the trace is the transferable asset. Pass `--strip-reasoning` if your training recipe
trains only on the final answer and tool calls.

## Outputs

All under `experiments/<exp_name>/output/`:

| File | Contents |
|---|---|
| `queries.jsonl` | Synthesized seed queries (only when `query_gen` runs) |
| `seeds.jsonl` | Deduplicated, clustered, sampled seeds fed to generation |
| `raw.jsonl` | Generated trajectories before decoupled evaluation (keeps `reasoning_content`) |
| `sft.jsonl` | Trajectories that passed enabled gates (keeps `reasoning_content` by default) |
| `summary.json` | Keep rate and aggregate cluster, hop, grounding, and score metrics |

Per-row eval detail (objective + rubric) is not written to disk; it lives only in the
checkpointed DataDesigner artifacts and is aggregated into `summary.json`.

Raw rows contain:

```json
{
  "messages": [],
  "tools": [],
  "cluster_id": "c003",
  "hops_taken": 4,
  "conversation_status": true,
  "trajectory_judgment": "{}",
  "retrieval_log": "[]"
}
```

Some metadata values are serialized JSON strings because they pass through Data
Designer side-effect columns. Parse them before analysis.

## Code map

```text
retrieval_sdg/
├── config/pipeline.yaml       runtime configuration
├── pipeline.py                query preparation and generation driver
├── evaluate.py                deterministic and LLM evaluation
├── retriever/                 NeMo Retriever server, ingestion, and systemd units
├── retrieval_sdg/
│   ├── plugin.py              Data Designer registration
│   ├── query_prep/            embedding, deduplication, clustering, sampling
│   ├── conversation/          planning, generation, tools, context, judges, prompts
│   ├── retrieval/client.py    retrieval HTTP adapter
│   └── core/                  model clients and message helpers
└── tests/                     focused unit tests
```

## Tests

```bash
pytest -q tests
```

The focused suite covers deduplication, retrieval-client behavior, and tool-call schema
verification. Full generation requires reachable model and retrieval endpoints; run a
small `--limit` batch and evaluate it before starting a large job.
