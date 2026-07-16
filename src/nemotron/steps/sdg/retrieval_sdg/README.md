# retrieval_sdg — multi-turn retrieval-conversation SDG

Generate synthetic **multi-turn, multi-step, tool-calling research conversations**
grounded in a retrieval backend — ready for SFT. Each output row is one full agent
trajectory (OpenAI `messages` + `tools`) where a user asks, an assistant researches
with tools, reads what it retrieves, searches again, and answers — with follow-up turns.

Two dependencies are **external and out of scope**; this package consumes them:

| External input | What it is | How we use it |
|---|---|---|
| **query source** | a JSONL list of queries (question-gen and/or customer search logs) | dedup / cluster / sample into a seed set |
| **retrieval service** | a POST API returning retrieved chunks for a query | wrap it thinly (oversample + randomize) |

```
queries.jsonl (query source)
  → dedup → cluster → sample                     (query_prep — offline, MiniLM only)
  → per query: planner samples turn shape → tool loop (retrieval service, 2× oversample+randomize)
               + follow-ups + context compression + inline judge   (conversation engine — Data Designer)
  → raw trajectories → evaluate.py (objective gate + LLM rubric) → final SFT jsonl
```

Everything domain-specific lives in `config/pipeline.yaml` (the corpus is the retrieval
service's; the tools, models, and personas are config). The Python never names a domain.

---

## Layout

```
retrieval_sdg/
├── config/pipeline.yaml        every knob (I/O, models, retrieval, tools, engine)
├── pipeline.py                 orchestrator: query_prep → conversation generation
├── evaluate.py                 decoupled judge/filter (raw → filtered SFT + .scored + .summary)
├── retrieval_sdg/              installable package (Data Designer plugin)
│   ├── plugin.py               DD Plugin() registration (entry point: retrieval-sdg)
│   ├── query_prep/             dedup · cluster · sample   ("query generator", offline)
│   │   └── embed.py            MiniLM embedding (dedup + clustering only)
│   ├── conversation/           the engine ("conversation generator")
│   │   ├── config.py           ConversationSimulatorConfig — every engine knob
│   │   ├── generator.py        the DD ColumnGenerator: one row → one trajectory
│   │   ├── planner.py          samples each conversation's turn/depth shape
│   │   ├── tools.py            tool env: retrieval → service, others → LLM-simulated
│   │   ├── context.py          view-only context compression (head/tail preserve)
│   │   ├── judges.py           inline judge (XML <explanation>/<rating>)
│   │   ├── verifiers.py        tool-call JSON-schema validation
│   │   └── prompts.py          ALL prompts, one file
│   ├── retrieval/client.py     thin retrieval-service HTTP wrapper (2× oversample + randomize)
│   └── core/                   llm (direct clients + majority vote + DD async bridge) ·
│                               messages · persona · caller
└── tests/                      unit tests (dedup · retrieval client · verifiers)
```

---

## Quick start

```bash
pip install -e .                              # registers the DD plugin entry point
export LEPTON_API_KEY=""                       # per-provider key env (open endpoints => empty is fine)

# Stage A — dedup/cluster/sample the incoming queries (offline; MiniLM only, no API)
python pipeline.py --config config/pipeline.yaml --stage query_prep

# Stage B — generate trajectories (needs retrieval.endpoint set to a live retrieval service)
python pipeline.py --config config/pipeline.yaml --stage generate --limit 5

# Judge & filter into the final SFT set (re-runnable without regenerating)
python evaluate.py --input output/sdg/retrieval_sdg.raw.jsonl \
  --out output/sdg/retrieval_sdg.jsonl --judge
```

`--stage all` runs A then B. `--limit N` caps sampled seeds / generated rows (smoke tests).

---

## How generation works

### The models (four aliases, all config-driven)
| Alias | Role |
|---|---|
| `assistant_model` | the research agent being trained — reasons, calls tools, answers |
| `user_model` | role-plays the human (opening request, follow-ups, clarification answers) |
| `aux_response_model` | simulates every non-retrieval tool's JSON response |
| `judge_model` | the query gate + the trajectory judge |

Each alias maps to a provider (`endpoint` + `api_key` env). **Tool-calling is routed
through direct OpenAI clients** (`core/llm.py`), bypassing DD's facade because it
mis-parses tool-call names on some endpoints. Self-consistency is available via
`majority_vote_n`.

### One shared message list, three views
There is a **single `messages` list per row** — it *is* the trajectory that gets
saved. Every agent reads a role-appropriate **projection** of it and appends back:

- **assistant** → a **compressed** view (`build_assistant_view`) + the tool schemas.
- **user** → the transcript **flattened to a bounded script** (`format_history_compact`,
  system-stripped, tool outputs snipped, capped) so it stays in character and never
  overflows a small context window.
- **judge** → a bounded render of the whole conversation.

### The loop (per user turn — `generator._tool_loop`)
1. Assistant is called with the compressed view. While `hops < min_hops` its
   `tool_choice` is forced to `required`, so it must keep researching (no nudge
   messages are injected — the model drives itself).
2. Tool calls are schema-verified (with a correction loop), then executed:
   the **retrieval** tool → the HTTP service; **other tools** → `aux_response_model`.
   Results are appended as `tool` messages; retrievals also distill a note into the
   running scratchpad.
3. Once `hops ≥ min_hops` and the assistant answers with no tool call, that's the
   grounded answer for the turn. If `max_steps` is hit, one final tool-free call closes it.
4. Follow-up turns: `user_model` produces the next question (shape from the planner),
   and the loop repeats.

### Retrieval (`retrieval/client.py`)
Each `search` call: POST the service for `top_k × oversample_factor` chunks, then
randomly keep `top_k`. **`top_k` is fixed by the config knob — the model cannot change
it** (it's not in the tool schema). A single search is deliberately lossy, pushing the
agent to take more hops. Request/response field names are `field_map`-configurable;
chunks with no id get a stable **content hash** so they remain citable. (No cross-hop
dedup — the same chunk may recur.)

### Tools (the four shipped in config)
- `search` → routed to the retrieval service (chunks).
- `web_search` → LLM-simulated; **gated to recency only** (latest/current info the KB
  can't have) — not a general fallback.
- `memory_read` / `memory_write` → LLM-simulated; recall/persist a user preference.

Swap this list for the customer's — any tool not named in `retrieval.tools` is
LLM-simulated automatically.

### Context compression (`context.py`) — view only
Compression shapes only what the model *reads*; the **saved trajectory keeps full
chunks and full reasoning**. The leading system prompt (**head**) and everything from
the last user turn onward (**tail**) are kept verbatim; only the **middle** is
compressed, and only once its estimated tokens exceed `compression_token_limit` (older
tool results become `[tool result <id>: snippet…]`, last `context_window_k` kept raw).
A global cap (`HARD_VIEW_TOKENS`) guarantees the view never exceeds the model window
even on a very deep turn. Token estimate ≈ `len(content) / 4` — it counts only what is
actually sent.

### Reasoning traces
`reasoning_content` is **saved on every assistant turn** (tool-call, answer, and
forced-final) for training — but it is **not replayed** to the model on later steps
(only `content` + `tool_calls` are sent). The distilled scratchpad carries findings forward.

### Judging (embedded + decoupled)
An **inline** query gate skips weak openings, and an inline trajectory judge sets
`conversation_status`. The same scoring is **also** a standalone stage (`evaluate.py`)
so you can re-judge without regenerating: an objective gate (every tool call schema-valid
+ ≥1 retrieval + ends on a grounded answer) plus an optional LLM rubric (faithfulness /
coherence / completeness / tool_use / user_realism, 1–5). Set `inline_judge: false` to
keep all raw and judge only in the decoupled stage.

---

## Output schema

**Raw** (`output/sdg/*.raw.jsonl`) — one row per trajectory:
`{messages, tools, cluster_id, hops_taken, conversation_status, trajectory_judgment, retrieval_log}`.
`evaluate.py` writes the **filtered** SFT set (`--out`), a `.scored.jsonl` (every row +
eval), and a `.summary.json` (keep rate, per-cluster/hop distribution, mean rubric).

---

## Configuration (`config/pipeline.yaml`)

Key groups: `input_path`/`seeds_path`/`output_path` (paths are relative to the config
file) · `query_prep` (dedup threshold, clustering algo/k, sample `n_target`) ·
`providers` + `models` (the four aliases) · `retrieval` (endpoint, `field_map`, `top_k`,
`oversample_factor`) · `tools` (the user-defined list; must include the retrieval tool
named in `retrieval.tools`) · `engine` (`min_hops`, `max_steps`, `max_turns`,
`force_first_tool`, `context_window_k`, `compression_token_limit`, `gate_query`,
`inline_judge`, `majority_vote_n`, `conversation_plan`).

**To retarget:** point `input_path` at your queries, set `retrieval.endpoint` +
`field_map` at your service, swap the `tools` list, pick your `models`/`providers`, and
tune the engine knobs. Nothing in the Python changes.

---

## Status
Runs end-to-end against live retrieval + model endpoints (Gemma/Qwen validated). Unit
tests cover query dedup, the retrieval client (oversample / randomize / content-hash id),
and tool-call verification. Full generation requires a live retrieval-service endpoint
and model API access.
