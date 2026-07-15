# 🔎 Agentic RAG — Deep-Research Trajectory SDG

Generate synthetic training data that teaches a model to **research like an agent**:
read a question, search a knowledge base, reason over what it finds, search again
if needed, and answer — with every claim grounded in real retrieved text.

```
question → (clarify?) → plan → search → reason → search → … → grounded answer
```

The engine is **domain-agnostic**. It ships with the **Constitution of India** as a
worked example; point it at any documents + tools and it generates a new domain.

---

## What you get

Each output row is one full **agent trajectory** in OpenAI `messages` + `tools`
format — the user's question, the assistant's reasoning, its tool calls, the real
retrieved passages, and the final answer — ready for SFT. Every row also carries
metadata (how many hops, whether the gold passage was retrieved, etc.).

---

## Quick start

> Run everything from this folder: `.../steps/sdg/agentic_rag`

### 1 · Install

```bash
pip install -e .
export NVIDIA_API_KEY=nvapi-...        # from build.nvidia.com
```

Check the key can call the model (should print `200`):

```bash
curl -s -o /dev/null -w "%{http_code}\n" https://inference-api.nvidia.com/v1/chat/completions \
  -H "Authorization: Bearer $NVIDIA_API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"nvidia/openai/gpt-oss-120b","messages":[{"role":"user","content":"hi"}],"max_tokens":5}'
```

### 2 · Prepare the corpus (one time, no API key needed)

```bash
cd data_prep

# chunk the document                → data/constitution_chunks.jsonl
python chunk_document.py --input ../data/constitution_of_india.txt \
  --output ../data/constitution_chunks.jsonl --profile indian_statute

# build the embedding index (MiniLM) → data/index/
python retriever.py build --chunks ../data/constitution_chunks.jsonl \
  --index ../data/index --backend embedding

# build multi-hop question seeds     → data/bundles.jsonl
python bundle_builder.py --chunks ../data/constitution_chunks.jsonl \
  --output ../data/bundles.jsonl --mode entity_link --size 3 --num 200

cd ..
```

Quick retrieval check (should put **Article 32** at the top):

```bash
python data_prep/retriever.py query --index data/index --backend embedding \
  --q "which court to approach when a fundamental right is violated" --gold 32 226
```

### 3 · Generate trajectories

```bash
python step.py --config config/tiny.yaml --mode preview    # small smoke test
python step.py --config config/indian_legal.yaml --mode create   # full run
```

Results land in `output/sdg/`. Each run writes two files:
- `*.jsonl` — the training data (successful trajectories only)
- `*.raw.jsonl` — every generated row, for inspection/debugging

---

## How it works

Two stages: an **offline** prep step, then the **generation** step.

```
OFFLINE                          GENERATION (one row at a time)
─────────                        ──────────────────────────────
document                         pick a question seed + persona + style
  │ chunk                          │
chunks                           user asks a question
  │ embed                          │
MiniLM index                     assistant plans, then researches:
  │ link                            search → read → "enough yet?" → search again
multi-hop seeds  ───────────▶      │
                                 assistant writes a grounded answer
                                   │
                                 judge → keep / trim / drop
```

**Six ideas make the data good:**

1. **Multi-hop by design** — each question is built from a *set* of linked
   passages, so answering it genuinely requires combining sources.
2. **Real retrieval** — search tools return real passages from the corpus (via
   MiniLM embeddings), so citations are never made up.
3. **Reasoned depth** — the assistant keeps searching until a "do I have enough?"
   check passes, not for a fixed number of steps.
4. **Clarify first** — for vague questions it asks a clarifying question before
   researching, then works on its own.
5. **Stays coherent when long** — a sliding window plus a running notes
   "scratchpad" keep each step focused even across many hops.
6. **Built-in variety** — persona, question type, difficulty, and outcome are
   sampled per row, so the dataset isn't monotonous.

---

## Files

```
agentic_rag/
├── data_prep/            ← offline corpus tools (run without an API key)
│   ├── chunk_document.py     split a document into clean chunks
│   ├── retriever.py          MiniLM search index + retrieval checks
│   └── bundle_builder.py     group linked chunks into multi-hop seeds
├── agentic_rag/          ← the generation plugin
│   ├── config.py             every setting lives here (one source of truth)
│   ├── generator.py          the research loop
│   ├── tools.py              real search vs. simulated tools
│   ├── context.py            sliding window + scratchpad
│   ├── prompts.py            the agent/user/judge prompts
│   └── …                     llm, judges, messages, persona, verifiers
├── config/
│   ├── indian_legal.yaml     the full config (domain + all settings)
│   └── tiny.yaml             tiny smoke-test config
├── step.py               ← the runner
└── data/                 ← source doc + (generated) chunks/index/seeds
```

---

## Making it your own

Everything domain-specific lives in `config/indian_legal.yaml` — the corpus, the
tools, the personas, and every tuning knob (how deep to research, how many
passages to retrieve, how strict the judges are, how much variety to sample).
The Python code never mentions law or the Constitution. **To do a new domain:**
swap the documents and tool definitions in the YAML and rerun — nothing else changes.

---

## Status

The offline pipeline (chunk → index → seeds) is built and validated on the real
Constitution, and retrieval grounds correctly. The generator runs end-to-end
against the live model endpoint and produces trajectories. Ongoing tuning:
improving how many trajectories pass the quality judges.
