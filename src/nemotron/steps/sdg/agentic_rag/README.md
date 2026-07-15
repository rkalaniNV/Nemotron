# 🔎 Agentic RAG — Deep-Research Trajectory SDG

Generate synthetic **multi-hop, tool-calling RAG trajectories** for aligning
models to agentic deep research over a real document corpus:

```
query → [clarify?] → research plan → (search → reason → search)×N → grounded answer → follow-up
```

The engine is domain-agnostic; the example ships with the **Constitution of India**.
Point it at any corpus + tools via config and it runs a new domain.

---

## 🚀 Quick start

Run from this directory (`.../steps/sdg/agentic_rag`).

### 0. Install

```bash
pip install -e .                 # installs the plugin + deps (data-designer, sentence-transformers)
export NVIDIA_API_KEY=nvapi-...  # from build.nvidia.com  (needs INFERENCE access)
```

Verify the key can actually call a model (must print `200`):

```bash
curl -s -o /dev/null -w "%{http_code}\n" https://integrate.api.nvidia.com/v1/chat/completions \
  -H "Authorization: Bearer $NVIDIA_API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"openai/gpt-oss-120b","messages":[{"role":"user","content":"hi"}],"max_tokens":5}'
```

### 1. Build the corpus (offline, one time)

```bash
cd data_prep

# a) chunk the document  →  data/constitution_chunks.jsonl   (~559 chunks / 481 articles)
python chunk_document.py \
  --input ../data/constitution_of_india.txt \
  --output ../data/constitution_chunks.jsonl --profile indian_statute

# b) build the MiniLM embedding index  →  data/index/         (all-MiniLM-L6-v2)
python retriever.py build \
  --chunks ../data/constitution_chunks.jsonl --index ../data/index --backend embedding

# c) build multi-hop evidence-set seeds  →  data/bundles.jsonl  (~156 bundles)
python bundle_builder.py \
  --chunks ../data/constitution_chunks.jsonl --output ../data/bundles.jsonl \
  --mode entity_link --size 3 --num 200

cd ..
```

Sanity-check retrieval (should surface the right article by meaning):

```bash
python data_prep/retriever.py query --index data/index --backend embedding \
  --q "which court to approach when a fundamental right is violated" --k 3 --gold 32 226
# expect Article 32 near the top, gold_rank = 1
```

### 2. Generate trajectories (NDD)

```bash
python step.py --config config/tiny.yaml     --mode preview   # fast smoke test: 2 records
python step.py --config config/indian_legal.yaml --mode create # full run: num_records in the YAML
```

Output → `output/sdg/agentic_rag_*.jsonl`: one OpenAI-format `messages`+`tools`
trajectory per row, plus metadata (`gold_rank_log`, `hops_taken`, `salvaged`,
sampler labels).

> **403 on the curl?** Your key lacks inference access — generate a fresh one at
> build.nvidia.com. The rest of the pipeline (chunk/retrieve/bundle) runs without a key.

---

## 🧩 How it works

### Core ideas
1. **Document-first, multi-hop by construction** — queries come from an *evidence
   set* of linked chunks, so answering needs synthesis across sources.
2. **Grounded retrieval** — RAG tools hit a real MiniLM retriever; only auxiliary
   tools are LLM-simulated. No fabricated citations.
3. **Depth is reasoned** — a sufficiency/gap check drives further retrieval to a
   `min_hops` floor (set per row by `depth_target`), capped at `max_steps`.
4. **Phased interaction** — DISCUSSION (clarify) → RESEARCH PLAN → autonomous
   TOOL LOOP → ANSWER. Strict phases keep ASK vs ANSWER unambiguous.
5. **Long context is assembled, never one-shot** — sliding window + a running
   findings scratchpad bound what each turn reads.
6. **Diversity is a configured cross-product** — persona × archetype × outcome ×
   ambiguity × depth samplers steer each row (not temperature).

### Four extensions over the base reference
| # | Extension | Where |
|---|-----------|-------|
| 1 | Real retriever for RAG tools + gold-rank + guided injection | `agentic_rag/tools.py` |
| 2 | Sufficiency/gap loop drives depth; `min_hops` floor | `generator._insufficient` |
| 3 | Sliding window + scratchpad | `agentic_rag/context.py` |
| 4 | Salvage-on-failure (truncate to last-good hop) | `generator._finalize` |

### Flow
```
OFFLINE (data_prep/):   document → chunks → MiniLM index → multi-hop bundles
NDD (step.py):          bundle seed + samplers
                          → user query (shaped by archetype/outcome/ambiguity)
                          → gate judge → DISCUSSION → RESEARCH PLAN
                          → TOOL LOOP: think → search(real retriever) → sufficiency → loop/answer
                          → follow-up → trajectory judge → keep/salvage/drop
                          → OpenAI messages+tools trajectory
```

---

## 🗂 Layout

```
agentic_rag/
├── data_prep/              # corpus side (standalone; no API key needed)
│   ├── chunk_document.py   #   generic profile-driven chunker
│   ├── retriever.py        #   MiniLM (+lexical fallback) + gold-rank
│   └── bundle_builder.py   #   multi-hop evidence-set builder
├── agentic_rag/            # NDD plugin (the simulation loop)
│   ├── config.py           #   DeepResearchSimulatorConfig — every knob (single source of truth)
│   ├── context.py prompts.py tools.py generator.py
│   ├── llm.py judges.py messages.py persona.py verifiers.py   # reused core
│   └── plugin_entry.py     #   DD plugin registration
├── config/
│   ├── indian_legal.yaml   # OUTER config — ALL domain specifics + knobs
│   └── tiny.yaml           # 2-record smoke test (gpt-oss-120b)
├── step.py                 # runner: YAML → Data Designer ConfigBuilder
├── pyproject.toml          # installs plugin via `data_designer.plugins` entry point
└── data/                   # seed_queries.jsonl tracked; corpus artifacts are generated (see .gitignore)
```

## ⚙️ Configuration

All behaviour is config-driven — nothing domain-specific is hard-coded in the
plugin. Tune knobs in `config/indian_legal.yaml` (depth, retrieval, context,
phases, quality gates) and the diversity samplers (persona locale, archetypes,
outcomes, ambiguity, depth). To retarget a new domain: swap the corpus + tool
schemas + persona locale in the YAML; the engine is unchanged.

## ✅ Status

Corpus side and all wiring are built and validated on the real Constitution;
the MiniLM retriever grounds correctly (gold articles surface by meaning). The
plugin loads against the real Data Designer runtime and reaches the model call.
A live end-to-end run needs an `NVIDIA_API_KEY` with inference access.
