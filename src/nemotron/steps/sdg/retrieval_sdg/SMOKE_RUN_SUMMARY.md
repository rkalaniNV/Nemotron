# retrieval_sdg — Smoke-Run Summary

A snapshot of the `retrieval_sdg` pipeline generating **multi-turn, retrieval-grounded,
tool-calling conversations** for SFT, end-to-end on a real corpus.

## Setup

| | |
|---|---|
| **Seed queries** | 100 natural-language questions — [`data/queries.jsonl`](data/queries.jsonl) (checked in) |
| **Knowledge base** | [Legal Dataset — SC Judgments of India, 1950–2024](https://www.kaggle.com/datasets/adarshsingh0903/legal-dataset-sc-judgments-india-19502024) (Kaggle), chunked + served through the retrieval endpoint |
| **Assistant / aux / judge** | `openai/gpt-oss-120b` (reasoning model) |
| **User simulator** | `google/gemma-4-31B-it` |
| **Retrieval** | HTTP endpoint over the chunked corpus; 2× oversample → random top-k, cross-hop dedup |
| **Domain coupling** | none in code — the legal domain enters *only* through the retrieved chunks and the seed queries |

The seeds are everyday questions a person would actually ask ("Back in the 1970s there was
a huge case about whether Parliament can amend any part of the Constitution — what did the
court finally decide?"). The pipeline is domain-agnostic; swap the corpus + queries and it
re-runs unchanged.

## Headline numbers

| Stage | Count | Rate |
|---|---:|---:|
| Seed queries | 100 | — |
| Trajectories generated (raw) | 95 | 95% *(5 lost to a transient retriever 502)* |
| Passed deterministic objective gate | 94 | 98.9% |
| **Kept for SFT** (objective + LLM defect gate) | **80** | **84.2%** |

### Grounding integrity (deterministic, exact)

| Check | Result |
|---|---|
| Total chunk-ids cited across all answers | **992** |
| **Fabricated citations** (cited but never retrieved) | **0** |
| Rows failing citation integrity | **0** |

Every citation the assistant emitted traces to a chunk it actually retrieved — verified by
code, not by the LLM.

## Conversation shape

### Turns per conversation (kept)

| Turns | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|
| Count | 1 | 3 | 9 | 23 | **30** | 14 |

Center-weighted around 4–5 turns, as configured (`min_turns 3`, `max_turns 6`).

### Research depth — hops (search rounds) per conversation

| Hops | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Count | 1 | 8 | 19 | 12 | 16 | 7 | 4 | 4 | 5 | 2 | 2 |

Median ≈ 6 hops, tail out to 20 — genuine multi-hop research, not single-shot lookups.

### Top conversation flow patterns

First letter of each message role — **S**ystem, **U**ser, **A**ssistant, **T**ool:

| Pattern | Count |
|---|---:|
| `SUATAUATAUATAUATAUATA` | 6 |
| `SUATATAUATAUATAUATA` | 5 |
| `SUATAUATAUATAUATA` | 4 |
| `SUATAUATATAUATAUATA` | 4 |
| `SUATAUATA` | 3 |

The recurring `…UATA…` motif = *user asks → assistant searches → tool returns → assistant
answers*, repeated per turn with extra `T`s where the assistant chained multiple searches.

## Length & size distributions

| Distribution | min | p25 | median | p75 | max | mean |
|---|---:|---:|---:|---:|---:|---:|
| **Context length** (tokens/trajectory) | 5,101 | 22,981 | 27,492 | 32,639 | 70,246 | 28,698 |
| **Messages** / trajectory | 5 | — | 23 | — | 51 | — |
| **Tool calls** / trajectory | 1 | — | 6 | — | 20 | — |
| **Final-answer length** (chars) | 7 | — | 4,291 | — | 7,346 | — |

Trajectories are substantial — a median of ~27.5k context tokens and ~23 messages per
conversation, with the deepest reaching 70k tokens. (A short 7-char "final answer" is an
honest *"the knowledge base doesn't cover that"* — a desirable behavior, not a failure.)

### Deterministic grounding overlap

Language/domain-agnostic proxy: fraction of each answer's character-12-grams that also
appear in the retrieved chunk text (weakest answer per trajectory).

| mean | median | p10 | min | max |
|---:|---:|---:|---:|---:|
| 0.133 | 0.129 | 0.092 | 0.072 | 0.234 |

Values are moderate by design — the assistant **paraphrases and synthesizes** the evidence
rather than copying it, so exact n-gram overlap is expected to be well below 1.0 even for
faithful answers. Citation integrity (above) is the exact grounding guarantee; this is a
supporting signal.

## Quality (LLM defect gate)

The LLM screens for **binary train-harmful defects** — not aesthetic 1–5 scores. A row is
rejected only if a defect clearly fires; citation-id validity is handled by code, so the
judge assesses prose only.

| Defect | Fire rate |
|---|---:|
| `unsupported_claims` (prose drift from evidence) | 14.9% |
| `no_real_research` | 1.1% |
| `request_unresolved` | 1.1% |
| `incoherent` | 0.0% |
| `user_out_of_character` | 0.0% |
| **Clean (zero defects)** | **85.1%** |

### Soft quality score (reporting only)

| Quality | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| Count | 1 | 2 | 8 | 5 | **78** |

**Mean quality: 4.67 / 5** — 82% of judged trajectories scored a perfect 5.

## Takeaways

- **84.2% keep-rate** on a real legal corpus with **zero fabricated citations** across 992 citations.
- Conversations are **deep and multi-turn** — median ~5 turns and ~6 hops, tails to 6 turns / 20 hops.
- The `incoherent` and `user_out_of_character` defects never fired — the assistant stays
  consistent and the simulated user stays in character throughout.
- The only meaningful reject reason is genuine **prose drift** (14.9%) — misattributed
  quotes or specifics not in the evidence — exactly what the gate should catch.

*Generated from `output/sdg/retrieval_sdg.summary.json` (95-trajectory run).*
