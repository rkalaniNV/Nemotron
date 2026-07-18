# Long-context SDG samples

This directory contains five complete, accepted trajectories selected from a
100-query generation run. They are intended as inspectable examples of the
pipeline's canonical record format, including the episode plan, messages, tool
definitions and transcripts, compaction events, validation result, and judge
scores.

## Run statistics

| Outcome | Count | Share |
| --- | ---: | ---: |
| Accepted and exported | 46 | 46% |
| Rejected | 18 | 18% |
| Quarantined | 11 | 11% |
| Generation failed | 25 | 25% |
| Total checkpointed and evaluated | 100 | 100% |

Only accepted records were selected for this directory. Every included sample
has `status: "accepted"`, `validation.ok: true`, and a `success` judge rating.

## Included samples

| Sample | Query ID | Turns | First-turn intent | Retrieval depth | Retrieval calls | Compactions | Minimum judge score |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| [Sample 1](sample_01_rag-0014.json) | `rag-0014` | 7 | `scope` | 1 | 2 | 0 | 4 |
| [Sample 2](sample_02_rag-0020.json) | `rag-0020` | 18 | `user_context` | 2 | 9 | 4 | 4 |
| [Sample 3](sample_03_rag-0021.json) | `rag-0021` | 21 | `misconception_check` | 1 | 8 | 2 | 5 |
| [Sample 4](sample_04_rag-0076.json) | `rag-0076` | 26 | `research` | 1 | 9 | 3 | 5 |
| [Sample 5](sample_05_rag-0081.json) | `rag-0081` | 38 | `example_first` | 1 | 7 | 3 | 5 |

The set is deliberately varied rather than being the first five output rows. It
spans 7–38 turns, five distinct first-turn intents, retrieval depths 1–2, 2–9
retrieval calls, and 0–4 context-compaction events.

## Reading a sample

Each JSON file is a single pretty-printed canonical trajectory. Useful fields
include:

- `episode_plan`: planned intent and retrieval behavior for each turn.
- `messages`: the complete multi-turn conversation and tool messages.
- `retrieval_transcript`: executed retrieval calls and returned chunks.
- `compaction_events` and `memory_events`: long-context management history.
- `validation`: structural and policy validation results.
- `judgment`: per-dimension scores, overall rating, and explanation.
- `metadata`: query, turn budget, retrieval depth, and aggregate counts.

These files are examples, not inputs required by the pipeline or fixtures used
by its test suite.
