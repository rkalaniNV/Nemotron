# Sovereign Benchmarks

## Source Of Truth

- Benchmark definitions: `sovereign-evals/core_evals/sovereign/framework.yml`
- Container wiring: `nvidia-core-evals-mr733/frameworks/llm/sovereign/Dockerfile`
- Launcher contract: `evaluator-pr953/packages/nemo-evaluator-launcher`

PR 953 supports task-level `container` and `endpoint_type`. Use that for
sovereign tasks until the task IRs are packaged in the public Evaluator build.

## Language Codes

Indic BYOB tasks use:

| Code | Language |
|---|---|
| `hi` | Hindi |
| `bn` | Bengali |
| `gu` | Gujarati |
| `kn` | Kannada |
| `mr` | Marathi |
| `ml` | Malayalam |
| `or` | Odia |
| `pa` | Punjabi |
| `ta` | Tamil |
| `te` | Telugu |

Indonesian uses `indommlu`. MILU uses language names:
`Hindi`, `Bengali`, `Tamil`, `Telugu`, `Marathi`, `Gujarati`, `Kannada`,
`Malayalam`, `Punjabi`, `Odia`, `English`.

## Task Families

| Goal | Chat task pattern | Completions/logprob pattern | Metric |
|---|---|---|---|
| Math | `gsm8k_hi`, `gsm8k_indic_<lang>` | `*_completions` | `pass@1.correct` |
| Knowledge MCQ | `mmlu_indic_<lang>`, `indicmmlu_pro_<lang>` | `*_completions`, `*_logprob` | `pass@1.correct` or `acc` |
| Translation | `indicgenbench_flores_in_<lang>` | `*_completions` | `chrf` |
| Summarization | `indicgenbench_crosssum_in_<lang>` | `*_completions` | `rouge_l` |
| Science MCQ | `arc_challenge_indic_<lang>` | `*_logprob` | `correct` or `acc_norm` |
| Reading comprehension | `boolq_indic_<lang>` | `*_logprob` | `correct` or `acc_norm` |
| Indonesian knowledge | `indommlu` | `indommlu_completions`, `indommlu_logprob` | `correct` or `acc` |
| MILU | none | `milu_<Language>` | `lm_eval acc,none` |

Use chat tasks for instruction/chat endpoints. Use `_completions` for base
models exposed at `/v1/completions`. Use `_logprob` only when the endpoint
supports completion logprobs.

## Recommended Starter Sets

For a Hindi chat model:

- `sovereign.gsm8k_indic_hi`
- `sovereign.mmlu_indic_hi`
- `sovereign.indicgenbench_flores_in_hi`
- `sovereign.indicgenbench_crosssum_in_hi`
- `sovereign.boolq_indic_hi`

For another Indic chat model, replace `hi` with the selected language code.
For a completions-only model, use the `_completions` variants. For logprob
benchmarks, verify logprob support before selecting `_logprob` tasks.

## Container

Set `run.sovereign.container` or `SOVEREIGN_EVAL_CONTAINER` to the sovereign
evaluation image. If working locally from the checked-out repos, build or obtain
the image from `nvidia-core-evals-mr733/frameworks/llm/sovereign`.
