# Container-Backed Benchmarks

## Source Of Truth

Use the selected evaluation container as the source of truth for benchmark
metadata. Evaluator-compatible containers publish their task catalog through
Evaluator metadata, including benchmark defaults, supported endpoint types, and
command wiring.

Configure container-backed tasks with task-level `container` and
`endpoint_type` so the launcher can resolve the task definition from the
selected image. This applies to custom language benchmarks, sovereign
benchmarks, standard English, tool, agent, or any other Evaluator-compatible
container.

## Benchmark Choice

Choose benchmarks based on the user's target language and capability, not on
where the model is hosted or which market it serves. For example, a sovereign
model served behind an OpenAI-compatible chat endpoint can run standard English
benchmarks if the user wants English knowledge, instruction-following, or
reasoning measurement.

Common standard English starting points:

- `lm-evaluation-harness.mmlu_instruct` or `lm-evaluation-harness.mmlu`
- `lm-evaluation-harness.ifeval`
- `simple_evals.gpqa_diamond`

Use the container metadata or Evaluator task listing for the exact task names
available in the selected image.

## Custom Language Containers

For user-developed containers in another language or domain, treat the
container as authoritative:

- Use its published task names and harness prefix.
- Use its language codes, dataset names, and metrics as defined by the image.
- Ask whether the endpoint is chat, completions, or logprob-capable.
- Start with a one-sample smoke run before recommending a full benchmark set.

The sovereign/Indic sections below are examples of how a language-specific
container can organize tasks; they are illustrative, not prescriptive.

## Sovereign/Indic Example Language Codes

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

## Sovereign/Indic Example Task Families

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

## Sovereign/Indic Example Starter Sets

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

Set the task-level `container` to the evaluation image that owns the selected
benchmark. For the built-in sovereign config, set `run.sovereign.container` or
`SOVEREIGN_EVAL_CONTAINER` to the sovereign evaluation image. Prefer a fully
qualified registry image so Evaluator Launcher can read metadata directly.
Local images can be useful for smoke testing, but may require an explicit trust
flag if the launcher cannot discover their metadata.
