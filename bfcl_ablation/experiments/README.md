# Experiment reports

One document per rung of the ablation ladder. Each opens with its insights — what the numbers
mean and what they change — and only then gives the full figures, the limitations, and where
the artifacts live.

| arm | question | status |
| --- | --- | --- |
| [a0.md](a0.md) | How much friction exists, and what does the benchmark look like? | measured |
| [a1.md](a1.md) | How much comes off with no model, provably? | measured |
| [a2.md](a2.md) | How much diversity can an LLM add while conclusions hold? | measured |
| [a3.md](a3.md) | Can an LLM propose the tasks themselves? | measured |
| [a4.md](a4.md) | Are the assertions actually checking anything? | measured |
| [a5.md](a5.md) | Does rewording the request change the benchmark's verdict on a model? | measured |
| [a6.md](a6.md) | Is the oracle itself falsifiable — does anything check the backend? | measured |
| [a2_rerun.md](a2_rerun.md) | Does A2 reproduce — and what does a passing run hide? | control |

[findings.md](findings.md) collects every insight across the arms in one place.

`a2_rerun` is a **control**, not a rung: it opens no degree of freedom. It is filed here because
the inference it rules out — "the numbers reproduce, therefore the numbers are right" — is the
most natural wrong reading of this study.

Metric definitions are fixed and versioned in [`../results/METRICS.md`](../results/METRICS.md).
Every `metrics.json` records the `metrics_version` it was computed under; arms recorded under
different versions are not comparable.

Methodology, layout and how to run each arm: [../README.md](../README.md).

Every arm runs the unmodified production pipeline. An arm is defined by the pack and config
it feeds in, never by a patch to `runtime/benchmark_families/bfcl` — a patched generator would
measure the patch instead of the pack.

A0 and A1 use no model. A2, A3 and A4 call a local vLLM server
(`openai/gpt-oss-120b` at `http://127.0.0.1:8000/v1`); every call is disk-cached under
`_generated/llm_cache`, so a re-run reproduces the same benchmark and the cache doubles as the
record of what the model was asked and what it answered.

A5 is the only arm whose subject is a *model* rather than the pack. It calls the same server
through `/v1/responses` (tool calling is not parsed on `/chat/completions` unless the server is
started with `--enable-auto-tool-choice --tool-call-parser openai`) and caches under
`_generated/target_cache`.
