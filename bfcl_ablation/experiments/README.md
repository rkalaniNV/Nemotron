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

[findings.md](findings.md) collects every insight across the five arms in one place.

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
