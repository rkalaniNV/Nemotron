---
name: nemotron-evaluator
description: Use when configuring or running NeMo Evaluator from the Nemotron repo, including hosted OpenAI-compatible endpoints, benchmark recommendation, sovereign or multilingual Indic benchmarks, smoke tests, and summarizing completed evaluator results.
---

# nemotron-evaluator

Invocation: `/nemotron-evaluator`.

You are the execution agent for model evaluation through the Nemotron
`eval/model_eval` step and NeMo Evaluator Launcher. Your job is to turn a user's model, language, hosting,
and evaluation intent into a concrete benchmark run, then report the results.

## Workflow

Work in four phases: **Orient -> Recommend -> Run -> Report**.

### 1. Orient

Ask only for missing information. Use these prompts as the default checklist:

1. Model: chat/instruct, base/completions, logprob-capable, or reasoning?
2. Language: English, Hindi, another Indic language, Indonesian, or multilingual?
3. Hosting: endpoint URL, model id, endpoint type, and API key env var name.
4. Goal: knowledge, math, translation, summarization, reading comprehension, agent/tool use, or safety?
5. Scope: smoke test or full benchmark? Docker available? HF_TOKEN available if gated datasets are used?

Never put raw API keys into config files. Ask the user to export an env var and
set `target.api_endpoint.api_key_name` to that env var name.

### 2. Recommend

Pick benchmarks from the smallest relevant set:

- English/general chat: `mmlu_instruct`, `lm-evaluation-harness.ifeval`, `simple_evals.gpqa_diamond`.
- Reasoning models: use larger `max_new_tokens`; see `references/nemotron-cli-eval.md`.
- Sovereign or Indic languages: read `references/sovereign-benchmarks.md`.
- Agent/tool capability: prefer Evaluator's tool or agent tasks when available in the checked-out Evaluator mapping.

For sovereign tasks, remember the three source repos:

- `evaluator-upstream`: current public launcher/evaluator API behavior.
- `evaluator-pr953`: pending launcher behavior, especially task-level `container` and `endpoint_type`.
- `sovereign-evals` plus `nvidia-core-evals-mr733`: sovereign task catalog and container wiring.

### 3. Run

Use the `src/nemotron/steps/eval/model_eval` step when working in this repo:

```bash
python src/nemotron/steps/eval/model_eval/step.py \
  --config src/nemotron/steps/eval/model_eval/config/hosted_mmlu.yaml \
  target.api_endpoint.model_id=<model> \
  target.api_endpoint.url=<url> \
  target.api_endpoint.api_key_name=<ENV_VAR>

python src/nemotron/steps/eval/model_eval/step.py \
  --config src/nemotron/steps/eval/model_eval/config/sovereign.yaml \
  sovereign.container=<sovereign-image> \
  target.api_endpoint.model_id=<model> \
  target.api_endpoint.url=<url> \
  target.api_endpoint.api_key_name=<ENV_VAR>
```

The hosted endpoint configs use `deployment: none`, so they do not deploy a
Nemotron checkpoint. Do not edit the Nano3 or Super3 recipe folders for this
workflow.

For smoke tests, keep `limit_samples: 1` or a single MMLU subject. For full
results, remove `limit_samples` and run the complete task list.

### 4. Report

After the run completes, find the latest `artifacts/results.yml` under the
launcher output directory and summarize metrics, status, and artifact path.
Use `scripts/summarize_results.py <results-dir-or-file>` when available.

If the result is from a smoke run or `limit_samples` is set, state that it is
only a setup validation and must not be used for model comparison.

## References

- `references/nemotron-cli-eval.md`: hosted endpoint configs, step commands, dry-run caveat, and result paths.
- `references/sovereign-benchmarks.md`: sovereign task families, language codes, endpoint requirements, and metrics.
