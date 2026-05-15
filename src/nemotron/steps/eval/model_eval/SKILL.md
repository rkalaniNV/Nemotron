---
name: nemotron-eval-model-eval
description: Configure and run the Nemotron eval/model_eval step with NeMo Evaluator for hosted OpenAI-compatible endpoints, standard English benchmarks such as MMLU-Pro, launcher-backed task containers, and result summaries.
---

# Model Evaluation

Use `eval/model_eval` when a model is already available through an
OpenAI-compatible endpoint or when an Evaluator task container should be wired
through Launcher. Do not edit Nano3 or Super3 recipes for this workflow.

## Choose A Runner

- Use `runner: direct` for the fastest hosted endpoint smoke test with standard
  NeMo Evaluator tasks such as `mmlu_pro`, `mmlu`, `hellaswag`, or
  `arc_challenge`.
- Use `runner: launcher` when the benchmark needs a task container, a custom
  framework, or Launcher execution controls.

## Hosted MMLU-Pro Smoke

Start from `config/hosted_mmlu_pro.yaml` for a standard English setup
validation:

```bash
export NVIDIA_API_KEY=<key>
export NEMO_EVALUATOR_API_KEY_NAME=NVIDIA_API_KEY
export NEMO_EVALUATOR_MODEL_URL=https://inference-api.nvidia.com/v1/chat/completions
export NEMO_EVALUATOR_MODEL_ID=nvcf/openai/gpt-oss-120b

python src/nemotron/steps/eval/model_eval/step.py \
  --config src/nemotron/steps/eval/model_eval/config/hosted_mmlu_pro.yaml
```

The default config uses `limit_samples: 2`. Treat the result as a workflow
smoke test, not a model-comparison score.

## Endpoint Discipline

- Put the raw key only in an environment variable. Configs should use
  `deployment.api_key_name` or `target.api_endpoint.api_key_name`.
- Verify hosted model IDs with the endpoint's models API when possible. Some
  services expose provider-prefixed IDs such as `nvcf/openai/gpt-oss-120b`
  rather than the marketing name or a shorter alias.
- Match `endpoint_type` to the task. Chat/instruction tasks use `chat`;
  log-probability tasks need a completions endpoint and tokenizer settings.

## Local Environment

Install the evaluator framework package used by the task. For MMLU-Pro through
`simple_evals`, the runtime needs `nemo-evaluator` and `nvidia-simple-evals`.
Install them in this repo with `uv sync --extra eval`. The step prepends the
active Python interpreter's `bin` directory to `PATH` so subprocesses can find
console scripts such as `simple_evals`.

## Results

Direct runs usually write `results.yml`, `eval_factory_metrics.json`, and task
JSON/HTML files below `<output_dir>/<benchmark>/`. Launcher runs may place
`artifacts/results.yml` under task-specific invocation directories. The step
prints numeric score-like fields when it finds either shape.
