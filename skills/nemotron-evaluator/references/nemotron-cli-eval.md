# Nemotron Eval Step

## Configs

Use these config variants from `src/nemotron/steps/eval/model_eval/config/`:

| Config | Purpose |
|---|---|
| `hosted_mmlu_pro` | Hosted OpenAI-compatible chat endpoint smoke for standard MMLU-Pro validation through the direct NeMo Evaluator API. |
| `hosted_mmlu` | Hosted OpenAI-compatible chat endpoint smoke for MMLU-style setup validation. |
| `sovereign` | Hosted endpoint plus sovereign benchmark task container. |

`hosted_mmlu_pro` uses the direct runner and reads endpoint settings from
`deployment.*`. The Launcher-backed configs use `deployment: none`, so
`target.api_endpoint.url` and `target.api_endpoint.model_id` are required.
`api_key_name` is always the environment variable name, not the raw key.

## Endpoint Overrides

```bash
export NVIDIA_API_KEY=...
export NEMO_EVALUATOR_MODEL_ID=nvcf/openai/gpt-oss-120b
export NEMO_EVALUATOR_MODEL_URL=https://inference-api.nvidia.com/v1/chat/completions

python src/nemotron/steps/eval/model_eval/step.py \
  --config src/nemotron/steps/eval/model_eval/config/hosted_mmlu_pro.yaml

python src/nemotron/steps/eval/model_eval/step.py \
  --config src/nemotron/steps/eval/model_eval/config/hosted_mmlu.yaml
```

If the key is exported under a different name, override the API key name.
Launcher will pass that host env var into the evaluator container:

```bash
python src/nemotron/steps/eval/model_eval/step.py \
  --config src/nemotron/steps/eval/model_eval/config/hosted_mmlu.yaml \
  target.api_endpoint.api_key_name=API_KEY
```

Before running an expensive benchmark, verify that the key can access the exact
model id. Hosted services may expose `nvcf/openai/gpt-oss-120b` or another
provider-prefixed id even when a model is discussed by a shorter name:

```bash
curl -sS \
  -H "Authorization: Bearer $NVIDIA_API_KEY" \
  https://inference-api.nvidia.com/v1/models
```

Use the returned `id` exactly in `NEMO_EVALUATOR_MODEL_ID` or the config
override. A `401` with `key_model_access_denied` means the key is valid but not
authorized for that model.

## Direct Runner Environment

The direct runner invokes framework console scripts such as `simple_evals`.
Install the repo's eval extra in the Python environment used to launch the step:

```bash
uv sync --extra eval
```

The step prepends the active Python interpreter's `bin` directory to `PATH`, so
`.venv/bin/python src/nemotron/steps/eval/model_eval/step.py ...` can find
console scripts installed in `.venv/bin`.

## Reasoning Models

Reasoning models often need higher output budgets than older chat models.
For GPT-5.1-style endpoints, start smoke tests at `max_new_tokens: 1024`.
For complex reasoning tasks, use much larger limits and the model-card
recommended temperature/top-p. If the endpoint returns a max-token finish
error, increase `evaluation.nemo_evaluator_config.config.params.max_new_tokens`
or set a task-level override.

## Dry Run Caveat

Set `dry_run=true` on the eval step config to make Launcher prepare and print
the resolved evaluation config. For sovereign tasks, validate the sovereign image
separately:

```bash
nemo-evaluator-launcher ls tasks --from <sovereign-image>
```

## Results

For local runs, Launcher writes timestamped run directories below
`execution.output_dir`. Each task typically has:

- `artifacts/results.yml`
- `artifacts/eval_factory_metrics.json`
- task logs near the invocation directory

Use the bundled script:

```bash
python skills/nemotron-evaluator/scripts/summarize_results.py <results-dir>
```

Direct NeMo Evaluator runs usually write `results.yml`,
`eval_factory_metrics.json`, and task JSON/HTML files below
`<output_dir>/<benchmark>/`. Smoke runs with `limit_samples` are only setup
validations.
