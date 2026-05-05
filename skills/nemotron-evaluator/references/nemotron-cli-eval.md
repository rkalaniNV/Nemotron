# Nemotron Eval Step

## Configs

Use these config variants from `src/nemotron/steps/eval/model_eval/config/`:

| Config | Purpose |
|---|---|
| `hosted_mmlu` | Hosted OpenAI-compatible chat endpoint smoke for MMLU-style setup validation. |
| `sovereign` | Hosted endpoint plus sovereign benchmark task container. |

Both variants use `deployment: none`, so `target.api_endpoint.url` and
`target.api_endpoint.model_id` are required. `target.api_endpoint.api_key_name`
is the environment variable name inside the evaluator container.

## Endpoint Overrides

```bash
export NVIDIA_API_KEY=...
export NEMO_EVALUATOR_MODEL_ID=openai/openai/gpt-5.1
export NEMO_EVALUATOR_MODEL_URL=https://inference-api.nvidia.com/v1/chat/completions

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
