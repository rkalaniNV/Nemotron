# Build Your Own Benchmark

This example shows the current Nemotron BYOB flow for creating a domain-specific MCQ benchmark from finance-related text.

The notebook is intentionally aligned with the agentic step structure:

- The reusable step lives in `src/nemotron/steps/byob/`.
- Starter configs live in `src/nemotron/steps/byob/config/`.
- Runtime execution goes through `uv run --no-sync nemotron byob`.
- The notebook creates only example-local inputs, outputs, and a working config.

## Files

| File | Purpose |
| --- | --- |
| `build_mcq_benchmark.ipynb` | End-to-end notebook for preparing finance text, writing a tiny-run config, running BYOB, and validating the final parquet. |
| `assets/` | Example corpus location. The notebook creates `assets/wiki_finance/*.txt`. |
| `config/` | Example working config location. The notebook writes generation and Curator experimental translation configs from the packaged step configs. |
| `outputs/` | Example output location for `seed.parquet`, `stage_cache/`, `benchmark_raw.parquet`, and `benchmark.parquet`. |

## Run

From the repository root:

```bash
uv sync --extra byob --group run
uv run --extra byob --group run --with jupyterlab jupyter lab use-case-examples/build-your-own-benchmark/build_mcq_benchmark.ipynb
```

The notebook defaults to a dry command preview so it does not accidentally spend model quota. Set `RUN_BYOB = True` in the run cell after reviewing the generated config and credentials. NVIDIA-hosted endpoints use `NGC_API_KEY` or `NVIDIA_API_KEY`; optional model overrides are read from `BYOB_MODEL`, `TRANSLATION_MODEL`, and `TRANSLATION_BASE_URL`.

The equivalent CLI is:

```bash
uv run --no-sync nemotron byob --list-families
uv run --no-sync nemotron byob --family mcq --stage all --config use-case-examples/build-your-own-benchmark/config/finance_wiki.yaml
```

Optional translation uses:

```bash
uv run --no-sync nemotron byob --family mcq --stage translate --config use-case-examples/build-your-own-benchmark/config/finance_wiki_translate.yaml
```

For failure isolation, `all` can be split into the same two stages:

```bash
uv run --no-sync nemotron byob --family mcq --stage prepare --config use-case-examples/build-your-own-benchmark/config/finance_wiki.yaml
uv run --no-sync nemotron byob --family mcq --stage generate --config use-case-examples/build-your-own-benchmark/config/finance_wiki.yaml
```
