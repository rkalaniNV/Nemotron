# Build Your Own Benchmark

This example shows the current Nemotron BYOB flow for creating a domain-specific MCQ benchmark from finance-related text.

The notebook is intentionally aligned with the agentic step structure:

- The reusable step lives in `src/nemotron/steps/byob/`.
- Starter configs live in `src/nemotron/steps/byob/config/`.
- Runtime execution goes through `nemotron byob`.
- The notebook creates only example-local inputs, outputs, and a working config.

## Files


| File                        | Purpose                                                                                                                                                        |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `build_mcq_benchmark.ipynb` | End-to-end notebook for preparing finance text, writing a working config, running BYOB, and previewing the final parquet.                                      |
| `assets/`                   | Example corpus root. The notebook writes finance `*.txt` files into `assets/wiki_finance/` if the directory is empty.                                          |
| `config/`                   | Example working config location. The notebook writes generation and Curator experimental translation configs from the packaged step configs.                   |
| `outputs/`                  | Example output location. After a successful run contains `<expt_name>/seed.parquet`, `<expt_name>/benchmark_raw.parquet`, and `<expt_name>/benchmark.parquet`. |


## Setup

From the repository root (`Nemotron/`), provision the project venv with the `byob` extra:

```bash
# 1. install uv (one-time; skip if you already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. add jupyter 
uv add jupyter
```

## Run

From the repository root:

```bash
uv run --extra byob jupyter lab use-case-examples/build-your-own-benchmark/build_mcq_benchmark.ipynb
```

The notebook defaults to a dry command preview (`RUN_BYOB = False`, `RUN_TRANSLATION = False`) so it does not accidentally spend model quota. Flip those flags to `True` in their respective cells after you have reviewed the generated config.

### Do you already have a dataset?

- **No, I don't have a dataset.** Run **notebook sections 1 → 4** (`## 1. Locate the repository and BYOB step assets` through `## 4. Write the example config from the packaged BYOB config`). These cells perform the env / Ray / Curator sanity checks, set credentials, download the example finance corpus into `assets/wiki_finance/`, and write the working config to `config/finance_wiki.yaml`. After section 4 you can stop the notebook and continue from the CLI examples below.
- **Yes, I already have a dataset.** Skip the notebook entirely and jump to [Bring your own documents](#bring-your-own-documents).

Once `assets/wiki_finance/` and `config/finance_wiki.yaml` are in place (either via the notebook or by hand), run the same pipeline from the CLI without re-executing the notebook. For example:

```bash
uv run --extra byob nemotron byob --list-families
uv run --extra byob nemotron byob --family mcq --stage prepare  --config use-case-examples/build-your-own-benchmark/config/finance_wiki.yaml
uv run --extra byob nemotron byob --family mcq --stage generate --config use-case-examples/build-your-own-benchmark/config/finance_wiki.yaml
```

Optional translation uses:

```bash
uv run --extra byob nemotron byob --family mcq --stage translate --config use-case-examples/build-your-own-benchmark/config/finance_wiki_translate.yaml
```

> **Need a translation config?** If you do not already have `config/finance_wiki_translate.yaml`, run section `**## 7. Optional translation config`** in the notebook. That cell loads the packaged BYOB translation config (`src/nemotron/steps/byob/config/translate.yaml`), applies the example-local overrides, writes `config/finance_wiki_translate.yaml`, and (when `RUN_TRANSLATION = True`) invokes the `translate` stage.

## Bring your own documents

If you already have a corpus of documents and want to skip the notebook's example-corpus creation step, you can run the CLI directly against your own files.

1. Place your documents as `*.txt` files under `<input_dir>/<target_subject>/`, where `<target_subject>` matches a key in `target_source_mapping` in your YAML config. For example, with the packaged `finance_wiki.yaml` the layout is:
  ```text
   /path/to/your/corpus/
   └── wiki_finance/
       ├── doc_01.txt
       ├── doc_02.txt
       └── ...
  ```
2. Update `input_dir` in your config to point at the corpus root (the parent of `wiki_finance/`), and adjust `target_source_mapping`, `source_subjects`, and `hf_dataset` if your domain differs from the finance example.
3. Export the credentials required by the pipeline before running the CLI. NVIDIA-hosted generation/judge models require `NGC_API_KEY` (or `NVIDIA_API_KEY`); `HF_TOKEN` is needed when the configured `hf_dataset` or embedding model is gated on Hugging Face:
  ```bash
   export NGC_API_KEY=nvapi-...        # or export NVIDIA_API_KEY=nvapi-...
   export HF_TOKEN=hf_...               # optional, but required for gated HF assets
  ```
4. Run the BYOB pipeline against your config:
  ```bash
   uv run --extra byob nemotron byob --list-families
   uv run --extra byob nemotron byob --family mcq --stage prepare  --config /path/to/your_config.yaml
   uv run --extra byob nemotron byob --family mcq --stage generate --config /path/to/your_config.yaml
  ```
   The `prepare` stage writes `<output_dir>/<expt_name>/seed.parquet` from your `*.txt` files, and `generate` produces `benchmark.parquet` under the same output directory.

