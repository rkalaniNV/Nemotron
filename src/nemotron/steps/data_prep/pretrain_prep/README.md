# Pretrain Bin/Idx Prep

Use `data_prep/pretrain_prep` when downstream pretraining expects Megatron `binidx` data.

Use this README for workflow and pitfalls; use `step.toml` for the exact artifact, parameter, strategy, and error manifest before editing configs or code.

## Inputs And Outputs

- Consume curated text data through a blend file, usually local JSONL, parquet, or HF dataset references.
- Produce bin/idx shards plus `blend.json` split metadata.
- Validate tokenization and emitted `blend.json` on a small subset before full prep.

## CLI And Overlay Knobs

Start from `config/tiny.yaml` for wiring and `config/default.yaml` for the
production-shaped example. In a project overlay, developers usually change:

- `blend_path`: source text blend with local or HF-backed entries.
- `tokenizer.model`: tokenizer used by the downstream pretraining model.
- `num_shards`: match filesystem and trainer throughput.
- `valid_shards`, `test_shards`, and `split_seed`: keep validation reproducible.
- `max_doc_tokens` and `text_field`: set only when the data policy requires it.

Example template; replace the quoted placeholders before running:

```bash
uv run nemotron steps run data_prep/pretrain_prep \
  -c "<config-path>" \
  blend_path="<blend-path>" \
  tokenizer.model="<tokenizer>"
```

Related patterns:

- Check `src/nemotron/steps/patterns/prep-data-is-tokenizer-locked.md` before changing tokenization, split, or sharding behavior.
- Check `src/nemotron/steps/patterns/pretrain-token-budget-before-scale.md` before creating production pretraining data.

## Config Nuances

- Emit and preserve `blend.json`; use its split paths and weights in the downstream trainer's supported dataset schema.
- Size tokenization workers to the available CPU and memory budget, leaving capacity for the pipeline driver.
- Keep `valid_shards`, `test_shards`, and `split_seed` explicit for deterministic prep.
- Rebuild bin/idx whenever `tokenizer.model` or `sequence_length` assumptions change.

## Run It

Run from the repository root after installing dependencies. The following are
command templates: replace the quoted placeholders with a config path and, for
remote execution, an existing profile from local `env.toml`.

```bash
# Validate the config without running tokenization.
uv run nemotron steps run data_prep/pretrain_prep \
  -c "<config-path>" --dry-run

# Run locally with locally accessible inputs and output storage.
uv run nemotron steps run data_prep/pretrain_prep \
  -c "<config-path>"

# Validate the selected Slurm or Lepton profile, then submit a batch job.
uv run nemotron steps run data_prep/pretrain_prep \
  -c "<config-path>" --batch "<profile>" --dry-run
uv run nemotron steps run data_prep/pretrain_prep \
  -c "<config-path>" --batch "<profile>"
```

The profile selects the remote executor, resources, container, storage mounts,
and credentials. Local execution is the default when neither the config nor an
execution flag selects a remote executor. A dry-run checks configuration
assembly; it does not verify dataset access or resource availability on the
target machine.

## Repository Layout

- Manifest: `src/nemotron/steps/data_prep/pretrain_prep/step.toml`
- Runner: `src/nemotron/steps/data_prep/pretrain_prep/step.py`
- Configs: `src/nemotron/steps/data_prep/pretrain_prep/config/default.yaml`, `src/nemotron/steps/data_prep/pretrain_prep/config/tiny.yaml`
- Sample blend: `src/nemotron/steps/data_prep/pretrain_prep/data/blend_tiny.json`

## Reuse Defaults and Data

Reuse an existing config and blend when their parameters match the task. For a
few changes, use CLI overrides; for a reusable variation, inherit a compatible
config with `defaults: default.yaml` and include only the differences. Document
operational instructions here instead of creating another identical config.

A new blend JSON is needed only when dataset sources, subsets, splits, or weights
change. The input schema defaults `weight` to `1.0` and `text_field` to `text`,
so those values can be omitted. A source blend identifies text datasets; the
output `blend.json` identifies prepared shards. Keep those two roles distinct.

For a smoke test, limit rows, document length, and shard count. `max_rows` applies
per shard. Use the same tokenizer as the downstream model and preserve the
generated split manifest. Compatible prepared data can be reused across training
runs without repeating tokenization.

## Choose an Execution Environment

| Environment | Suggested use | Setup considerations |
| --- | --- | --- |
| Local | Small public-data samples, tokenizer checks, and debugging | CPU execution is sufficient for this step; reserve RAM and disk space for downloads and tokenized output. |
| Slurm | Larger prep jobs on an existing cluster | Prefer a CPU partition, set CPU and memory requests, and write output to storage visible to the training allocation. |
| Lepton | Containerized prep with workspace-managed resources | Select a CPU resource shape and mount persistent storage shared with the training job. |

Match `tokenization.cpus_per_worker` to the resources available after reserving
capacity for the driver. For a small CPU allocation, explicit batch execution
can simplify scheduling. Increasing worker count beyond available memory or CPU
capacity will not improve throughput.

Authenticate through the local environment or executor profile. Keep credentials
out of blend files and shared YAML. If W&B is enabled, use a common project and
group for prep and [CPT](../../pretrain/megatron_bridge/README.md), with distinct
run names for each stage; choose those names in the execution environment.

## Guardrails

- Treat bin/idx as tokenizer-locked; rebuild it when the tokenizer changes.
- Keep train, validation, and test split names consistent with downstream pretrain configs.
- Validate token counts, empty-document rates, and split leakage before a full pretraining run.
