# Megatron-Bridge Pretrain

Use `pretrain/megatron_bridge` when model size, sequence length, or throughput requires Megatron distributed parallelism.

Use this README for workflow and pitfalls; use `step.toml` for the exact artifact, parameter, strategy, and error manifest before editing configs or code.

## Inputs And Outputs

- Consume `binidx` data and `blend.json` from `data_prep/pretrain_prep`.
- Optionally initialize from a base checkpoint or HF weights for continued pretraining.
- Produce `checkpoint_megatron`.
- Validate data loading, parallelism, and checkpoint output with a short run before scaling token budget.

## CLI And Overlay Knobs

Start from `config/tiny.yaml` for launch validation and `config/default.yaml`
for the production-shaped topology. In a project overlay, developers usually
change:

- `dataset`: use the schema supported by the selected Bridge runtime, with
  bin/idx prefixes and weights from the prep-emitted `blend.json`.
- `model.seq_length` and `dataset.seq_length`: keep them aligned. Set
  `recipe.seq_length` only when the recipe accepts that argument.
- `load_hf_weights` or checkpoint fields: set explicitly for CPT or resume.
- `train.micro_batch_size`, `train.global_batch_size`, and TP/PP/CP/EP sizes:
  size them against the env profile.
- Checkpoint save directory, validation cadence, learning-rate schedule, and
  train iterations.

Example template; replace the quoted placeholders before running:

```bash
uv run nemotron steps run pretrain/megatron_bridge \
  -c "<config-path>" \
  model.seq_length="<sequence-length>" \
  dataset.seq_length="<sequence-length>"
```

Related patterns:

- Check `src/nemotron/steps/patterns/pretrain-token-budget-before-scale.md` before changing distributed strategy.
- Check `src/nemotron/steps/patterns/prep-data-is-tokenizer-locked.md` before reusing bin/idx data.

## Config Nuances

- Keep model and dataset sequence lengths identical; Bridge validates these before setup. Check the selected recipe's signature before adding arguments; some recipes take no arguments.
- Preserve the prep-emitted `blend.json` and map its bin/idx prefixes and weights into the supported dataset schema, such as `blend_per_split`. Do not assume that a runtime accepts the manifest directly as `dataset.data_paths`.
- For supported MoE configurations, keep `model.sequence_parallel: true` with tensor parallelism.
- If Transformer Engine userbuffers are enabled on a system without CUDA multicast support, set `run.env.env_vars.UB_SKIPMC: "1"` or default it in `step.py` before Bridge initialization.
- Use `train.global_batch_size` as a multiple of data-parallel size; start with `train.micro_batch_size: 1` when validating a new parallelism shape.

## Run It

Run from the repository root after installing dependencies. These are command
templates: replace the quoted placeholders with a compatible config path and,
for remote execution, an existing profile from local `env.toml`.

```bash
# Validate the config without allocating GPUs.
uv run nemotron steps run pretrain/megatron_bridge \
  -c "<config-path>" --dry-run

# Run locally only with a configuration that fits one CUDA GPU.
uv run nemotron steps run pretrain/megatron_bridge \
  -c "<config-path>"

# Validate the selected Slurm or Lepton profile, then submit a batch job.
uv run nemotron steps run pretrain/megatron_bridge \
  -c "<config-path>" --batch "<profile>" --dry-run
uv run nemotron steps run pretrain/megatron_bridge \
  -c "<config-path>" --batch "<profile>"

# Alternatively, attach to a short Slurm debugging run.
uv run nemotron steps run pretrain/megatron_bridge \
  -c "<config-path>" --run "<profile>"
```

Start with a short training run, validation, and a checkpoint save before scaling
the token budget. A dry-run checks configuration assembly; only a real run
validates GPU memory, distributed communication, data access, and checkpoint I/O.

## Reuse Defaults

Use an existing config when its parameters already match the task. For small
changes, prefer CLI overrides; create a reusable config only when there are
meaningful differences in model, data, training, or runtime requirements.

Inherit `default.yaml` only when its recipe and dataset schema are compatible.
Inheritance deep-merges mappings, so changing a recipe target does not remove
the previous recipe's arguments. Likewise, obsolete dataset fields remain unless
explicitly handled. Keep a standalone config when inheritance would retain
unsupported settings.

Do not remove model overrides solely because they match recipe defaults: HF
AutoBridge replaces the model provider, and the runner reapplies explicit
`model` overrides afterward. Compare the effective configuration, including
runner and library defaults, before removing repeated values.

## Choose an Execution Environment

| Environment | Suggested use | Setup considerations |
| --- | --- | --- |
| Local | Data/config checks and short training runs that fit one GPU | The current Steps local training backend launches one process. Use a compatible CUDA GPU and installed software stack; CPU-only machines can prepare data and inspect configs. |
| Slurm | Multi-node CPT on an existing GPU cluster | Configure the account, partition, time limit, nodes, GPUs, container, and shared filesystem. Use the cluster's supported networking configuration. |
| Lepton | Multi-node CPT with workspace-managed containers and GPU resources | Configure the node group, resource shape, worker count, image, and persistent mounts. Ensure every worker can reach the same data and checkpoints. |

For Slurm, prefer batch submission for long runs; use an attached allocation for
short debugging sessions. Keep communication-heavy parallel groups within a node
where practical and use the cluster interconnect for communication across nodes.

For local training, adapt the model and parallelism to one process. Increasing
node or GPU counts in the YAML does not enable local distributed training in the
current Steps backend. A multi-node config is not a local smoke config simply
because it runs for fewer iterations. Choose a smaller compatible model or use
remote GPUs when the full model and optimizer state do not fit. The local
command uses the installed software environment; naming a container in the YAML
does not turn it into a container launch. Local execution is the default when
neither the config nor an execution flag selects a remote executor.

When changing environments, align the node/GPU settings in both the config and
profile, and replace storage paths with locations visible to the target runtime.
YAML resource settings take precedence over inherited profile values; explicit
CLI overrides are applied last. Authentication belongs in the local environment
or executor profile, not shared training configs.

## Memory and Batch Sizing

Start with microbatch size one. Global batch size must be divisible by microbatch
size times data-parallel size. Recalculate this when changing GPU count or model
parallelism; mixture-of-experts models also have separate expert data-parallel
groups.

If OOM occurs during forward/backward, examine sequence length, microbatch size,
activation recomputation, and model parallelism. If it first occurs during the
optimizer update, examine optimizer-state memory and distributed-optimizer
sharding. More nodes reduce this memory only when the parallel layout provides
additional replicas over which the optimizer can shard state. CPU optimizer
offload is another option, with additional host-memory use and transfer overhead.

## Tracking and Checkpoints

Reuse compatible [prepared data](../../data_prep/pretrain_prep/README.md).
Configure W&B in the selected environment and enable the trainer's logger fields.
Use a common project/group across pipeline stages and distinct run names. Track
loss, learning rate, gradient norm, memory, validation, and checkpoint completion.

Reusing a checkpoint save directory auto-resumes when a loadable checkpoint is
present. For a fresh run from HF weights, choose a new checkpoint directory and
new logging destinations. A short smoke run verifies the pipeline; it does not
establish model quality.

## Repository Layout

- Manifest: `src/nemotron/steps/pretrain/megatron_bridge/step.toml`
- Runner: `src/nemotron/steps/pretrain/megatron_bridge/step.py`
- Configs: `src/nemotron/steps/pretrain/megatron_bridge/config/default.yaml`, `src/nemotron/steps/pretrain/megatron_bridge/config/tiny.yaml`
- Shared runner: `src/nemotron/steps/_runners/megatron_bridge.py`

## Guardrails

- Run `data_prep/pretrain_prep` first unless compatible bin/idx data already exists.
- Verify data paths and checkpoint writes on the target executor before long jobs.
- Convert Megatron checkpoints only when the downstream consumer requires HF layout.
