---
name: nemotron-airgap
description: Prepare Nemotron Customizer step workflows for disconnected customers by locking selected steps, building the local submitter/runtime image, staging remote assets, and validating no public-network behavior.
---

# Airgap Readiness

Use this for customer airgap preparation of workflows represented by
`src/nemotron/steps/`. It does not certify unrelated recipes, cookbooks, docs,
or examples.

## Mental Model

- **Submitter image**: local image with repo source, `uv`, offline wheels, and
  offline env. Use it to submit remote jobs during real smoke tests.
- **Remote task image**: mirrored or derivative Nano3, Super3, RL, eval, prep,
  or optimization image that Lepton/DGX/Slurm workers actually run.
- **Persistent assets**: models, datasets, git repos, URL payloads, checkpoints,
  and remote wheelhouses on executor-visible storage.

Do not put large models and datasets into the submitter image by default.

## Workflow

1. **Select and lock** all workflow targets up front, for example
   `prep/sft_packing:tiny sft/megatron_bridge:tiny`.
2. **Fetch submitter runtime** wheels and small support assets.
3. **Probe and rebuild task images** with `probe-task-images` and
   `build-task-image`; bake small missing packages and repo checkouts into
   derivative task images for production.
4. **Stage persistent assets** on executor-visible storage: HF cache, datasets,
   checkpoints, and optional remote wheelhouse.
5. **Build submitter image** from `airgap-bundle/runtime/`; keep large assets
   outside the image.
6. **Smoke remotely from inside the submitter image**, while the remote workers
   use the mirrored task image selected by the executor profile.
7. **Verify logs** show mounted wheelhouse/cache paths and no public fetches.

When guiding a user, name the current stage before each command or diagnosis.
If a one-off fix is needed, fold the learning back into one of these stages so
the customer path stays simple.

For a config-driven catalog step, use `nemotron step run env/airgap` with
`targets`, `profiles`, and `env_file` overrides. Use the `nemotron step airgap`
CLI for fetch/build/release commands.

## Remote Rules

- Lepton/DGX/Slurm jobs should use the mirrored task image from the executor
  profile, not the submitter image, unless intentionally configured otherwise.
- NeMo-RL jobs (`rl/nemo_rl/dpo`, RLVR, RLHF) should use
  `nvcr.io/nvidia/nemo-rl:v0.6.0` or the customer-mirrored digest of that
  image as the remote task image.
- Use `pip_install_mode=preinstalled` when the task image already contains
  required packages.
- Use `offline_wheelhouse` only with a mounted wheelhouse. Prefer
  `pip_no_deps=true` plus `pip_required_imports` for small gap-fills. The same
  pip policy applies to Lepton, DGX Cloud/run:ai, and Slurm.
- Remove online `startup_commands` such as `pip install`, `wget`, `curl`, and
  `git clone`. Move those into the task image, asset bundle, or wheelhouse.
- Use `airgap.yaml` next to each step to keep model/data/service path hints and
  task-image import hints close to the step that owns them.
- For Lepton, provide `lepton_env_to_pytorch.sh` through mounted assets or set
  `NEMOTRON_LEPTON_INIT_MODE=skip` when the image/profile already supplies the
  needed distributed env.
- For RL data prep, prefer materialized local JSONL/parquet blends on
  persistent storage when validating airgap. Cached HF datasets can be offline
  but still fail when their metadata was written by a different `datasets`
  version than the remote task image.
- For one-iteration Megatron Bridge smoke tests, set
  `scheduler.lr_warmup_iters=0`. Otherwise `train.train_iters` must be greater
  than warmup, or Megatron fails during optimizer scheduler setup.

## Proven Smoke Pattern

Data prep:

```bash
docker run --rm -it --network host \
  -v "$HOME/.cache/lepton:/root/.cache/lepton:ro" \
  -v "$PWD/env.toml:/workspace/Nemotron/env.toml:ro" \
  -e NEMOTRON_ENV_FILE=/workspace/Nemotron/env.toml \
  "$IMAGE_TAG" \
  uv run nemotron step run prep/sft_packing \
    -c tiny \
    -b lepton_prep_sft_packing_tiny \
    output_dir=/mnt/lustre-shared/output/test/sft_dataprep_airgap_az \
    force=true \
    'run.env.startup_commands=[]' \
    run.env.pip_install_mode=offline_wheelhouse \
    run.env.pip_wheelhouse=/mnt/lustre-shared/airgap/wheels/nemo-25.11-nano \
    run.env.pip_no_deps=true \
    'run.env.pip_required_imports=[cosmos_xenna]' \
    'run.env.env_vars.WANDB_PROJECT=""' \
    run.env.env_vars.WANDB_ENABLED=false
```

SFT after HF cache staging:

```bash
docker run --rm -it --network host \
  -v "$HOME/.cache/lepton:/root/.cache/lepton:ro" \
  -v "$PWD/env.toml:/workspace/Nemotron/env.toml:ro" \
  -e NEMOTRON_ENV_FILE=/workspace/Nemotron/env.toml \
  "$IMAGE_TAG" \
  uv run nemotron step run sft/megatron_bridge \
    -c tiny \
    -b lepton_sft_megatron_bridge_tiny \
    'dataset.packed_sequence_specs.packed_train_data_path=/mnt/lustre-shared/output/test/sft_dataprep_airgap_az/splits/train/*.parquet' \
    checkpoint.save=/mnt/lustre-shared/output/test/sft_megatron_bridge_airgap_az/sft-tiny-2node \
    logger.wandb_project=null \
    'run.env.startup_commands=[]' \
    run.env.pip_install_mode=preinstalled \
    'run.env.pip_required_imports=[typer,rich,pydantic_settings]' \
    run.env.env_vars.NEMOTRON_AIRGAP_REPOS=/mnt/lustre-shared/airgap/assets/repos \
    run.env.env_vars.HF_HOME=/mnt/lustre-shared/airgap/assets/hf-cache \
    run.env.env_vars.HF_HUB_CACHE=/mnt/lustre-shared/airgap/assets/hf-cache/hub \
    run.env.env_vars.HF_HUB_OFFLINE=1 \
    run.env.env_vars.TRANSFORMERS_OFFLINE=1 \
    run.env.env_vars.HF_DATASETS_OFFLINE=1 \
    'run.env.env_vars.WANDB_PROJECT=""' \
    run.env.env_vars.WANDB_ENABLED=false
```

If SFT fails with `LocalEntryNotFoundError`, offline mode is working and the
model is missing from the mounted HF cache. The expected cache root contains
`models--<org>--<repo>/refs`, `snapshots`, and `blobs`.

CPT/pretrain smoke follows the same pattern:

- run `prep/pretrain_prep` against a pre-staged local blend on persistent
  storage, usually with `execution_mode=batch` and a small CPU profile
- run `pretrain/megatron_bridge` against the generated `blend.json`
- add `scheduler.lr_warmup_iters=0` when using `train.train_iters=1`
- keep `HF_HOME`, `HF_HUB_CACHE`, `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`,
  `NEMOTRON_LEPTON_INIT_SCRIPT`, and the remote wheelhouse path explicit
