# Nemotron Customizer Airgap

This folder is scoped to **Nemotron Customizer** only: the workflows represented
by `src/nemotron/steps/`. It is not a repo-wide airgap promise for every recipe,
demo, cookbook, or team-owned area in this repository.

The airgap design has two images and one persistent asset area:

- **Submitter image**: lightweight local image with this repo, `uv`, the
  Nemotron Python environment, offline env defaults, and the wheelhouse needed
  to submit/navigate steps.
- **Task images**: executor images that actually run data prep, training, RL,
  eval, conversion, or optimization. These should be mirrored or rebuilt as
  derivative images for production.
- **Persistent assets**: models, datasets, checkpoints, customer JSONL/parquet,
  HF caches, and optional repo caches on storage visible to Lepton, Slurm, or
  run:ai/DGX Cloud workers.

Large models and datasets are **not** baked into either image by default. Stage
them on persistent storage and pass those paths through step config overrides.

## Stage Map

Use this flow for one step or a multi-step workflow:

| Stage | Where | Output | Rule |
| --- | --- | --- | --- |
| 1. Select and lock | Connected prep machine | `airgap.lock.yaml` | Lock all `step_id:config` targets together. |
| 2. Fetch submitter runtime | Connected prep machine | `airgap-bundle/runtime/` | Fetch repo wheels and small support assets only. |
| 3. Probe and rebuild task images | Connected prep machine with task images | `remote-gaps.yaml`, derivative task images | Probe exact executor images, then bake missing small packages and repo checkouts. |
| 4. Stage persistent assets | Customer/executor storage | HF cache, datasets, checkpoints, repos | Remote jobs read models/data from mounted storage. |
| 5. Build submitter image | Connected or airgapped Docker host | Transportable local image | Bake source, `uv`, offline env, and submitter wheels. |
| 6. Smoke and verify | Inside submitter image | Remote job logs and outputs | Logs should show offline/mounted paths and no public fetches. |

## Step Hints

Each step may include an `airgap.yaml` beside `step.py`. The file is intentionally
small and step-local:

```yaml
version: 1
config_fields:
  - path: hf_model_path
    kind: hf_model
  - path: output_dir
    kind: local_path
task_container:
  family: megatron-bridge
  required_imports: [typer, rich, pydantic_settings, omegaconf]
  repo_mounts:
    - repo: Megatron-Bridge
      target: /opt/Megatron-Bridge
```

The lock compiler combines `airgap.yaml`, `step.toml`, config values, and static
imports. If a model/dataset field is overridden to `/mnt/...`, it is treated as a
customer-mounted path, not as a Hugging Face download.
Use `required: false` for example or pattern-default assets; those stay visible
as optional plan items but are not part of the required customer download list.

## Connected Prep

Run these stages on a machine with access to git, package indexes, Docker
registries, Hugging Face, and any private artifact stores.

### 1. Clone and sync

```bash
git clone "$NEMOTRON_REPO_URL" Nemotron
cd Nemotron
git rev-parse HEAD

uv sync --frozen --no-dev
uv run nemotron step list
```

Record the repo commit SHA in the customer handoff.

### 2. Select and lock a workflow

Prefer locking the full workflow at once:

```bash
export WORKFLOW_NAME=customer-workflow
export EXECUTOR_ENV_FILE=env.toml

uv run nemotron step airgap lock-workflow \
  --name "$WORKFLOW_NAME" \
  --env-file "$EXECUTOR_ENV_FILE" \
  --profile lepton_prep_sft_packing_tiny \
  --profile lepton_sft_megatron_bridge_tiny \
  -o deploy/nemotron-customizer/airgap/airgap.lock.yaml \
  prep/sft_packing:tiny \
  sft/megatron_bridge:tiny
```

Single-step locking is still useful for debugging:

```bash
uv run nemotron step airgap lock sft/megatron_bridge \
  -c tiny \
  -o deploy/nemotron-customizer/airgap/airgap.lock.yaml
```

Inspect the plan and static checks:

```bash
uv run nemotron step airgap plan deploy/nemotron-customizer/airgap/airgap.lock.yaml
uv run nemotron step airgap verify deploy/nemotron-customizer/airgap/airgap.lock.yaml
```

### 3. Fetch submitter runtime

```bash
uv run nemotron step airgap fetch deploy/nemotron-customizer/airgap/airgap.lock.yaml \
  -b deploy/nemotron-customizer/airgap/airgap-bundle \
  --include-wheels
```

This creates:

```text
deploy/nemotron-customizer/airgap/airgap-bundle/
  runtime/
    wheels/
    requirements-airgap.txt
    requirements-airgap.source.txt
    requirements-build-system.txt
    offline.env
  assets/
    repos/
    lepton/
    hf-cache/hub/
```

`--include-wheels` is for the submitter image. It does not download large HF
models or datasets.

If a task image needs repo checkouts baked or staged, fetch only repos:

```bash
uv run nemotron step airgap fetch deploy/nemotron-customizer/airgap/airgap.lock.yaml \
  -b deploy/nemotron-customizer/airgap/airgap-bundle \
  --include-repos
```

Use `--include-assets` only for local smoke/transport bundles where downloading
all external assets is intentional.

## Task Images

Task images are the images named in `env.toml`/runspec and pulled by the remote
executor. The submitter image does not replace them.

### 1. Probe exact task images

First ensure the task images are available on the connected prep machine. Then:

```bash
uv run nemotron step airgap probe-task-images \
  deploy/nemotron-customizer/airgap/airgap.lock.yaml \
  --execute \
  -o deploy/nemotron-customizer/airgap/remote-gaps.yaml \
  --requirements-dir deploy/nemotron-customizer/airgap \
  --repo-mounts-output deploy/nemotron-customizer/airgap/task-repo-mounts.json
```

The probe imports only the modules inferred from step code and `airgap.yaml`.
It writes the missing imports and package hints. If nothing is missing, keep
`task-requirements.txt` empty and use `pip_install_mode=preinstalled`.

### 2. Build a derivative task image for production

Use the per-image requirements file emitted by the probe, or keep the default
empty `task-requirements.txt` when the image already has everything:

```bash
uv run nemotron step airgap build-task-image \
  deploy/nemotron-customizer/airgap/airgap.lock.yaml \
  --step-id sft/megatron_bridge \
  -f deploy/nemotron-customizer/airgap/Dockerfile.task \
  --task-requirements deploy/nemotron-customizer/airgap/task-requirements-sft-megatron_bridge-<hash>.txt \
  --task-repo-mounts deploy/nemotron-customizer/airgap/task-repo-mounts.json \
  -t "$CUSTOMER_REGISTRY/nemotron-sft-task-airgap:<tag>" \
  --execute
```

Push/mirror the derivative task image to the registry or image path reachable
by the executor, then pin the executor profile to that image digest. For
NeMo-RL steps, start from a NeMo-RL-compatible image such as
`nvcr.io/nvidia/nemo-rl:v0.6.0` or the customer-mirrored digest of it.
If the workflow has no repo mounts, `build-task-image` will materialize an
empty `task-repo-mounts.json` instead of making the customer create one by hand.

For smoke-only runs, an offline wheelhouse mounted on persistent storage is
acceptable:

```toml
pip_install_mode = "offline_wheelhouse"
pip_wheelhouse = "/mnt/lustre-shared/airgap/wheels/nemo-25.11-nano"
pip_no_deps = true
pip_required_imports = ["cosmos_xenna", "obstore", "cattrs", "portpicker"]
```

For production, prefer:

```toml
pip_install_mode = "preinstalled"
pip_required_imports = ["typer", "rich", "pydantic_settings", "omegaconf"]
```

## Persistent Assets

Stage models and datasets directly on executor-visible storage. Typical paths:

```bash
export AIRGAP_ROOT=/mnt/lustre-shared/airgap
export AIRGAP_HF_HOME=$AIRGAP_ROOT/assets/hf-cache
export AIRGAP_HF_CACHE=$AIRGAP_HF_HOME/hub
export AIRGAP_DATASETS_CACHE=/mnt/lustre-shared/hf/datasets
export AIRGAP_REPOS=$AIRGAP_ROOT/assets/repos
```

HF cache shape for a model:

```text
$AIRGAP_HF_CACHE/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/
  refs/
  snapshots/
  blobs/
```

Or pass a plain mounted directory:

```bash
hf_model_path=/mnt/lustre-shared/models/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
```

For customer datasets, prefer local JSONL/parquet/blend files on persistent
storage:

```bash
blend_path=/mnt/lustre-shared/customer/rl_blends/blend.json
dataset.path_or_dataset_id=/mnt/lustre-shared/customer/sft/train.jsonl
```

## Submitter Image

Build the local submitter image after `fetch --include-wheels`:

```bash
export IMAGE_TAG=nemotron-customizer-airgap:latest

uv run nemotron step airgap build deploy/nemotron-customizer/airgap/airgap.lock.yaml \
  -f deploy/nemotron-customizer/airgap/Dockerfile \
  -t "$IMAGE_TAG" \
  --execute
```

If the base image exposes a non-default interpreter, pass
`--python-bin python3.10`, `--python-bin python3`, or a full path.

Save it for transport:

```bash
mkdir -p deploy/nemotron-customizer/airgap/release
docker save "$IMAGE_TAG" | gzip > deploy/nemotron-customizer/airgap/release/submitter-image.tar.gz
tar -czf deploy/nemotron-customizer/airgap/release/metadata.tar.gz \
  deploy/nemotron-customizer/airgap/airgap.lock.yaml \
  deploy/nemotron-customizer/airgap/Dockerfile \
  deploy/nemotron-customizer/airgap/Dockerfile.task \
  deploy/nemotron-customizer/airgap/README.md
```

## Remote Smoke

Run from inside the submitter image. The remote worker still uses the task image
selected by the executor profile or override.

```bash
docker run --rm -it \
  --network host \
  -v "$HOME/.cache/lepton:/root/.cache/lepton:ro" \
  -v "$PWD/env.toml:/workspace/Nemotron/env.toml:ro" \
  -e NEMOTRON_ENV_FILE=/workspace/Nemotron/env.toml \
  "$IMAGE_TAG" \
  uv run nemotron step run prep/sft_packing \
    -c tiny \
    -b lepton_prep_sft_packing_tiny \
    blend_path=/mnt/lustre-shared/customer/sft/blend.json \
    output_dir=/mnt/lustre-shared/output/test/sft_dataprep_airgap \
    force=true \
    'run.env.startup_commands=[]' \
    run.env.pip_install_mode=preinstalled \
    run.env.env_vars.HF_HOME="$AIRGAP_HF_HOME" \
    run.env.env_vars.HF_HUB_CACHE="$AIRGAP_HF_CACHE" \
    run.env.env_vars.HF_HUB_OFFLINE=1 \
    run.env.env_vars.TRANSFORMERS_OFFLINE=1 \
    run.env.env_vars.HF_DATASETS_OFFLINE=1 \
    'run.env.env_vars.WANDB_PROJECT=""' \
    run.env.env_vars.WANDB_ENABLED=false
```

Read logs with this checklist:

- no `raw.githubusercontent.com`, unless a connected fallback was intentional
- no PyPI/index access; pip should show `--no-index` or be skipped
- no public `git clone`; repo checkouts should come from baked or mounted repos
- no Hugging Face network downloads in offline mode
- HF cache root points at the directory containing `models--...`
- W&B is disabled, offline, or pointing at an in-network service

If a run fails with Hugging Face `LocalEntryNotFoundError`, offline mode is
working; the model/tokenizer is missing from the mounted cache or config path.
