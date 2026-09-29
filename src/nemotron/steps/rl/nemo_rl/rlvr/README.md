# NeMo-RL RLVR

Use `rl/nemo_rl/rlvr` when reward signals are verifiable and can be computed
programmatically. The Lightning preset is RLVR: NeMo-RL optimizes the policy
with GRPO while NeMo-Gym supplies verifier rewards.

Use this README for workflow and pitfalls; use `step.toml` for the exact
artifact, parameter, strategy, and error manifest before editing configs or
code.

## Inputs And Outputs

- Consume prompt `training_jsonl` with verifier fields such as answers.
- Consume an SFT `checkpoint_megatron` policy.
- Produce an RLVR-aligned `checkpoint_megatron`.
- Smoke the generic wiring with `nemotron steps run rl/nemo_rl/rlvr -c tiny`.

## CLI And Overlay Knobs

Start from `config/tiny.yaml` for runner validation. Use
`config/nemo_gym.yaml` when resource-server rewards are required. In a project
overlay, developers usually change:

- `data.train.data_path` and `data.validation.data_path`: prompt JSONL with
  verifier fields.
- `grpo.num_prompts_per_step` and `grpo.num_generations_per_prompt`: rollout
  batch size and reward variance.
- `policy.logprob_batch_size`: per-worker logprob microbatch after sharding.
- `env.should_use_nemo_gym`: switch to the NeMo-Gym runner only with matching
  resource-server config.
- `env.nemo_gym.config_paths`: resource-server configs for NeMo-Gym mode.

Example shape:

```bash
uv run nemotron steps run rl/nemo_rl/rlvr \
  -c "<project>/config/rlvr.yaml" \
  "data.train.data_path=<rl-prep>/train.jsonl" \
  "data.validation.data_path=<rl-prep>/validation.jsonl"
```

## Nemotron 3.5 Lightning On Lepton

`config/lightning35.yaml` is the policy-training subset of the released
Lightning workload. It uses the native Megatron-Bridge SFT checkpoint (no HF
conversion), GRPO, colocated vLLM generation, and the supported verifier agents
from the released RL blend. It does not provision the external judge and
sandbox pools required by the complete reference workload.

The [step entrypoint](step.py) selects the
[Lightning runner](../../../_runners/nemo_rl_lightning35.py) through
`nemotron.runner: lightning35`. This adapter supports **synchronous GRPO only**;
async GRPO and trajectory-collection mode fail early. It registers the prep
manifest resolver, checks checkpoint/data paths and agent coverage, then calls
the pinned NeMo-RL setup and training loop. It does not convert checkpoints
to HF or import a recipe runner. Use the matching
[Lightning data-prep preset](../../../data_prep/rl_prep/README.md#nemotron-35-lightning).

### 1. Build And Push The Pinned Image

Users must currently build and push their own NeMo-RL Docker image using the
dependency pins and patches documented below before running Lightning RL on
Lepton. The Steps launcher uses this image; it does not build or publish it.

We plan to update the configuration and instructions to use an official
NeMo-RL image once a compatible release becomes available.

Build on Linux x86_64. A GPU is not required for the build; the resulting
CUDA image runs on Lepton H100 nodes. The image must contain the pinned
Lightning refit fix and exactly the Ray version selected for the Lepton
cluster.

The following uses the 2.55.1 compatibility build needed by Lepton workspaces
that reject the pinned source's Ray 2.56.1:

```bash
git clone --recursive https://github.com/NVIDIA-NeMo/RL.git nemo-rl-lightning35
cd nemo-rl-lightning35

git fetch --no-recurse-submodules origin \
  7fa6e55192530ff1346d670ce74f9c70cab8f75b
git checkout -B lightning35-lepton \
  7fa6e55192530ff1346d670ce74f9c70cab8f75b
git submodule sync --recursive
git submodule update --init --recursive
git fetch --no-recurse-submodules origin ruit/fix-nemotronh-moe-refit-shard-dim
git -c user.name="Image Builder" -c user.email="builder@localhost" \
  cherry-pick 6bb55bc03adbdc4943fba5c9e452586c04afee88

sed -i \
  's/await self\.llm\.sleep(level=1)/await self.llm.sleep(level=2)/' \
  nemo_rl/models/generation/vllm/vllm_worker_async.py

sed -i \
  's/"ray\[default\]>=2\.55\.1"/"ray[default]==2.55.1"/' \
  pyproject.toml
sed -i \
  's/"ray\[default\]>=2\.56\.1"/"ray[default]==2.55.1"/' \
  3rdparty/Gym-workspace/Gym/pyproject.toml
uv lock --upgrade-package ray
uv lock --check

docker login nvcr.io --username '$oauthtoken'
export LIGHTNING35_RAY_VERSION=2.55.1
export LIGHTNING35_RL_IMAGE="nvcr.io/<ngc-org>/nemo-rl:lightning35-ray2551-7fa6e55-6bb55bc"

docker buildx build \
  --platform linux/amd64 \
  --progress=plain \
  --build-context nemo-rl=. \
  -f docker/Dockerfile \
  --target release \
  --build-arg MAX_JOBS=8 \
  --build-arg SKIP_SGLANG_BUILD=1 \
  --build-arg SKIP_TRTLLM_BUILD=1 \
  --build-arg NEMO_GYM_PREFETCH_CONFIGS="examples/nemo_gym/nemotron-3.5-lightning/rlvr.yaml" \
  -t "${LIGHTNING35_RL_IMAGE}" \
  --push .

docker buildx imagetools inspect "${LIGHTNING35_RL_IMAGE}"

docker pull --platform linux/amd64 "${LIGHTNING35_RL_IMAGE}"
docker run --rm "${LIGHTNING35_RL_IMAGE}" bash -lc '
  set -e
  /opt/nemo_rl_venv/bin/python -c \
    "import ray; assert ray.__version__ == \"2.55.1\", ray.__version__; print(ray.__version__)"
  test -d /opt/ray_venvs
  find /opt/ray_venvs -path "*/bin/python" -exec \
    {} -c "import ray,sys; assert ray.__version__ == \"2.55.1\", (sys.executable, ray.__version__); print(sys.executable, ray.__version__)" \;
'
```

The 2.55.1 dependency edit is a downstream compatibility patch because the
pinned NeMo-Gym source requests 2.56.1. Validate a one-step job before scaling.
If the workspace supports 2.56.1, omit those dependency edits and use 2.56.1
for both the image and `run.env.ray_version`.
The Lepton workspace's **RayClusters > Create Cluster > Ray Version** selector
is the authoritative allowlist; a Steps `--dry-run` cannot validate it.

Create a private-registry credential in Lepton and use its auth object name in
`image_pull_secrets`; do not put the NGC API key there.

### 2. Configure The Lepton Profiles

Run the remaining `nemotron steps` commands from your **Nemotron checkout**,
not from the separate NeMo-RL image-build directory. Replace angle-bracket
placeholders with your own values before running any example.

The repository's Lepton env generator includes these two focused profiles.
They inherit site-specific mounts, node group, and resource shapes from the
generic profiles:

```toml
[lepton_lightning35_rl_prep]
extends = "lepton_prep_rl_prep"
pip_extras = ["omegaconf", "cosmos-xenna", "typer", "rich", "pydantic-settings", "huggingface_hub", "datasets"]

[lepton_lightning35_rl_prep.env_vars]
RL_OUTPUT_DIR = "${oc.env:RL_PREP_OUTPUT_DIR}"
RL_PREP_OUTPUT_DIR = "${oc.env:RL_PREP_OUTPUT_DIR}"

[lepton_lightning35_rlvr]
extends = "lepton_rl_nemo_rl_rlvr"
container_image = "${oc.env:LIGHTNING35_RL_IMAGE}"
image_pull_secrets = ["${oc.env:LEPTON_REGISTRY_AUTH}"]
ray_version = "${oc.env:LIGHTNING35_RAY_VERSION}"
nodes = 31

[lepton_lightning35_rlvr.env_vars]
RL_PREP_OUTPUT_DIR = "${oc.env:RL_PREP_OUTPUT_DIR}"
RL_INITIAL_CHECKPOINT = "${oc.env:RL_INITIAL_CHECKPOINT}"
RL_OUTPUT_DIR = "${oc.env:RL_OUTPUT_DIR}"
```

Generate a new Lepton file once, or keep using an existing `env.toml` that has
equivalent profiles. For a new file:

```bash
# Run from the root of your Nemotron checkout.
uv run nemotron steps run env/env_toml -c lepton
export NEMOTRON_ENV_FILE="${PWD}/env.lepton.toml"
```

Edit the generated node group, resource shapes, and storage mount to match
your workspace before submission; the template contains site placeholders.
The generic GPU shape defaults to `gpu.8xa100-80gb`; for H100 runs, set
`LEPTON_GPU_SHAPE` to your workspace's eight-GPU H100 shape or edit the profile.
For existing profiles, use `export NEMOTRON_ENV_FILE="${PWD}/env.toml"` instead.
Do not store API tokens in the env file. Export them in the submitting shell
so the inherited base profile resolves them.

### 3. Prepare And Verify The Released RL Data

```bash
export RL_PREP_OUTPUT_DIR="/mnt/lustre-shared/<user>/data/processed/lightning35_rl"
export RL_INITIAL_CHECKPOINT="/mnt/lustre-shared/<user>/checkpoints/lightning35-sft/iter_0000100"
export RL_OUTPUT_DIR="/mnt/lustre-shared/<user>/checkpoints/lightning35-rl-grpo-$(date +%Y%m%d-%H%M%S)"
export LEPTON_REGISTRY_AUTH="<lepton-registry-auth-name>"
export LIGHTNING35_RAY_VERSION=2.55.1
export LIGHTNING35_RL_IMAGE="nvcr.io/<ngc-org>/nemo-rl:lightning35-ray2551-7fa6e55-6bb55bc"

uv run nemotron steps run data_prep/rl_prep \
  -c lightning35 --batch lepton_lightning35_rl_prep --dry-run \
  "run.env.env_vars.RL_PREP_OUTPUT_DIR=${RL_PREP_OUTPUT_DIR:?Set RL_PREP_OUTPUT_DIR}" \
  "run.env.env_vars.RL_OUTPUT_DIR=${RL_PREP_OUTPUT_DIR:?Set RL_PREP_OUTPUT_DIR}"

# Remove --dry-run after checking the rendered mount and output directory.
uv run nemotron steps run data_prep/rl_prep \
  -c lightning35 --batch lepton_lightning35_rl_prep \
  "run.env.env_vars.RL_PREP_OUTPUT_DIR=${RL_PREP_OUTPUT_DIR:?Set RL_PREP_OUTPUT_DIR}" \
  "run.env.env_vars.RL_OUTPUT_DIR=${RL_PREP_OUTPUT_DIR:?Set RL_PREP_OUTPUT_DIR}"
```

The prep preset downloads the released `rlvr.jsonl`, restores its Hub-backed
question placeholders, keeps the verifier agents supplied by this topology,
and writes non-empty train and validation splits plus `manifest.json`. It also
removes collection-time `_ng_task_index` and `_ng_rollout_index` fields; rerun
prep after updating so old cached data is refreshed. Batch submission is
asynchronous. After the prep job finishes, run these checks on a
pod or host that mounts the shared filesystem (not on an unmounted submit Mac):

```bash
test -s "${RL_PREP_OUTPUT_DIR:?Set RL_PREP_OUTPUT_DIR}/manifest.json"
jq '{train, val, train_rows, val_rows, allowed_agent_names, filtered_rows}' \
  "${RL_PREP_OUTPUT_DIR}/manifest.json"
test -d "${RL_INITIAL_CHECKPOINT:?Set RL_INITIAL_CHECKPOINT}"
```

### 4. Compile, Smoke, Then Submit

For an eight-host diagnostic, use seven worker pods plus the Lepton Ray head.
Each host must supply eight GPUs. The batch sizes below preserve
`128 prompts * 16 generations = 2048` trajectories per update; they do not
guarantee memory safety for every sequence length. First compile it:

```bash
uv run nemotron steps run rl/nemo_rl/rlvr \
  -c lightning35 --batch lepton_lightning35_rlvr --dry-run \
  "run.env.container_image=${LIGHTNING35_RL_IMAGE:?Set LIGHTNING35_RL_IMAGE}" \
  "run.env.image_pull_secrets=[${LEPTON_REGISTRY_AUTH:?Set LEPTON_REGISTRY_AUTH}]" \
  "run.env.ray_version=${LIGHTNING35_RAY_VERSION:?Set LIGHTNING35_RAY_VERSION}" \
  run.env.nodes=7 \
  "run.env.env_vars.RL_PREP_OUTPUT_DIR=${RL_PREP_OUTPUT_DIR:?Set RL_PREP_OUTPUT_DIR}" \
  "run.env.env_vars.RL_INITIAL_CHECKPOINT=${RL_INITIAL_CHECKPOINT:?Set RL_INITIAL_CHECKPOINT}" \
  "run.env.env_vars.RL_OUTPUT_DIR=${RL_OUTPUT_DIR:?Set RL_OUTPUT_DIR}" \
  cluster.num_nodes=8 \
  policy.megatron_cfg.tensor_model_parallel_size=2 \
  policy.megatron_cfg.context_parallel_size=2 \
  policy.megatron_cfg.expert_model_parallel_size=16 \
  grpo.num_prompts_per_step=128 \
  policy.train_global_batch_size=2048 \
  grpo.val_batch_size=64 \
  grpo.max_num_steps=1
```

Inspect the dry-run for the private image, registry auth name, Ray 2.55.1,
shared mount, checkpoint, manifest, and 7/8 node counts. Remove `--dry-run` to
submit the one-step smoke. Keep the eight-host topology for the first real run:
remove both `--dry-run` and `grpo.max_num_steps=1`. Set `RL_OUTPUT_DIR` to a
fresh directory; do not reuse an older run containing a synthetic `step_0`
checkpoint. The 32-host preset is optional after the eight-host run is stable;
it uses `run.env.nodes=31`, `cluster.num_nodes=32`, and the preset's default
TP/CP and batch sizes. With the current Lepton backend, always keep
`cluster.num_nodes = run.env.nodes + 1` because the former includes the GPU
head and the latter counts worker pods.

A one-step smoke does not exercise the normal validation/save cadence or
prove long-run memory stability. With the preset, validation occurs every
5 steps and checkpoint saves every 10 (also subject to a wall-clock save
deadline). Exercise those phases before treating the run as validated.
`--dry-run` only compiles the launch configuration; it does not check remote
files, registry access, available GPUs, or runtime memory.

From a shell in the Ray head pod, inspect the submitted job with:

```bash
export RAY_ADDRESS=http://127.0.0.1:8265
ray status
ray job list
# Copy the submission_id from the job list.
SID="your-submission-id"
ray job status "${SID}"
ray job logs "${SID}" --follow
```

### 5. Troubleshooting Without Changing Recipes

**`None/manifest.json` or a missing data path.** An empty shell expansion such
as `run.env.env_vars.RL_PREP_OUTPUT_DIR=` overrides a valid profile value with
null, which the launcher exports as the string `None`. Keep the guarded
`${VAR:?Set VAR}` arguments above, pass a literal path, or omit the override
when the profile already supplies it. Check the manifest and both referenced
JSONL files on the mounted cluster filesystem, then submit a new job from the
current checkout. This error does not require rebuilding the image.

**Unsupported agent or `_ng_*` metadata.** Rerun the Lightning prep step and
check that training consumes its current manifest. Do not remove the agent
allowlist unless the additional reward services are also provisioned.

**OOM after initially successful steps.** Record the first traceback, failed
worker/node, step, and phase (rollout, log-probs, training, validation, save).
An `ActorDiedError` alone or a long delay before failure does not establish
the cause or prove a memory leak. Collect logs in the Ray head pod:

```bash
export RAY_ADDRESS=http://127.0.0.1:8265
ray job list
SID="your-submission-id"
ray job logs "${SID}" > "/tmp/lightning35-${SID}.log" 2>&1
grep -n -B 60 -A 100 -Ei \
  'out of memory|OutOfMemoryError|OOMKilled|memory pressure|memory usage|ActorDiedError' \
  "/tmp/lightning35-${SID}.log"
```

If the driver log only reports a dead actor, inspect that worker pod's Ray
logs and Lepton termination reason. GPU and host-memory failures need
different mitigations; Ray's node-memory monitor concerns host RAM, not CUDA
memory. See [Ray OOM prevention](https://docs.ray.io/en/master/ray-core/scheduling/ray-oom-prevention.html).

| Failure evidence | Config-only diagnostic to consider | Tradeoff / caution |
|---|---|---|
| CUDA OOM on long rollout/training/log-prob batches | `policy.max_total_sequence_length=16384` | Default is 73728 even in the eight-host command. This also changes interpolated generation, vLLM, and packing limits; it changes the workload and can exclude long-input examples. |
| OOM specifically during validation | `grpo.val_batch_size=16` | Smaller batches, more validation iterations. Keep `grpo.max_val_samples=null`; the runner includes the final partial batch. |
| Host RAM pressure while code verification is active | `env.nemo_gym.code_gen.resources_servers.code_gen.num_processes=32` | Default is 1024; lower concurrency reduces verification throughput. Confirm the verifier is the memory consumer first. |
| OOM during checkpoint saving | Inspect save/serialization logs and per-node RAM | Do not disable optimizer saves casually: doing so changes what can be restored on resume. |

Apply one diagnostic override at a time after identifying the failing phase;
these are not a confirmed fix for an uninspected OOM. Training and log-prob
microbatch sizes are already `1`, and default packing token budgets are 73728.
Reducing only the global batch is not equivalent to reducing these budgets.
Do not assume `policy.generation_batch_size` caps NeMo-Gym's HTTP request
concurrency, or disable Ray's memory monitor to hide host-memory pressure.

## Config Nuances

- Keep configs aligned with the NeMo-RL image schema. Missing `grpo`,
  `loss_fn`, `policy`, `checkpointing`, or `logger` keys usually surface as
  runtime `KeyError`s.
- For Lightning, keep `grpo.max_val_samples=null`; the runner bounds validation
  from the prepared validation split and preserves `grpo.val_batch_size`.
- Size rollout, validation, and training batches for the active Ray worker
  topology. `policy.logprob_batch_size` is per worker after sharding.
- A Ray `ActorDiedError` is usually secondary; diagnose the preceding actor or
  NeMo-Gym traceback.

## Repository Layout

- Manifest: `src/nemotron/steps/rl/nemo_rl/rlvr/step.toml`
- Runner: `src/nemotron/steps/rl/nemo_rl/rlvr/step.py`
- Configs: `src/nemotron/steps/rl/nemo_rl/rlvr/config/`

## Guardrails

- Validate reward functions on sample rollouts before training.
- Keep reward outputs bounded and deterministic when possible.
- Do not mix NeMo-Gym resource-server config with the upstream generic GRPO
  data schema.
