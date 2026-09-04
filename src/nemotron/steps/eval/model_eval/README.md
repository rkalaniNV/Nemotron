# Model Evaluation (NeMo Evaluator)

Use `eval/model_eval` to evaluate a model on benchmark suites. The step has two
modes:

- **`launcher`** (default, `-c default` / `-c tiny_chat`) — hands the run to
  NeMo Evaluator Launcher and its executors, following the same pattern as
  `nano3 eval` and `super3 eval`: compile YAML, strip Nemotron-only run
  metadata, then call `run_eval()`. Deploys a Megatron checkpoint or targets a
  hosted endpoint.
- **`direct`** (`-c direct`) — runs the harness **inside this step's own job**
  against an endpoint you host, so scheduling is done by whichever runspec
  profile you pass to `--batch`. Use it for a backend the launcher has no
  executor for, or for a harness image whose tasks the launcher registry does
  not route (its task registry is whatever the image ships).

Use this README for the workflow and task-selection rules; use `step.toml` for
the full strategies, errors, and parameter list.

## Inputs and outputs

- Consume a hosted endpoint config or `checkpoint_megatron`.
- Produce `eval_results`: benchmark metrics, artifacts, logs, and optional W&B export.

## CLI And Overlay Knobs

Use `config/tiny_chat.yaml` for hosted endpoint smoke tests and `config/default.yaml`
for Megatron checkpoint evaluation. In a project overlay, developers usually
change:

- `evaluation.tasks`: task IDs from `nemo-evaluator-launcher ls tasks`.
- `target.api_endpoint.url`, `target.api_endpoint.model_id`, and
  `target.api_endpoint.type` for hosted endpoints.
- `target.api_endpoint.api_key_name`: environment variable name, not the key.
- `deployment.checkpoint_path`: concrete Megatron `iter_*` path.
- `evaluation.nemo_evaluator_config.config.params.limit_samples`: smoke before full runs.
- `dry_run`: preview compiled launcher config.

Example shape:

```bash
uv run nemotron steps run eval/model_eval \
  -c tiny_chat \
  target.api_endpoint.model_id=<served-model-id> \
  evaluation.nemo_evaluator_config.config.params.limit_samples=1
```

Related patterns:

- Reference [src/nemotron/steps/patterns/eval-before-and-after-training.md](../../patterns/eval-before-and-after-training.md)

## Repository Layout

- Manifest: [step.toml](step.toml)
- Runner: [step.py](step.py)
- Runtime helpers: [runtime.py](runtime.py)
- Configs:
  - `config/direct.yaml` — **direct mode**: run the harness in this step's own
    job against an endpoint you host. Backend is whichever `--batch` profile
    submits the step.
  - `config/default.yaml` — launcher mode, Megatron checkpoint deployment
  - `config/tiny_chat.yaml` — launcher mode, hosted chat smoke test

## Install

```bash
uv sync --extra evaluator
```

That installs NeMo Evaluator Launcher. It does **not** install any benchmark
harness — `nemo-evaluator ls` in a fresh checkout reports zero harness packages.
Harnesses ship as container images, which both modes below use.

## Quickstart — hosted endpoint smoke test (~5 minutes)

The shortest path that produces a real number. Needs only an OpenAI-compatible
endpoint and its token.

```bash
export NEMO_EVALUATOR_MODEL_ID=<exact-model-id>          # must equal what the endpoint serves
export NEMO_EVALUATOR_MODEL_URL=https://<host>/v1/chat/completions
export NEMO_EVALUATOR_ENDPOINT_TYPE=chat
export NEMO_EVALUATOR_API_KEY_NAME=NVIDIA_API_KEY        # the NAME of the variable
export NVIDIA_API_KEY=<your-token>                       # the token itself

# 1. see the compiled config without running anything
uv run nemotron steps run eval/model_eval -c tiny_chat dry_run=true

# 2. one sample, end to end
uv run nemotron steps run eval/model_eval -c tiny_chat \
  evaluation.nemo_evaluator_config.config.params.limit_samples=1

# 3. the full task -- tiny_chat pins limit_samples: 1, so lift it explicitly
uv run nemotron steps run eval/model_eval -c tiny_chat \
  evaluation.nemo_evaluator_config.config.params.limit_samples=null
```

Results land in `output_dir`. For a Megatron checkpoint instead of a hosted
endpoint, use `-c default` and set `deployment.checkpoint_path` to a concrete
`iter_*` directory.

## Who owns the endpoint

This step evaluates a model over HTTP. Which layer *creates* that endpoint
depends on the mode and backend, and the honest current picture is:

| Backend | Launcher mode (`run.env.launcher_executor=…`) | Direct mode (`--batch lepton_eval_direct`) |
|---|---|---|
| Local | `local` — launcher can start a configured model server | runs the harness against a URL you provide |
| Slurm | `slurm` — launcher can start the server, health-check it, evaluate, clean up | no server lifecycle — supply a URL |
| Lepton | `lepton` — **experimental**, see below | no server lifecycle — supply a URL |
| Run:ai / DGX Cloud | no launcher executor exists | no server lifecycle — supply a URL |

**Direct mode never creates or tears down an endpoint.** You host the model
yourself — see *Standing the endpoint up yourself* below — and pass
`EVAL_ENDPOINT_URL`, `EVAL_MODEL_HANDLE` and the credential.

> **Launcher-managed Lepton deployment is experimental.** The launcher's Lepton
> executor accepts only specific deployment types (vLLM, SGLang, NIM, or none) —
> not the `generic` type this step's `default.yaml` uses — no complete Lepton
> launcher config ships here, created endpoints are **not** torn down after a
> successful run, and a multi-task run can create one GPU endpoint per task.
> Treat Lepton as: host the endpoint yourself, evaluate it with direct mode.

> **Two schedulers, two fields.** Launcher mode has an outer scheduler (the
> runspec executor that runs this step) and an inner one (the backend NeMo
> Evaluator Launcher submits to). They are separate config fields on purpose:
>
> - `run.env.executor` — outer; overridden by `--run` / `--batch`.
> - `run.env.launcher_executor` — inner; `local`, `slurm` or `lepton`.
>
> Pointing both at one field made "outer Slurm" silently mean "launcher submits
> again from inside the allocation", and left "outer local, launcher Slurm" —
> the normal way to use launcher mode on a cluster — inexpressible:
>
> ```bash
> # submit from the login node; the launcher does the sbatch
> uv run nemotron steps run eval/model_eval -c default \
>   run.env.launcher_executor=slurm run.env.account=<acct> run.env.partition=<part>
> ```
>
> Use direct mode when you want the outer `--batch` profile to be the only
> scheduler.

> **Auto-squash does not run for launcher mode.** Squashing Docker images to
> `.sqsh` on a remote cluster is keyed to `run.env`, and the generic runner
> strips `run.env` before invoking the step — so `_maybe_auto_squash` returns
> immediately in a submitted run, whatever the executor. On a Slurm cluster
> that cannot pull from your registry, pre-squash the deployment and evaluation
> images yourself and reference the `.sqsh` paths in the config. Direct mode is
> unaffected: the harness image is the step's own container, which the `--batch`
> profile handles like any other step's.

## Advanced — direct mode

`-c direct` runs the harness **inside this step's own job**, against a model you
host yourself. Nemotron's runspec executors do the scheduling, so the backend is
whatever `--batch` profile you point at — including Run:ai, which the launcher
has no executor for at all.

```bash
export EVAL_ENDPOINT_URL=https://<host>/v1/completions
export EVAL_MODEL_HANDLE=my-model        # MUST equal the endpoint's --served-model-name
export EVAL_RESULTS_DIR=/mnt/shared/eval-results/my-run   # must be a mount, not ephemeral disk
export ENDPOINT_TOKEN=<token>

uv run nemotron steps run eval/model_eval -c direct --batch lepton_eval_direct \
  -t adlr_arc_challenge_llama_25_shot -t hellaswag
```

Swap benchmark suites by swapping the harness image; `nemo-evaluator ls` inside
any image lists what it offers.

```bash
# discover with :latest, then pin the digest (see Reproducibility below)
export EVAL_HARNESS_IMAGE=nvcr.io/nvidia/eval-factory/sovereign@sha256:<digest>
uv run nemotron steps run eval/model_eval -c direct --batch lepton_eval_direct \
  -t milu_Hindi -t milu_English
```

Smoke with `EVAL_LIMIT_SAMPLES=5` first — one cheap job proves endpoint, token,
tokenizer and mount are wired before you spend a full split.

**The config resolves inside the pod.** Every `${oc.env:...}` it references must
be forwarded by your `--batch` profile's `env_vars`. Exporting a variable in
your local shell only affects submission — the job will not see it.



### Standing the endpoint up yourself

Direct mode needs a URL. Any OpenAI-compatible server will do; these are the two
shapes that come up most.

**Lepton.** One endpoint per checkpoint:

```bash
lep endpoint create -n my-ckpt \
  --resource-shape gpu.a100-80gb \
  --node-group <node-group-id> \
  --image vllm/vllm-openai:latest \
  --mount <path>:<storage>:<mount_path> \
  --port 8000 --min-replicas 1 --max-replicas 1 \
  --command "vllm serve <mount_path>/<hf-export> \
    --served-model-name my-ckpt --port 8000 \
    --tensor-parallel-size 1 --gpu-memory-utilization 0.94 \
    --max-model-len 8192 --trust-remote-code"

lep endpoint get -n my-ckpt        # wait for Ready; read the URL and api token
```

Then `EVAL_ENDPOINT_URL=https://my-ckpt.<your-domain>/v1/completions` and
`EVAL_MODEL_HANDLE=my-ckpt`. **`--served-model-name` and `EVAL_MODEL_HANDLE`
must match exactly** — if they differ, every request 404s and `summary.json`
reports only `failed(N)`.

Deployments take the node-group **ID**; jobs take its **name**. Set
`--min-replicas 0` when you are done rather than deleting, so the endpoint can
be scaled back up for a re-run:

```bash
lep endpoint update -n my-ckpt --min-replicas 0 --max-replicas 0
```

**Run:ai / DGX Cloud.** The launcher has no Run:ai executor at all, so this is
the only way to evaluate there. Serve with a normal inference workload and
expose it as a service:

```bash
runai submit my-ckpt --image vllm/vllm-openai:latest --gpu 1 \
  --pvc <claim>:<mount_path> --service-type nodeport --port 8000:8000 \
  --command -- vllm serve <mount_path>/<hf-export> \
    --served-model-name my-ckpt --port 8000 --trust-remote-code
```

then point `EVAL_ENDPOINT_URL` at the resulting service address and submit the
eval with `--batch <dgxcloud_profile>`.

> The Lepton block above is transcribed from an endpoint spec that has served
> evaluations in this repo. The Run:ai block is the standard `runai submit`
> shape and has **not** been run as written — check it against your cluster's
> service and PVC conventions before relying on it.

Whichever you use, smoke with `EVAL_LIMIT_SAMPLES=5` before the full split.

### Several checkpoints, several benchmarks

The common case. Direct mode runs **one job per endpoint, all tasks inside it**,
so the loop is over checkpoints, not over benchmarks:

```bash
# fixed for the whole sweep
export EVAL_HARNESS_IMAGE=nvcr.io/nvidia/eval-factory/lm-evaluation-harness@sha256:<digest>
export EVAL_ENDPOINT_TYPE=completions      # base models; `chat` for instruct
export EVAL_API_KEY_NAME=ENDPOINT_TOKEN
export ENDPOINT_TOKEN=<token>

for CKPT in run-a run-b run-c; do
  export EVAL_MODEL_HANDLE=$CKPT           # == the server's --served-model-name
  export EVAL_ENDPOINT_URL=https://$CKPT.example.com/v1/completions
  export EVAL_RESULTS_DIR=/mnt/shared/eval/$CKPT
  uv run nemotron steps run eval/model_eval -c direct --batch lepton_eval_direct \
    -t adlr_arc_challenge_llama_25_shot -t hellaswag -t mmlu_prox_completions
done
```

Each checkpoint gets its own `output_dir` with one subdirectory per task, plus
`summary.json` and `run_manifest.json`. Comparing checkpoints then means
comparing manifests: same harness image, same merged `params`, different model.

Two things this step does **not** do:

- **It does not serve your checkpoints.** Direct mode needs a URL. Stand each
  endpoint up first — see *Standing the endpoint up yourself* above — or use
  launcher mode, which can deploy a Megatron checkpoint itself.
- **It does not sweep checkpoints for you.** The loop above is the interface.

Smoke the first checkpoint with `EVAL_LIMIT_SAMPLES=5` before looping: one cheap
job proves endpoint, token, model handle, tokenizer and mount are all wired.

### The adapter proxy, and why caching is not optional

Between the harness and your endpoint sits an adapter proxy, configured under
`evaluation.nemo_evaluator_config.target.api_endpoint.adapter_config`. Direct
mode forwards this namespace whole, alongside `config.params.*`.

`use_caching: true` is on by default in `direct.yaml` and should stay on.
Without it the harness holds every request and response for a task in memory:
around 30k requests it dies client-side, on a large CPU shape, **while the
endpoint is still returning 200 to every call** — so the symptom points at the
model server when the problem is the eval job. MILU-Hindi at full split is past
that line; it completed at `limit_samples: 2000`.

Adapter keys are deliberately **not** allowlisted (unlike `config.params.*`,
where an unknown key is rejected). The interceptor set is open-ended and version
dependent, and an unknown adapter key is inert rather than a silent change to
how the model is scored. Leaving `output_dir: null` gives each task its own
`<output_dir>/<task>/adapter`, so two tasks in one run cannot collide.

`run_manifest.json` records the merged `adapter_config` per task next to
`params`, so a cached run and an uncached one are distinguishable after the
fact. The adapter's own `endpoint_type` defaults to `chat`; direct mode fills it
in from `target.api_endpoint.type` unless you set it explicitly, so it cannot
silently disagree with the endpoint being called.

> Caching removes the memory *ceiling*; it has not been shown to make an
> unbounded full split finish. The largest verified run is MILU-Hindi at
> `limit_samples: 2000` with caching on (peak 4.55 GB). `limit_samples` is a
> deterministic first-N, so the subset is identical across checkpoints and
> scores stay comparable within an ablation — but it is not a full-split score,
> and should not be reported as one.

### Discovering a task's real defaults

Task defaults differ substantially — few-shot counts, endpoint type, whether
log-probabilities are required. Inspect the exact image you intend to pin,
because the registry is per-image:

```bash
nemo-evaluator-launcher ls tasks --from <pinned-image>
nemo-evaluator-launcher ls task <task-name> --from <pinned-image> --json
# direct mode / harness-only tasks:
docker run --rm <pinned-image> nemo-evaluator ls
```

Map what you find onto `evaluation.nemo_evaluator_config`. A useful split:

- **global** — `request_timeout`, `max_retries`, `parallelism`, `limit_samples`
- **per task** — `task` (subset/language), `temperature`, `top_p`,
  `max_new_tokens`, and anything harness-specific under `extra:`
  (for example `extra.num_fewshot`, `extra.tokenizer`)

Direct mode **rejects** an unrecognised top-level param rather than dropping it,
so a typo fails loudly instead of silently changing generation settings.

### `run_manifest.json`

Every direct-mode run writes `run_manifest.json` into `output_dir` **before the
first task starts**, so a preempted job still records what it was doing:

```json
{
  "schema_version": 1,
  "generated_at": "2026-09-04T11:02:31+00:00",
  "mode": "direct",
  "model": {
    "id": "my-model",
    "endpoint_url": "https://host/v1/completions",
    "endpoint_type": "completions",
    "api_key_name": "ENDPOINT_TOKEN"
  },
  "harness": {
    "image": "nvcr.io/nvidia/eval-factory/lm-evaluation-harness@sha256:...",
    "image_pinned_by_digest": true,
    "packages": {"nemo-evaluator": "..."}
  },
  "code": {"nemotron_version": "...", "git_sha": "...", "git_dirty": false},
  "tasks": {"hellaswag": {"output_dir": "...", "params": {...}, "command": [...]}},
  "output_dir": "...",
  "summary_path": ".../summary.json"
}
```

`params` is the **merged** global + per-task set, i.e. what the task actually
ran with. `harness` says what computed the score; `code` says what decided its
inputs — its three fields are always present, `null` when unavailable, so "not a
git checkout" is distinguishable from provenance quietly dropped.

`image_pinned_by_digest` requires a real digest — `sha256:` followed by exactly
64 hex characters. If it is `false`, the run is not reproducible; re-run against
a digest-pinned image before citing the number.

The manifest is safe to attach to a report. The endpoint URL is stored with
userinfo and query string stripped, only the credential's *name* is recorded,
and credential-valued parameters (`api_key`, `*_token`, `password`, …) are
replaced with `***` in both `params` and `command`. The command echoed to the
job log is scrubbed identically. `tokenizer` is deliberately not treated as a
credential, and `api_key_name` is kept because it is a variable name.

Per-task outcomes stay in `summary.json`; a dry run writes
`run_manifest.dry-run.json` so it cannot overwrite a real one.

**Direct mode does not log metrics to W&B**, by design: it writes files and does
not require credentials. Nothing in the eval path requests W&B, so a `wandb`
login on the submitting host is neither read nor forwarded. To publish results,
upload `run_manifest.json` and `summary.json` yourself, or use launcher mode
with `execution.auto_export` and `export.wandb` enabled.

### Reproducibility: pin the harness by digest

A harness image is the source of truth for task definitions, prompt formatting
and dependencies, so a mutable tag makes a run unreproducible. Two tags of the
same image family have been observed to expose different task sets while
reporting the same internal version, so the tag alone is not a reliable record.

Use `:latest` only to discover what exists, then record the digest and pin it:

```bash
# discover
docker run --rm <image>:latest nemo-evaluator ls
# resolve the immutable digest
docker buildx imagetools inspect <image>:latest --format '{{.Manifest.Digest}}'
# pin it for the run you intend to cite
export EVAL_HARNESS_IMAGE=<image>@sha256:<digest>
```

Report the digest alongside any number you publish.

### The `--batch` profile

A direct-mode profile ships for each backend, so there is nothing to author:

| Backend | Profile |
|---|---|
| Lepton | `lepton_eval_direct` |
| Slurm | `slurm_eval_direct` |
| Run:ai / DGX Cloud | `dgxcloud_eval_direct` |

They come from `steps/env/env_toml/config/{lepton,slurm,dgxcloud}.yaml`; run
`nemotron steps run env/env_toml -c <backend>` to regenerate your `env.toml`
after pulling. All three are **CPU-only** — the harness drives an endpoint you
host yourself, so the GPUs are wherever the model is served.

If you write your own, direct mode resolves its config **inside the job**, so
the profile must forward every `${oc.env:...}` the config reads — exporting them
in your shell only affects submission:

```toml
[my_direct_profile]
executor = "<your backend>"                 # local, slurm, lepton, dgxcloud
# No container_image here: `direct.yaml` owns it (see below).
# The harness image is not a Nemotron image and ships none of its deps.
pip_extras = ["typer", "rich", "pydantic-settings", "omegaconf", "tomli", "wandb"]
env_vars = { EVAL_ENDPOINT_URL = "${oc.env:EVAL_ENDPOINT_URL,''}", EVAL_MODEL_HANDLE = "${oc.env:EVAL_MODEL_HANDLE,''}", EVAL_RESULTS_DIR = "${oc.env:EVAL_RESULTS_DIR,''}", EVAL_ENDPOINT_TYPE = "${oc.env:EVAL_ENDPOINT_TYPE,completions}", EVAL_API_KEY_NAME = "${oc.env:EVAL_API_KEY_NAME,ENDPOINT_TOKEN}", EVAL_LIMIT_SAMPLES = "${oc.env:EVAL_LIMIT_SAMPLES,null}", EVAL_TOKENIZER = "${oc.env:EVAL_TOKENIZER,null}", HF_HOME = "${oc.env:HF_HOME,/tmp/hf}" }
```

On Lepton, a CPU-only profile must also override the base's
`shared_memory_size: 65536` (say, `8192`): no `cpu.*` shape can satisfy 64 GB of
`/dev/shm`, and Lepton rejects the submission with HTTP 400 before a job is
created. Neither `env.toml` generation nor `--dry-run` catches that — neither
talks to the scheduler.

Note what is **not** in `env_vars`: the token. Values placed there are stored
verbatim in the submitted job spec, so anyone who can read the job can read the
token. On Lepton, store it once as a platform secret and reference it by name:

```bash
lep secret create -n eval-endpoint-token -v <token>
```

```toml
# ENV_VAR_NAME = "<lepton-secret-name>"; the value is injected by the platform
# and never appears in the profile, the job spec, or `lep job get` output.
secret_vars = { ENDPOINT_TOKEN = "eval-endpoint-token" }
```

`EVAL_API_KEY_NAME` still names the *variable* (`ENDPOINT_TOKEN`) — only where
its value comes from changes. On backends without a secret store, keep the token
in the submitting shell and forward it through `env_vars`, and treat the job
spec as sensitive.

**`EVAL_HARNESS_IMAGE` is the image selector, not the profile.** A profile's
`container_image` has no effect here: `build_job_config()` re-applies
YAML-owned resource keys (`container_image`, `nodes`, `gpus_per_node`, …) after
merging `env.toml`, so a config that sets `container_image` — as `direct.yaml`
does — always wins. Pick the image with `EVAL_HARNESS_IMAGE`, or per-run on the
command line:

```bash
uv run nemotron steps run eval/model_eval -c direct --batch lepton_eval_direct \
  run.env.container_image=<image>@sha256:<digest>
```

CLI overrides land in the YAML before the profile is merged, so those do win.

`EVAL_HARNESS_IMAGE` is also absent from the profile's `env_vars` on purpose:
`direct.yaml` forwards it into the job from `run.env.container_image`, so
`run_manifest.json` records the image the executor was actually given. Without
that forwarding the manifest would record the *default tag* — the config is
serialised unresolved, so an `${oc.env:...}` image reference is re-evaluated
inside the container, where a variable you exported on the submitting host does
not exist.

Add whatever mounts your backend needs so that `EVAL_RESULTS_DIR` is durable
storage — results written to a container's ephemeral disk are lost when the job
is reaped.

## Troubleshooting

Symptom-to-cause tables for executor quirks, harness-container differences,
checkpoint tokenizers and coverage gaps live in
[TROUBLESHOOTING.md](TROUBLESHOOTING.md).

## Guardrails

- Don't compare scores across different endpoint types or different
  generation settings.
- Don't add checkpoint conversion "just in case"; pick the artifact format and configure the matching deployment path.
- Pick the mode deliberately: launcher mode for its 421-task registry and
  managed deployments; direct mode when you host the endpoint yourself, need a
  harness image the registry doesn't route to, or need a backend the launcher
  has no executor for (Run:ai).
- Don't put raw API keys in YAML or command output.
- Inspect a handful of generations before trusting aggregate metrics.
