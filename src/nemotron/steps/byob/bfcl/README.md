# BYOB BFCL

The `bfcl` benchmark family builds function-calling benchmark artifacts from an
executable oracle pack. Unlike the MCQ family, BFCL does not generate questions
with a model: pack templates define the conversation, while the backend and
assertions establish the expected tool behavior.

## Quick Start

Install the BYOB dependencies:

```bash
uv sync --extra byob
```

Run the bundled tiny reference pack:

```bash
nemotron steps run byob/bfcl \
  -c src/nemotron/steps/byob/bfcl/config/tiny.yaml \
  stage=all \
  family=bfcl
```

For a larger example covering every supported conversation policy, replace
`tiny.yaml` with `banking_vn.yaml`.

Use `stage=prepare` to validate a pack without generating benchmark rows:

```bash
python -m nemotron.steps.byob.scripts.validate_oracle_pack --config <CONFIG>
```

## Pipeline

BFCL supports three stage values:

- `prepare`: normalize and validate the oracle pack.
- `generate`: require a gold-eligible pack, generate tasks, replay them, and
  publish artifacts.
- `all`: run `prepare` followed by `generate`.

Generation runs:

```text
expand -> state_machine -> render -> expected_trace
       -> schema_validation -> executable_replay -> final_output
```

BFCL does not currently support `translate` or `skip_until`.

## Oracle Pack

A runnable pack uses these files:

| File | Purpose |
| --- | --- |
| `manifest.yaml` | Identifies the pack and declares languages, paths, prompts, primary keys, and confirmation behavior. |
| `tools.json` | Defines the model-facing function schemas and pack-local mutation or confirmation metadata. |
| `backend.py` | Implements the local executable oracle. Use this or `endpoint_config.yaml`, never both. |
| `endpoint_config.yaml` | Connects to a BFCL Oracle HTTP v1 service over HTTPS and pins its identity/content digest. Use this or `backend.py`, never both. |
| `fixtures.json` | Supplies deterministic records and initial backend state for generated tasks. |
| `task_templates.yaml` | Declares intents, slot sources, conversation policies, milestones, and success assertions. |
| `assertions.py` | Checks the final backend state and executed trace to decide whether a replay actually succeeded. |
| `validation_cases.yaml` | Provides positive and negative probes for backend/schema alignment, determinism, errors, and confirmation behavior. |

### Direct HTTPS endpoint

An endpoint-backed pack declares `paths.endpoint: endpoint_config.yaml` in its
manifest, or sets `oracle_pack.endpoint_config_path` in the run config. The
endpoint must implement BFCL Oracle HTTP v1:

- `GET /v1/metadata`
- `GET /v1/tools`
- `POST /v1/sessions`
- `POST /v1/sessions/{session_id}/calls`
- `GET /v1/sessions/{session_id}/state`
- `DELETE /v1/sessions/{session_id}`

Session creation receives the frozen clock, seed, task id, timeout, and fixtures.
It returns a unique `session_id` plus the oracle identity. The endpoint identity
and `content_digest` must match `endpoint_config.yaml` during validation, replay,
and final publication.

Only HTTPS is accepted. Bearer tokens and custom secret headers are referenced by
environment-variable name; their values are never stored in the pack, report, or
manifest. See
[`../references/bfcl-endpoint-config.example.yaml`](../references/bfcl-endpoint-config.example.yaml)
for a complete configuration example.

Pack code must live under an `oracle_runtime.allowed_roots` entry. Gold
eligibility requires `oracle_runtime.worker: process`; thread mode is available
only for debugging.

Start from the bundled packs under `../data/`:

- `tiny_oracle_pack`: smallest end-to-end example.
- `banking_vn_oracle_pack`: domain-sized example with all supported turn
  policies.

## Outputs

Artifacts are written to `output_dir/expt_name/`:

- `benchmark_raw.parquet`: replay- and schema-valid rows before publication
  surface filtering.
- `benchmark.parquet`: published benchmark rows.
- `run_manifest.json`: lineage, fingerprints, stage counts, and artifact hashes.
- `stage_cache/`: normalized inputs and one table per generation stage, keyed by
  `task_id`.

## Configuration

Start from `config/default.yaml` for a new pack. The main settings are:

- `oracle_pack.manifest_path`
- `oracle_runtime.clock`, timeouts, `worker`, and `allowed_roots`
- `task_generation.tasks_per_category`
- `surface_generation.language`
- `lineage.policy`

For the complete pack contract, validation rules, turn policies, and schema
requirements, see
[`../references/bfcl-oracle-pack.md`](../references/bfcl-oracle-pack.md).
