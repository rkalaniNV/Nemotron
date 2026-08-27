# MCP-610 tiny_library pilot protocol

This is the locked collection protocol for the first real three-flow ablation. Do not fill
missing observations with estimates or reuse one run under another repetition.

## Pinned comparison contract

- Input schema: `bfcl-onboarding-ablation-input-v2`
- Experiment ID: `tiny-library-onboarding-pilot-v1`
- Domain artifact: [`bfcl-mcp-ablation-tiny-domain-brief.md`](bfcl-mcp-ablation-tiny-domain-brief.md)
- Domain artifact digest (raw file bytes): `sha256:209d2da5cd2f1e9e2f510774334499c3edb876a05e643eba43e9557e6b72fff3`
- Evaluator model: `not_run`
- Evaluation config: `sha256:cbfc55d6101a2441adf57e3bad9d76c45f4d42e97f539f8d98459e4b5267cc51`
- Held-out policy: `sha256:82c5d90c44c93c01a058b2aeb202c8320d24bfe1021f94369356edbf20adbc2a`
- Repetitions per flow: 3
- Evaluation scores: omit for all nine pilot runs

If the tiny pack changes, stop and create a new experiment ID and domain digest. Do not mix
observations across artifact revisions.

## Fixed execution sequence

Use this order to reduce simple learning/order bias:

1. `manual`, repetition 1
2. `llm_backend`, repetition 1
3. `llm_mcp`, repetition 1
4. `llm_mcp`, repetition 2
5. `manual`, repetition 2
6. `llm_backend`, repetition 2
7. `llm_backend`, repetition 3
8. `llm_mcp`, repetition 3
9. `manual`, repetition 3

Start every run from a clean workspace copied from the same pinned domain artifact. Do not carry
draft files, model conversations, validation fixes, or notes into the next run.

## Measurement rules

- `sequence` and `repetition` come from the fixed schedule above.
- `run_digest` is the SHA-256 identity of that run's immutable output bundle. Every run must have
  a different digest.
- `user_authored_fields` counts scalar leaves in structured YAML/JSON inputs whose semantic
  values the operator authored. For the manual flow, count the BFCL config plus manifest, tools,
  templates, and validation cases; exclude shared fixtures and Python source. For generated
  flows, count only operator-authored intake/config documents, never generated pack artifacts.
  Accepting, regenerating, or lightly formatting a generated value does not turn it into a
  user-authored field.
- `authoring_minutes` starts when the clean flow-specific intake opens and stops when the first
  complete canonical pack is ready for review.
- `review_minutes` starts immediately afterward and stops when the operator either approves the
  reviewed pack or records that the run failed. Keep the timer running while fixing review
  findings.
- `validation_pass_rate` is passing non-conditional checks divided by attempted non-conditional
  checks in the final fresh validation report. A skipped required check is attempted and not
  passing.
- `tool_coverage` is the fraction of declared tools having both a successful and a negative
  validation observation.
- `replay_stability` is the fraction of declared tools whose fresh deterministic replay check
  passes.
- `benchmark_rows` is the row count in the generated benchmark artifact and must be greater than
  zero.

Use a monotonic timer and record full-precision minutes; do not round during collection. Breaks
unrelated to the run must be paused and noted outside the machine-readable input.

## Flow boundaries

- `manual`: author the Oracle Pack without model-generated pack fields and without MCP discovery
  generating those fields.
- `llm_backend`: use the evidence-bound LLM authoring flow with a conventional backend, without
  MCP discovery or the MCP gateway.
- `llm_mcp`: use MCP discovery, evidence-bound drafting, gateway conformance, review, freeze, and
  publication handoff.

The operator may consult the same normative BFCL documentation in every flow. Domain-specific
notes, generated drafts, and prior-run fixes are not shared.

## Collection commands

For sequence 1, start the timer before authoring:

```bash
python -m nemotron.steps.byob.scripts.collect_bfcl_onboarding_observation begin \
  --state observations/run-01.state.json \
  --flow manual \
  --repetition 1 \
  --sequence 1
```

When the first complete pack is ready for review:

```bash
python -m nemotron.steps.byob.scripts.collect_bfcl_onboarding_observation review \
  --state observations/run-01.state.json
```

After review and fresh generation finish, stop the timer immediately:

```bash
python -m nemotron.steps.byob.scripts.collect_bfcl_onboarding_observation stop \
  --state observations/run-01.state.json
```

Then record measured metrics and digest the immutable run artifact directory. Time spent looking
up or entering these metrics is not counted:

```bash
python -m nemotron.steps.byob.scripts.collect_bfcl_onboarding_observation finish \
  --state observations/run-01.state.json \
  --output observations/run-01.observation.json \
  --run-artifact output/run-01 \
  --user-authored-fields <count> \
  --validation-pass-rate <0..1> \
  --tool-coverage <0..1> \
  --replay-stability <0..1> \
  --benchmark-rows <count>
```

Use `--excluded-authoring-minutes` or `--excluded-review-minutes` only for a recorded unrelated
break. The tool refuses negative elapsed time, duplicate outputs, empty or symlinked artifact
trees, invalid metrics, and a second finish.

## Completion

After all nine immutable run bundles exist, assemble the observations. Pass `--observation` once
for each of the nine files:

```bash
python -m nemotron.steps.byob.scripts.collect_bfcl_onboarding_observation assemble \
  --observation observations/run-01.observation.json \
  --observation observations/run-02.observation.json \
  --observation observations/run-03.observation.json \
  --observation observations/run-04.observation.json \
  --observation observations/run-05.observation.json \
  --observation observations/run-06.observation.json \
  --observation observations/run-07.observation.json \
  --observation observations/run-08.observation.json \
  --observation observations/run-09.observation.json \
  --output three_flow_observations.json \
  --experiment-id tiny-library-onboarding-pilot-v1 \
  --domain-artifact-digest sha256:209d2da5cd2f1e9e2f510774334499c3edb876a05e643eba43e9557e6b72fff3 \
  --evaluator-model not_run \
  --evaluation-config-digest sha256:cbfc55d6101a2441adf57e3bad9d76c45f4d42e97f539f8d98459e4b5267cc51 \
  --held-out-policy-digest sha256:82c5d90c44c93c01a058b2aeb202c8320d24bfe1021f94369356edbf20adbc2a
```

Then generate the deterministic report:

```bash
python -m nemotron.steps.byob.scripts.compare_bfcl_onboarding_flows \
  --input three_flow_observations.json \
  --output three_flow_ablation_report.json
```

The loader rejects missing repetitions, duplicate run digests, incomplete sequence numbers,
mixed score availability, non-finite metrics, and comparison-contract drift.
