# RL Prep

Use `data_prep/rl_prep` before NeMo-RL when prompt or preference data needs HF resolution, local materialization, or split sharding.

Use this README for workflow and pitfalls; use `step.toml` for the exact artifact, parameter, strategy, and error manifest before editing configs or code.

## Inputs And Outputs

- Consume `training_jsonl` through an RL data blend.
- Produce sharded `training_jsonl` ready for `rl/nemo_rl/dpo`, `rl/nemo_rl/rlvr`, or `rl/nemo_rl/rlhf`.
- Smoke with `nemotron steps run data_prep/rl_prep -c tiny`.

## CLI And Overlay Knobs

Start from `config/tiny.yaml` for wiring and `config/default.yaml` for the
production-shaped example. In a project overlay, developers usually change:

- `blend_path`: source prompt/preference blend.
- `num_shards_per_split`: size shards for dataset size and filesystem behavior.
- `resolve_hf_placeholders`: keep `true` when the training cluster cannot reach
  the Hub.
- `compression`: choose only if downstream readers support it.
- `max_rows`: useful for representative smoke runs.

Example shape:

```bash
uv run nemotron steps run data_prep/rl_prep \
  -c "<project>/config/rl_prep.yaml" \
  "blend_path=<project>/data/rl_blend.json" \
  resolve_hf_placeholders=true
```

Related patterns:

- Check `src/nemotron/steps/patterns/prep-data-is-tokenizer-locked.md` before changing RL data layout.
- Check `src/nemotron/steps/patterns/rl-validate-rewards-before-scale.md` before scaling RL jobs from prepared data.

## Run It

Smoke first to validate wiring, imports, data access, and output paths:

```bash
uv run nemotron steps run data_prep/rl_prep -c tiny --dry-run
```

Then run the real job from a project overlay:

```bash
uv run nemotron steps run data_prep/rl_prep \
  -c "<project>/config/data_prep_rl_prep.yaml"
```

## Nemotron 3.5 Lightning

Use `-c lightning35` before `rl/nemo_rl/rlvr -c lightning35`. This selects
the Steps-owned [released JSONL backend](released_blend.py), not the generic
Xenna sharding path. See the [Lightning Lepton runbook](../../rl/nemo_rl/rlvr/README.md#nemotron-35-lightning-on-lepton)
for environment setup and the complete prep-to-training sequence.

The backend downloads `rlvr.jsonl` from
`nvidia/Nemotron-RL-Lightning-Training-Blend`, restores DAPO/Skywork
`_hf_question_placeholder` questions, and removes only top-level
`_ng_task_index` / `_ng_rollout_index` collection IDs. It filters to the
configured `allowed_agent_names` before sampling and holding out validation
rows. Keep that allowlist aligned with the training preset's
`env.nemo_gym.config_paths`; the full blend needs additional reward services.

Relevant options are `max_rows` (or `sample`), `val_holdout`,
`allowed_agent_names`, and `force`. Both splits must be nonempty:
when set, `max_rows` must exceed `val_holdout` (default `1000`), which must be
positive, and enough eligible rows must exist. The generic `compression`,
`num_shards_per_split`, and `resolve_hf_placeholders` options do not control this backend. A small
`max_rows` limits the selected output, not the initial download/restoration
of the full blend when agent filtering is enabled.

Outputs are atomically published train/validation JSONL plus
`${RL_PREP_OUTPUT_DIR}/manifest.json`, whose `train` and `val` keys contain
absolute file paths. The training preset reads those keys; do not hard-code
the hashed `runs/` subdirectory. All Ray nodes must see the same mount paths.
Rerunning prep reuses valid cached outputs and refreshes stale transform
versions or truncated files; `force=true` also forces the source download.

## Repository Layout

- Manifest: `src/nemotron/steps/data_prep/rl_prep/step.toml`
- Runner: `src/nemotron/steps/data_prep/rl_prep/step.py`
- Configs: `src/nemotron/steps/data_prep/rl_prep/config/default.yaml`, `src/nemotron/steps/data_prep/rl_prep/config/tiny.yaml`
- Lightning config and blend: `config/lightning35.yaml`, `data/blend_lightning35.json`
- Sample blend: `src/nemotron/steps/data_prep/rl_prep/data/blend_tiny.json`

## Guardrails

- Validate output JSONL records from every split before launching RL.
- Preserve split names expected by the RL config.
- Keep DPO preference ordering and RLVR verifier fields explicit.
