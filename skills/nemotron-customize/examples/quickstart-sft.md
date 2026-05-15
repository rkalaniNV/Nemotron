# Quickstart: 2-stage pack + SFT pipeline

A canonical, copy-pasteable smoke test for both humans and agents. Runs the
SFT path end-to-end on the in-tree tiny blend so you can verify the
`nemotron-customize` workflow without staging custom data.

If you only have a developer laptop and no GPUs, run [Smoke variant: dry-run
plan only](#smoke-variant-dry-run-plan-only) instead — it stops after the
plan phase and never invokes a GPU step.

## Prerequisites

```bash
# From the Nemotron repo root.
uv sync
uv run nemotron --help
uv run nemotron steps list --json | head
```

Set the credential variables you actually need; defaults are public-only:

```bash
export HF_TOKEN=<your-hf-token>            # gated tokenizers/datasets
export WANDB_API_KEY=<your-wandb-key>      # optional, only if W&B is enabled
```

## Step 1 — Compose the plan

```text
/nemotron-customize

I want a 2-stage SFT pipeline using the in-tree tiny SFT blend as the data
source. Use Megatron-Bridge for SFT, packed-sequence Parquet, Nano3 30B as
the base model. Output to ./quickstart-sft/. 8x H100, 1 node. W&B off.
```

The agent will:

1. Read `data_prep/sft_packing/step.toml`, `sft/megatron_bridge/step.toml`,
   `types.toml`, `hardware.md`, and the matching patterns.
2. Emit a 2-stage plan (`01_pack` → `02_sft`) and wait for approval.
3. After approval, write only the YAML configs the existing repo can run
   directly.

## Step 2 — Expected generated tree

```
quickstart-sft/
└── configs/
    ├── 01_pack.yaml        # data_prep/sft_packing — adapts the tiny blend
    └── 02_sft.yaml         # sft/megatron_bridge — Nano3 30B, TP=4, PP=1
```

Each YAML mirrors the keys read by the existing `step.py` files. No new
Python, no scaffolds, no Slurm wrappers.

## Step 3 — Run the generated configs

```bash
# Pack the tiny blend → packed_parquet shards.
uv run nemotron steps run data_prep/sft_packing \
    -c quickstart-sft/configs/01_pack.yaml

# SFT on the packed shards → checkpoint_megatron.
uv run nemotron steps run sft/megatron_bridge \
    -c quickstart-sft/configs/02_sft.yaml
```

If you do not have 8 H100s on this host, swap `02_sft.yaml` for the in-tree
`tiny.yaml` to drive a 16-GPU Lepton smoke through the env profile system:

```bash
export NEMOTRON_ENV_FILE=env.lepton.toml
uv run nemotron steps run data_prep/sft_packing \
    -c tiny --batch lepton_prep_sft_packing
uv run nemotron steps run sft/megatron_bridge \
    -c tiny --batch lepton_sft_megatron_bridge
```

## Step 4 — Verify

```bash
ls quickstart-sft/configs                                    # 01_pack.yaml, 02_sft.yaml
ls -R ${SFT_OUTPUT_DIR:-./output/data_prep/sft_packing}/     # packed Parquet shards
ls -R ${MBRIDGE_OUTPUT_DIR:-./output/sft/megatron_bridge}/   # Megatron checkpoint
```

Artifact wiring contract:

| Stage | Consumes | Produces |
|---|---|---|
| `01_pack` (`data_prep/sft_packing`) | `training_jsonl` (in-tree tiny blend) | `packed_parquet` |
| `02_sft` (`sft/megatron_bridge`) | `packed_parquet` from stage 1 | `checkpoint_megatron` |

If `02_sft` cannot find `packed_train_data_path`, stage 1 did not finish or
`SFT_PACKED_DIR` does not point at the produced shards. Re-run stage 1
before debugging the trainer.

---

## Smoke variant: dry-run plan only

For agents on developer laptops without GPUs:

```bash
uv sync
uv run nemotron --help
uv run nemotron steps list --json
uv run nemotron steps show data_prep/sft_packing
uv run nemotron steps show sft/megatron_bridge
```

Then dispatch the same `/nemotron-customize` request, **stop after Plan**,
and inspect the proposed YAML keys against the printed `step.toml` of each
step. This is the cheapest way to catch artifact-wiring or parameter-name
mistakes before requesting GPU hardware.

For BYOB- or translation-shaped pipelines, follow the laptop smoke
recipes in
[`src/nemotron/steps/byob/references/cpu-smoke-path.md`](../../../src/nemotron/steps/byob/references/cpu-smoke-path.md)
and
[`use-case-examples/translate-corpus/`](../../../use-case-examples/translate-corpus/).
