# nemotron.cli

Entry point for the `nemotron` command-line interface.

## Overview

This package provides the CLI commands for Nemotron training recipes. The CLI is built on [Typer](https://typer.tiangolo.com/) and uses [`nemo_runspec`](../../nemo_runspec/README.md) for config loading, execution helpers, and command registration. Domain-specific artifacts and tracking come from [`nemotron.kit`](../kit/README.md).

Each command file contains **visible execution logic** -- you can read one file to understand exactly how a job is submitted. See [Design Philosophy](../../../docs/architecture/design-philosophy.md) for why.

## Entry Point

The `nemotron` command is registered as a console script in `pyproject.toml`:

```toml
[project.scripts]
nemotron = "nemotron.cli.bin.nemotron:main"
```

## Command Structure

```
nemotron
├── byob                     # Bring-your-own benchmark generation and translation
├── nano3                    # Nano3 training recipe
│   ├── pretrain             # Stage 0: Pretraining
│   ├── sft                  # Stage 1: Supervised fine-tuning
│   ├── rl                   # Stage 2: Reinforcement learning
│   ├── data
│   │   ├── prep
│   │   │   ├── pretrain     # Prepare pretrain data
│   │   │   ├── sft          # Prepare SFT data
│   │   │   └── rl           # Prepare RL data
│   │   └── import
│   │       ├── pretrain     # Import pretrain data artifact
│   │       ├── sft          # Import SFT data artifact
│   │       └── rl           # Import RL data artifact
│   └── model
│       ├── eval             # Evaluate model
│       └── import
│           ├── pretrain     # Import pretrain checkpoint
│           ├── sft          # Import SFT checkpoint
│           └── rl           # Import RL checkpoint
└── kit                      # Kit utilities
    └── squash               # Squash container images
```

## Module Structure

```
src/nemotron/cli/
├── __init__.py              # Package marker
├── bin/
│   └── nemotron.py          # Main entry point (typer app)
├── kit/
│   ├── app.py               # Kit utility commands
│   └── squash.py            # Container squashing
└── commands/
    └── nano3/               # Nano3 recipe CLI
        ├── _typer_group.py  # Command registration (RecipeTyper)
        ├── pretrain.py      # Pretrain command + execution logic
        ├── sft.py           # SFT command + execution logic
        ├── rl.py            # RL command + execution logic (Ray)
        ├── data/
        │   ├── _typer_group.py  # Data group
        │   ├── prep/            # Data prep commands
        │   └── import_/         # Data import commands
        └── model/
            ├── _typer_group.py  # Model group
            ├── eval.py          # Model evaluation
            └── import_/         # Model import commands
```

## Global Options

All commands support these global options (managed by `nemo_runspec.cli_context.GlobalContext`):

| Option | Short | Description |
|--------|-------|-------------|
| `--config` | `-c` | Config name or path |
| `--run` | `-r` | Attached execution via NeMo-Run |
| `--batch` | `-b` | Detached execution via NeMo-Run |
| `--dry-run` | `-d` | Preview config without execution |
| `--stage` | | Stage script to remote for debugging |
| `key=value` | | Dotlist overrides (any position) |

## Usage Examples

```bash
# Local execution with config
uv run nemotron nano3 pretrain -c tiny

# Submit to cluster (attached)
uv run nemotron nano3 pretrain -c tiny --run MY-CLUSTER

# Submit to cluster (detached)
uv run nemotron nano3 pretrain -c tiny --batch MY-CLUSTER

# Preview without execution
uv run nemotron nano3 pretrain -c tiny --dry-run

# Override config values
uv run nemotron nano3 pretrain -c tiny train.train_iters=5000

# BYOB benchmark generation
uv run nemotron byob --family mcq --stage prepare --config src/nemotron/steps/byob/mcq/config/default.yaml
uv run nemotron byob --family mcq --stage generate --config src/nemotron/steps/byob/mcq/config/default.yaml
uv run nemotron byob --list-families

# Data preparation
uv run nemotron nano3 data prep pretrain --run MY-CLUSTER
uv run nemotron nano3 data prep sft --run MY-CLUSTER
uv run nemotron nano3 data prep rl --run MY-CLUSTER
```

## Adding New Commands

To add a new recipe command:

1. Create the training script with a `[tool.runspec]` block (see [nemo_runspec](../../nemo_runspec/README.md))
2. Create a command module with visible execution logic
3. Register with `RecipeTyper.add_recipe_command()`

See [docs/nemotron/cli.md](../../../docs/nemotron/cli.md) for a step-by-step tutorial.

## Full Documentation

See [docs/nemotron/cli.md](../../../docs/nemotron/cli.md) for complete CLI framework documentation including:

- Command pattern with visible execution
- Configuration pipeline
- Execution modes
- Recipe building tutorial

See [docs/nemo_runspec/nemo-run.md](../../../docs/nemo_runspec/nemo-run.md) for execution profile configuration.
See [src/nemo_runspec/README.md](../../nemo_runspec/README.md) for the CLI toolkit.
