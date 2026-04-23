"""CLI command for the translation step skill."""

from __future__ import annotations

import typer

from nemo_runspec.recipe_config import parse_recipe_config
from nemo_runspec.recipe_typer import RecipeMeta
from nemotron.steps.translation.scripts.run import main as run_translation

META = RecipeMeta(
    name="steps/translation",
    script_path="src/nemotron/steps/translation/scripts/run.py",
    config_dir="src/nemotron/steps/translation/assets",
    default_config="default",
)


def translation(ctx: typer.Context) -> None:
    """Run the translation step skill with the checked-in default config surface."""
    cfg = parse_recipe_config(ctx)

    if cfg.mode != "local":
        typer.echo(
            "Error: nemotron steps translation currently supports local execution only.",
            err=True,
        )
        raise typer.Exit(1)

    argv: list[str] = []
    if cfg.config:
        argv.extend(["--config", cfg.config])
    argv.extend(cfg.dotlist)
    argv.extend(cfg.passthrough)

    run_translation(argv=argv)
