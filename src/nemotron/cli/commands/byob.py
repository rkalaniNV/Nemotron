# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""BYOB benchmark command."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from nemotron.steps.byob.scripts.runtime import (
    STAGE_CHOICES,
    list_family_names,
    load_dispatch_config,
    resolve_dispatch_value,
    run_byob,
)


def byob(
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            help="Path to the BYOB YAML config.",
        ),
    ] = None,
    family: Annotated[
        str | None,
        typer.Option(
            "--family",
            help="Benchmark family to run.",
        ),
    ] = None,
    stage: Annotated[
        str | None,
        typer.Option(
            "--stage",
            help="Pipeline stage to run: prepare, generate, translate, or all.",
        ),
    ] = None,
    skip_until: Annotated[
        str | None,
        typer.Option(
            "--skip-until",
            help="Resume from a family-specific stage enum name, such as JUDGEMENT or QUALITY_METRICS.",
        ),
    ] = None,
    list_families: Annotated[
        bool,
        typer.Option(
            "--list-families",
            help="List registered BYOB benchmark families.",
        ),
    ] = False,
) -> None:
    """Run BYOB benchmark generation or translation."""

    if list_families:
        for registered_family in list_family_names():
            typer.echo(registered_family)
        return

    if config is None:
        typer.echo("Error: --config is required unless --list-families is set", err=True)
        raise typer.Exit(1)

    yaml_dict = load_dispatch_config(config)
    stage = resolve_dispatch_value(stage, yaml_dict, "stage")
    family = resolve_dispatch_value(family, yaml_dict, "family", default="mcq")
    skip_until = resolve_dispatch_value(skip_until, yaml_dict, "skip_until")

    if stage is None:
        typer.echo("Error: --stage is required unless config contains `stage`", err=True)
        raise typer.Exit(1)

    if stage not in STAGE_CHOICES:
        valid = ", ".join(STAGE_CHOICES)
        typer.echo(f"Error: --stage must be one of: {valid}", err=True)
        raise typer.Exit(1)

    output_path = run_byob(
        config=config,
        stage=stage,
        family=family,
        skip_until=skip_until,
    )
    if output_path is not None:
        typer.echo(output_path)
