# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared step-id → StepInfo resolution helpers."""
from __future__ import annotations

import difflib

import typer

from nemotron.steps.index import StepInfo, discover_steps

# Legacy step ids kept resolvable for one release after the layout normalisation
# in commit "steps: normalise byob and translate step layout".
#
# Drop these once external skills, docs, and notebooks have migrated.
_LEGACY_ID_ALIASES: dict[str, str] = {
    "byob": "byob/mcq",
    "translate/translation": "translate/curator",
}


def resolve_step(step_id: str) -> StepInfo:
    """Find a step by id, with a helpful error if missing.

    Accepts the canonical id (``peft/automodel``), the directory tail
    (``automodel``) when unambiguous, or a legacy alias from
    ``_LEGACY_ID_ALIASES``.
    """
    steps = discover_steps()
    by_id = {s.id: s for s in steps}

    if step_id in by_id:
        return by_id[step_id]

    alias_target = _LEGACY_ID_ALIASES.get(step_id)
    if alias_target and alias_target in by_id:
        typer.echo(
            f"Note: step id {step_id!r} is deprecated; use {alias_target!r} instead.",
            err=True,
        )
        return by_id[alias_target]

    # Allow short form when unambiguous.
    tail_matches = [s for s in steps if s.path.name == step_id]
    if len(tail_matches) == 1:
        return tail_matches[0]

    typer.echo(f"Unknown step id: {step_id}", err=True)
    suggestions = difflib.get_close_matches(step_id, sorted(by_id), n=3, cutoff=0.5)
    if suggestions:
        typer.echo(f"Did you mean: {', '.join(suggestions)}?", err=True)
    typer.echo("Run `nemotron steps list` to see all available steps.", err=True)
    raise typer.Exit(1)
