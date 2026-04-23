"""Steps Typer group."""

from __future__ import annotations

from nemo_runspec.recipe_typer import RecipeTyper
from nemotron.cli.commands.steps.translation import META as TRANSLATION_META, translation

steps_app = RecipeTyper(
    name="steps",
    help="Agentic workflow step skills",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

steps_app.add_recipe_command(translation, meta=TRANSLATION_META)
