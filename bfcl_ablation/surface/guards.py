"""The mechanical checks a generated surface must pass before it reaches the pipeline.

These deliberately duplicate what the production render stage enforces
(`check_surface_guards`, and the paraphrase stage's novel-literal rule) rather than
inventing new ones. Running them early is what turns a bad variant into a rejection
statistic instead of a dropped task, and running exactly the production rules is what
keeps the rejection count honest: a variant accepted here is one the pipeline will
also accept, so the funnel stays flat and the arm measures wording rather than loss.

Every check here is syntactic. None of them can tell whether the sentence still asks
for the same thing; that gap is the subject of `intent_check`.
"""

from __future__ import annotations

import re
from collections import Counter

from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.paraphrase import _LITERAL
from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.render import (
    TOOL_NAME_RULE,
    _mentions_name,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.templating import placeholder_names

__all__ = ["TOOL_NAME_RULE", "placeholders", "mechanical_rejection"]


def placeholders(text: str) -> list[str]:
    return placeholder_names(text)


def _literals(text: str) -> Counter:
    """Count identifier-shaped tokens, ignoring the placeholder names themselves."""
    stripped = re.sub(r"\{[A-Za-z_][A-Za-z0-9_]*\}", " ", text)
    return Counter(_LITERAL.findall(stripped))


def mechanical_rejection(
    text: str,
    *,
    canonical: str,
    forbidden_tools: list[str],
    forbidden_phrases: list[str],
) -> str | None:
    """Return why `text` is unusable as a surface for `canonical`, or None."""
    if not text.strip():
        return "empty"

    wanted = Counter(placeholder_names(canonical))
    got = Counter(placeholder_names(text))
    if got != wanted:
        missing = wanted - got
        extra = got - wanted
        if missing and extra:
            return "placeholder_renamed"
        return "placeholder_dropped" if missing else "placeholder_added"

    novel = _literals(text) - _literals(canonical)
    if novel:
        return "novel_literal"

    lowered = text.lower()
    if any(_mentions_name(lowered, name.lower()) for name in forbidden_tools):
        return "tool_name_leak"
    if any(phrase.lower() in lowered for phrase in forbidden_phrases):
        return "forbidden_phrase"
    return None
