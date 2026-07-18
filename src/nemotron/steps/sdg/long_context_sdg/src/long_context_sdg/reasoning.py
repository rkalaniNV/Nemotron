"""Bound and provenance-check assistant reasoning."""

from __future__ import annotations

from collections.abc import Iterable

from .schemas import ReasoningContent, ValidationReport
from .tokens import count_tokens


def validate_reasoning(
    reasoning: ReasoningContent,
    known_chunk_ids: Iterable[str],
    *,
    max_tokens: int,
) -> ValidationReport:
    errors = []
    warnings = []
    n_tokens = count_tokens(reasoning.think)
    if n_tokens > max_tokens:
        errors.append(f"reasoning has {n_tokens} tokens; maximum is {max_tokens}")
    known = set(known_chunk_ids)
    unknown = sorted(set(reasoning.cited_chunk_ids) - known)
    if unknown:
        errors.append(f"reasoning cites chunks that were not retrieved: {unknown}")
    if reasoning.think and not reasoning.cited_chunk_ids and known:
        warnings.append("reasoning contains no evidence citations")
    if not reasoning.task_understanding:
        warnings.append("reasoning task_understanding is empty")
    if not reasoning.answer_plan:
        warnings.append("reasoning answer_plan is empty")
    return ValidationReport(ok=not errors, errors=errors, warnings=warnings)
