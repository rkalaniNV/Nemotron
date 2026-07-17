"""Bounded trainable think tokens.

Generate/validate ``reasoning_content`` as bounded, auditable think tokens rather
than unrestricted chain-of-thought. Rules enforced here:

- Token budget (default 400) counted with the real tokenizer.
- Every claim in ``claims`` cites a chunk actually returned by ``retrieve``.
- The trace records a retrieval assessment (did the last retrieval suffice?), so
  the corpus teaches the retrieve -> assess -> rewrite skill.
- No private free-form CoT beyond the bounded trace; no ungrounded conclusions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Set

from mtsdg.schemas import ReasoningContent
from mtsdg.tokens import count_tokens

MAX_REASONING_TOKENS = 400


def reasoning_to_text(rc: ReasoningContent) -> str:
    """Flatten a ReasoningContent into text (fallback for the token budget)."""
    parts: List[str] = [rc.task_understanding, rc.retrieval_assessment]
    for e in rc.evidence_selection:
        parts.append(f"{e.chunk_id} {e.purpose}")
    for c in rc.claims:
        parts.append(c.claim)
    parts += rc.answer_plan
    return "\n".join(p for p in parts if p)


@dataclass
class ReasoningValidation:
    """Outcome of validating a ``reasoning_content`` trace.

    ``errors`` are HARD (grounding + token budget) — they reject the trajectory.
    ``warnings`` are SOFT (scaffold completeness) — recorded, not rejecting.
    """

    ok: bool
    token_count: int
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.ok


def validate_reasoning_content(
    rc: ReasoningContent,
    returned_chunk_ids: Iterable[str],
    *,
    max_tokens: int = MAX_REASONING_TOKENS,
) -> ReasoningValidation:
    """Validate a think-token trace.

    ``returned_chunk_ids`` are chunk IDs actually returned by ``retrieve`` so far
    — every cited chunk must be one of them.
    """
    errors: List[str] = []
    warnings: List[str] = []
    returned: Set[str] = set(returned_chunk_ids)

    text = rc.think.strip() if rc.think.strip() else reasoning_to_text(rc)
    n_tokens = count_tokens(text)
    if n_tokens > max_tokens:
        errors.append(f"reasoning_content is {n_tokens} tokens (> {max_tokens}).")

    # Grounding gate. HARD = fabricated citation (claims a chunk retrieve never
    # returned) — that is the dangerous failure. SOFT = a claim with no citation at
    # all (an occasional uncited statement should not reject a whole 22-turn
    # trajectory; the trainable field is the natural-language `think`).
    for i, c in enumerate(rc.claims):
        if not c.supporting_chunk_ids:
            warnings.append(f"claims[{i}] has no supporting_chunk_ids.")
            continue
        unknown = [x for x in c.supporting_chunk_ids if x not in returned]
        if unknown:
            errors.append(f"claims[{i}] cites chunk(s) not returned by retrieve: {unknown}.")
    for e in rc.evidence_selection:
        if e.chunk_id not in returned:
            errors.append(
                f"evidence_selection references chunk `{e.chunk_id}` not returned by retrieve."
            )

    # SOFT: scaffold completeness.
    if not rc.think.strip():
        warnings.append("reasoning `think` trace is empty.")
    if not rc.task_understanding.strip():
        warnings.append("task_understanding is empty.")
    if not rc.answer_plan:
        warnings.append("answer_plan is empty.")

    return ReasoningValidation(ok=not errors, token_count=n_tokens, errors=errors, warnings=warnings)
