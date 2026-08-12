"""Structured model responses used by BFCL surface-only stages."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ReferenceProfileResult(BaseModel):
    style_hints: list[str] = Field(
        description="Concise style rules observed in the supplied reference conversations"
    )
    avoid: list[str] = Field(
        default_factory=list,
        description="Surface-writing patterns absent from or discouraged by the references",
    )


class ParaphraseVariant(BaseModel):
    user_turns: list[str] = Field(
        description="Ordered rewrites, one for each canonical user turn"
    )


class ParaphraseResult(BaseModel):
    variants: list[ParaphraseVariant] = Field(
        description="Ordered conversation variants; no explanations or metadata"
    )
