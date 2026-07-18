"""Deterministic leakage, language, retrievability, and deduplication checks."""

from __future__ import annotations

import re

from ..retrieval import RetrieverClient
from .config import QueryGenerationConfig
from .schemas import QueryCandidate, QueryDraft

_CHUNK_ID = re.compile(r"\b(?:h-[0-9a-f]{12,}|chunk-[\w-]+)\b", re.IGNORECASE)
_WORD = re.compile(r"\w+", re.UNICODE)
_DEVANAGARI = re.compile(r"[\u0900-\u097f]")


def tokens(text: str) -> list[str]:
    return [token.casefold() for token in _WORD.findall(text)]


def _shingles(text: str, width: int = 3) -> set[tuple[str, ...]]:
    values = tokens(text)
    if len(values) < width:
        return {tuple(values)} if values else set()
    return {tuple(values[index : index + width]) for index in range(len(values) - width + 1)}


def lexical_similarity(left: str, right: str) -> float:
    a, b = _shingles(left), _shingles(right)
    return len(a & b) / len(a | b) if a and b else 0.0


def _has_verbatim_span(query: str, evidence: str, width: int) -> bool:
    query_tokens = tokens(query)
    evidence_tokens = tokens(evidence)
    if len(query_tokens) < width:
        return False
    evidence_spans = {
        tuple(evidence_tokens[index : index + width]) for index in range(len(evidence_tokens) - width + 1)
    }
    return any(
        tuple(query_tokens[index : index + width]) in evidence_spans for index in range(len(query_tokens) - width + 1)
    )


def _language_error(text: str, language: str) -> str | None:
    letters = [character for character in text if character.isalpha()]
    if not letters:
        return "query contains no letters"
    devanagari_ratio = len(_DEVANAGARI.findall(text)) / len(letters)
    normalized = language.casefold()
    if "deva" in normalized and devanagari_ratio < 0.25:
        return f"expected Devanagari text, observed ratio {devanagari_ratio:.2f}"
    if normalized.startswith("en") and devanagari_ratio > 0.05:
        return f"expected English text, observed Devanagari ratio {devanagari_ratio:.2f}"
    return None


def validate_draft(
    draft: QueryDraft,
    candidate: QueryCandidate,
    cfg: QueryGenerationConfig,
    retriever: RetrieverClient,
) -> list[str]:
    errors: list[str] = []
    for name, text in (("query", draft.query), ("naive_query", draft.naive_query)):
        length = len(text.strip())
        if length < cfg.min_query_chars or length > cfg.max_query_chars:
            errors.append(f"{name} length {length} outside {cfg.min_query_chars}..{cfg.max_query_chars}")
        language_error = _language_error(text, candidate.language)
        if language_error:
            errors.append(f"{name}: {language_error}")
        leaked = sorted(chunk_id for chunk_id in {item.chunk_id for item in candidate.evidence} if chunk_id in text)
        if leaked or _CHUNK_ID.search(text):
            errors.append(f"{name} leaks chunk identifiers")
        for chunk in candidate.evidence:
            similarity = lexical_similarity(text, chunk.content)
            if similarity > cfg.evidence.max_lexical_overlap:
                errors.append(
                    f"{name} lexical overlap {similarity:.2f} exceeds {cfg.evidence.max_lexical_overlap:.2f}"
                )
                break
            if _has_verbatim_span(text, chunk.content, cfg.evidence.max_verbatim_tokens):
                errors.append(f"{name} copies at least {cfg.evidence.max_verbatim_tokens} evidence tokens")
                break

    try:
        retrieved = retriever.query(draft.query, top_k=cfg.evidence.retrievability_top_k)
    except Exception as exc:
        errors.append(f"retrievability check failed: {exc}")
        return errors
    anchors = {chunk.chunk_id: chunk for chunk in candidate.evidence}
    matched = [chunk for chunk in retrieved if chunk.chunk_id in anchors]
    if not matched:
        errors.append("canonical query did not retrieve any evidence anchor")
    if candidate.archetype == "comparison":
        matched_sources = {anchors[chunk.chunk_id].source for chunk in matched}
        matched_sources.discard("")
        if len(matched_sources) < 2:
            errors.append("comparison query did not recover two evidence sources")
    return errors


def query_similarity(left: QueryDraft, right: QueryDraft) -> float:
    return max(
        lexical_similarity(left.query, right.query),
        lexical_similarity(left.naive_query, right.naive_query),
    )
