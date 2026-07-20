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


def unigram_similarity(left: str, right: str) -> float:
    a, b = set(tokens(left)), set(tokens(right))
    return len(a & b) / len(a | b) if a and b else 0.0


def token_overlap_similarity(left: str, right: str) -> float:
    """Order-independent overlap that catches containment and reordered queries."""
    a, b = set(tokens(left)), set(tokens(right))
    return len(a & b) / min(len(a), len(b)) if a and b else 0.0


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
    anchor_ids = {item.chunk_id for item in candidate.evidence}
    needs = draft.evidence_needs
    if len(needs) < candidate.minimum_evidence_needs:
        errors.append(
            f"draft has {len(needs)} evidence need(s); archetype requires {candidate.minimum_evidence_needs}"
        )
    if candidate.evidence_scope == "single_facet" and len(needs) != 1:
        errors.append("single-facet drafts must contain exactly one evidence need")
    if candidate.evidence_scope == "conversational" and needs:
        errors.append("conversational drafts must not manufacture evidence needs")
    supported_sets: list[frozenset[str]] = []
    probes: list[str] = []
    for index, need in enumerate(needs):
        ids = set(need.supporting_chunk_ids)
        unknown = sorted(ids - anchor_ids)
        if unknown:
            errors.append(f"evidence_needs[{index}] references unknown chunk IDs: {unknown}")
        if need.supported_by_bundle and not ids:
            errors.append(f"evidence_needs[{index}] is marked supported but has no supporting chunk")
        if not need.supported_by_bundle and ids:
            errors.append(f"evidence_needs[{index}] is marked unsupported but lists supporting chunks")
        if ids:
            frozen = frozenset(ids)
            if frozen in supported_sets:
                errors.append("evidence needs reuse the same support set instead of representing distinct facets")
            supported_sets.append(frozen)
        probe = need.retrieval_probe.strip()
        if _CHUNK_ID.search(probe) or any(chunk_id in probe for chunk_id in anchor_ids):
            errors.append(f"evidence_needs[{index}].retrieval_probe leaks chunk identifiers")
        language_error = _language_error(probe, candidate.language)
        if language_error:
            errors.append(f"evidence_needs[{index}].retrieval_probe: {language_error}")
        probes.append(probe)
    for index, current in enumerate(probes):
        for previous in probes[:index]:
            similarity = max(lexical_similarity(current, previous), unigram_similarity(current, previous))
            if similarity >= cfg.evidence.max_probe_similarity:
                errors.append(
                    "evidence-need retrieval probes are too similar to demonstrate distinct facets: "
                    f"{similarity:.2f} >= {cfg.evidence.max_probe_similarity:.2f}"
                )
                break
    unsupported = [need for need in needs if not need.supported_by_bundle]
    if candidate.answerability == "answerable" and unsupported:
        errors.append("answerable draft contains an unsupported evidence need")
    if candidate.answerability == "insufficient" and not unsupported:
        errors.append("insufficient-evidence draft must contain an unsupported essential need")
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
        canonical_retrieved = retriever.query(draft.query, top_k=cfg.evidence.retrievability_top_k)
        naive_retrieved = retriever.query(draft.naive_query, top_k=cfg.evidence.retrievability_top_k)
    except Exception as exc:
        errors.append(f"retrievability check failed: {exc}")
        return errors
    canonical_ids = {chunk.chunk_id for chunk in canonical_retrieved}
    naive_ids = {chunk.chunk_id for chunk in naive_retrieved}
    canonical_matched = canonical_ids & anchor_ids
    naive_matched = naive_ids & anchor_ids
    if not canonical_matched:
        errors.append("canonical query did not retrieve any evidence anchor")
    profile = cfg.surface_form_profiles[candidate.surface_form]
    canonical_recall = len(canonical_matched) / len(anchor_ids)
    naive_recall = len(naive_matched) / len(anchor_ids)
    recall_gap = canonical_recall - naive_recall
    if recall_gap + 1e-9 < profile.minimum_anchor_recall_gap:
        errors.append(
            f"surface form `{candidate.surface_form}` improved anchor recall by {recall_gap:.2f}; "
            f"requires at least {profile.minimum_anchor_recall_gap:.2f}"
        )
    if profile.require_noncanonical_form and draft.query.casefold().strip() == draft.naive_query.casefold().strip():
        errors.append(f"surface form `{candidate.surface_form}` duplicates the canonical query")
    if profile.require_topic_overlap and unigram_similarity(draft.query, draft.naive_query) <= 0:
        errors.append("adjacent-intent naive query has no recoverable lexical connection to the canonical topic")

    for index, need in enumerate(needs):
        try:
            probe_results = retriever.query(
                need.retrieval_probe,
                top_k=cfg.evidence.retrievability_top_k,
            )
        except Exception as exc:
            errors.append(f"retrievability check for evidence need {index} failed: {exc}")
            continue
        probe_ids = {chunk.chunk_id for chunk in probe_results}
        if need.supported_by_bundle and not (set(need.supporting_chunk_ids) & probe_ids):
            errors.append(f"retrieval probe did not recover evidence for need {index}")
    return errors


def query_similarity(left: QueryDraft, right: QueryDraft) -> float:
    pairs = (
        (left.query, right.query),
        (left.naive_query, right.naive_query),
    )
    return max(
        score
        for left_text, right_text in pairs
        for score in (
            lexical_similarity(left_text, right_text),
            unigram_similarity(left_text, right_text),
            token_overlap_similarity(left_text, right_text),
        )
    )
