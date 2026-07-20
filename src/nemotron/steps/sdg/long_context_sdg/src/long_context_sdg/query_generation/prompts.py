"""Prompts for query drafting and independent quality judging."""

from __future__ import annotations

import json

from .schemas import (
    PersonaProjection,
    QueryCandidate,
    QueryDraft,
    QuerySynthesisJudgment,
)

JUDGE_DIMENSIONS = (
    "topic_fit",
    "persona_realism",
    "language_quality",
    "answerability",
    "retrieval_quality",
    "evidence_structure",
    "non_leakage",
    "surface_form_fidelity",
    "rewrite_value",
)


SURFACE_FORM_GUIDANCE = {
    "well_formed": (
        "Write a natural, complete first utterance. It may be less formal than the canonical query, "
        "but it should already express the information need clearly."
    ),
    "underspecified": (
        "Omit one or more material details such as the exact entity, time range, jurisdiction, comparison "
        "basis, or success criterion. Keep the utterance plausible and understandable, so clarification or "
        "later reformulation can reveal the full need."
    ),
    "retrieval_rewrite": (
        "Express the right underlying need with vague shorthand, pronouns, colloquial wording, or a weak "
        "search formulation. The canonical query must be a materially better retrieval query."
    ),
    "noisy_language": (
        "Use plausible human spelling, grammar, code-switching, or word-order imperfections for the target "
        "language. For English this may look like imperfect/non-native English. Never use gibberish, mock an "
        "accent, or encode a demographic stereotype."
    ),
    "keyword_fragment": (
        "Use a terse, plausible search-like fragment rather than a polished sentence. Preserve enough signal "
        "for a helpful conversation to begin."
    ),
    "overbroad": (
        "Ask about a broader area containing the canonical need, without naming all of its constraints. The "
        "utterance should naturally narrow over the conversation."
    ),
    "adjacent_intent": (
        "Start from a closely neighboring concern or goal that shares the same topic but sits just outside the "
        "canonical intent. The relationship must be plausible and recoverable through conversation; do not "
        "switch to an unrelated topic."
    ),
}


def draft_messages(
    candidate: QueryCandidate,
    persona: PersonaProjection,
    previous_errors: list[str] | None = None,
) -> list[dict[str, str]]:
    evidence = [
        {
            "chunk_id": chunk.chunk_id,
            "title": chunk.title,
            "source": chunk.source,
            "content": chunk.content,
        }
        for chunk in candidate.evidence
    ]
    prompt = {
        "taxonomy": {
            "id": candidate.taxonomy_id,
            "label": candidate.taxonomy_label,
            "description": candidate.taxonomy_description,
            "required_terms": candidate.taxonomy_required_terms,
        },
        "archetype": candidate.archetype,
        "evidence_scope": candidate.evidence_scope,
        "minimum_evidence_needs": candidate.minimum_evidence_needs,
        "answerability": candidate.answerability,
        "persona_mode": candidate.persona_mode,
        "surface_form": candidate.surface_form,
        "surface_form_guidance": SURFACE_FORM_GUIDANCE[candidate.surface_form],
        "target_language": candidate.language,
        "persona": persona.model_dump(),
        "evidence": evidence,
        "output_schema": QueryDraft.model_json_schema(),
    }
    if previous_errors:
        prompt["previous_attempt_errors"] = previous_errors
    return [
        {
            "role": "system",
            "content": (
                "Generate one realistic first-user query seed for a long research conversation. "
                "Return JSON only. `query` is a self-contained canonical information need used for "
                "retrievability checks; `naive_query` is the natural first user utterance. Write both "
                "in the target language. Use the persona only for plausible motivation, expertise, and "
                "style; never force unrelated demographic attributes into the question and never expose "
                "a name or source UUID. Do not answer the question, quote long source spans, mention the "
                "evidence bundle, or emit chunk IDs/citations. Follow surface_form_guidance exactly for the "
                "visible naive_query while keeping it plausible for the persona. The canonical query must remain "
                "well-formed and materially more retrievable whenever the selected profile calls for a weak, "
                "underspecified, noisy, broad, or adjacent first utterance. For clarification, the naive query "
                "may omit one material detail while the canonical query remains self-contained. For insufficient "
                "evidence, ask a relevant question whose essential answer is absent from the bundle. "
                "For single_facet, create one focused evidence need. For multi_facet, create the configured "
                "minimum number of substantively independent evidence needs that require different supplied "
                "chunks or sources; do not split one fact into paraphrases. For conversational scope, create an "
                "initial utterance where resolving context or ambiguity is a natural first step; never prescribe "
                "whether the downstream assistant should call a tool. Populate "
                "the internal evidence_needs audit field: describe each distinct need, add a concise hidden "
                "retrieval_probe for that facet, map it only to supplied chunk IDs, and mark an intentionally "
                "missing need unsupported for insufficient_evidence. Probes must be independently useful searches, "
                "not copies of the canonical query or paraphrases of each other. Never "
                "put those IDs in query or naive_query. Derive concise role, expertise, and style labels for the "
                "conversation simulator."
                " If previous-attempt errors are supplied, correct every one without discussing them."
            ),
        },
        {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
    ]


def judge_messages(
    candidate: QueryCandidate,
    persona: PersonaProjection,
    draft: QueryDraft,
) -> list[dict[str, str]]:
    payload = {
        "target": {
            "taxonomy_id": candidate.taxonomy_id,
            "taxonomy_label": candidate.taxonomy_label,
            "archetype": candidate.archetype,
            "evidence_scope": candidate.evidence_scope,
            "minimum_evidence_needs": candidate.minimum_evidence_needs,
            "answerability": candidate.answerability,
            "persona_mode": candidate.persona_mode,
            "surface_form": candidate.surface_form,
            "language": candidate.language,
        },
        "persona": persona.model_dump(),
        "draft": draft.model_dump(),
        "evidence": [chunk.model_dump() for chunk in candidate.evidence],
        "dimensions": JUDGE_DIMENSIONS,
        "output_schema": QuerySynthesisJudgment.model_json_schema(),
    }
    return [
        {
            "role": "system",
            "content": (
                "You are an independent synthetic-query quality judge. Return JSON only. Score every "
                "listed dimension from 1 to 5. Reject contrived persona motivation, stereotypes, source "
                "or answer leakage, wrong script/language, unsupported answerability labels, trivial "
                "factoids, and questions that do not support a substantive research conversation. The "
                "evidence needs must be genuinely independent for multi_facet tasks and must map accurately to "
                "the supplied chunks; reject paraphrased duplicate needs. The "
                "selected surface form must look like realistic user language, preserve a recoverable relationship "
                "to the canonical intent, and create the configured retrieval improvement without becoming "
                "gibberish. Plausible imperfect English is acceptable when requested, but stereotypes are not. The "
                "reported answerability must describe the supplied evidence bundle. Set rating to success "
                "only when the example is suitable for training."
            ),
        },
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
