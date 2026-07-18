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
    "non_leakage",
)


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
        "answerability": candidate.answerability,
        "persona_mode": candidate.persona_mode,
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
                "evidence bundle, or emit chunk IDs/citations. For clarification, the naive query may omit "
                "one material detail while the canonical query remains self-contained. For insufficient "
                "evidence, ask a relevant question whose essential answer is absent from the bundle. "
                "Derive concise role, expertise, and style labels for the conversation simulator."
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
            "answerability": candidate.answerability,
            "persona_mode": candidate.persona_mode,
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
                "reported answerability must describe the supplied evidence bundle. Set rating to success "
                "only when the example is suitable for training."
            ),
        },
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
