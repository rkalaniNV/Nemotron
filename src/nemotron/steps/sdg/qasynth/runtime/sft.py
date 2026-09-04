# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Answer-seed preparation, voting, quality gates, and SFT export."""

from __future__ import annotations

import hashlib
import random
import re
from collections import Counter
from typing import Any

from nemotron.steps.sdg.qasynth.runtime.answers import INSTRUCTIONS

DEVANAGARI = re.compile(r"[ऀ-ॿ]")
LATIN = re.compile(r"[A-Za-z]")
ANSWER_LINE = re.compile(r"(?:उत्तर|Answer)\s*[:：]", re.IGNORECASE)


def prepare_answer_seed(records: list[dict[str, Any]], _seed: int) -> list[dict[str, Any]]:
    """Render source-compatible ids and option shuffles for answer teachers."""
    prepared: list[dict[str, Any]] = []
    for record in records:
        choices = list(record["choices"])
        query_id = hashlib.blake2b(
            (
                record["language"]
                + "||"
                + record["question"]
                + "||"
                + "|".join(choices)
            ).encode(),
            digest_size=12,
        ).hexdigest()
        rng = random.Random(int(query_id[:8], 16))
        permutation = list(range(len(choices)))
        rng.shuffle(permutation)
        metadata = record.get("metadata") or {}
        prepared.append(
            {
                "query_id": query_id,
                "language": record["language"],
                "question": record["question"],
                "choices": [choices[index] for index in permutation],
                "original_choices": choices,
                "shuffle_permutation": permutation,
                "gen_model": metadata.get("source_model"),
                "difficulty": metadata.get("difficulty"),
                "topic": metadata.get("topic"),
                "facet": metadata.get("facet"),
                "region": metadata.get("region"),
                "persona_uuid": metadata.get("persona_uuid"),
            }
        )
    return prepared


def vote(letters: list[str | None], agreement: str) -> str | None:
    valid = [letter for letter in letters if letter]
    if agreement == "unanimous":
        return valid[0] if len(valid) == len(letters) and len(set(valid)) == 1 else None
    if agreement == "majority":
        if len(valid) < 2:
            return None
        letter, count = Counter(valid).most_common(1)[0]
        return letter if count >= 2 else None
    raise ValueError(f"Unsupported agreement policy: {agreement!r}")


def devanagari_fraction(value: str) -> float:
    devanagari = len(DEVANAGARI.findall(value))
    latin = len(LATIN.findall(value))
    return devanagari / (devanagari + latin) if devanagari + latin else 0.0


def build_sft_records(
    answers_by_model: dict[str, list[dict[str, Any]]],
    *,
    response_model: str,
    language: str,
    agreement: str,
    min_devanagari_fraction: float,
    max_devanagari_fraction: float,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    indexed = {
        model: {str(record["query_id"]): record for record in records}
        for model, records in answers_by_model.items()
    }
    if response_model not in indexed:
        raise ValueError(f"Unknown response model {response_model!r}")
    output: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    for query_id, teacher in indexed[response_model].items():
        letters = [indexed[model].get(query_id, {}).get("parsed_letter") for model in indexed]
        winner = vote(letters, agreement)
        if winner is None:
            skipped["no_agreement"] += 1
            continue
        if teacher.get("parsed_letter") != winner:
            skipped["teacher_dissents"] += 1
            continue
        if teacher.get("finish_reason") in {"length", "max_tokens"}:
            skipped["truncated"] += 1
            continue
        answer = str(teacher.get("answer") or "").strip()
        reasoning = str(teacher.get("reasoning") or "").strip()
        if not answer:
            skipped["empty_answer"] += 1
            continue
        if not reasoning:
            skipped["empty_reasoning"] += 1
            continue
        fraction = devanagari_fraction(answer)
        if not min_devanagari_fraction <= fraction <= max_devanagari_fraction:
            skipped["language_impurity"] += 1
            continue
        prompt = f"{INSTRUCTIONS[language]}\n\n{teacher['question']}\n\n" + "\n".join(
            f"{chr(65 + index)}) {choice}" for index, choice in enumerate(teacher["choices"])
        )
        output.append(
            {
                "messages": [
                    {"role": "system", "content": ""},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "reasoning_content": reasoning, "content": answer},
                ],
                "metadata": {
                    "query_id": query_id,
                    "language": language,
                    "response_model": response_model,
                    "agreement": agreement,
                    "voted_letter": winner,
                    "n_valid_votes": sum(letter is not None for letter in letters),
                    "difficulty": teacher.get("difficulty"),
                    "topic": teacher.get("topic"),
                    "facet": teacher.get("facet"),
                    "region": teacher.get("region"),
                    "gen_model": teacher.get("gen_model"),
                    "persona_uuid": teacher.get("persona_uuid"),
                },
            }
        )
    return output, {"kept": len(output), **dict(skipped)}


def _final_answer(content: str, language: str, voted: str) -> str:
    for line in reversed([line.strip() for line in content.splitlines() if line.strip()]):
        if ANSWER_LINE.search(line):
            return line
    return f"{'उत्तर' if language == 'hindi' else 'Answer'}: {voted}"


def sample_aligned_datasets(
    datasets: dict[str, dict[str, list[dict[str, Any]]]],
    *,
    sample_per_language: int,
    seed: int,
    reasoning_off_fraction: float,
    answer_variant: str,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    if not 0 <= reasoning_off_fraction <= 1:
        raise ValueError("reasoning_off_fraction must be between zero and one")
    indexed = {
        teacher: {
            language: {str(row["metadata"]["query_id"]): row for row in rows}
            for language, rows in by_language.items()
        }
        for teacher, by_language in datasets.items()
    }
    languages = sorted(set.intersection(*(set(by_language) for by_language in indexed.values())))
    rng = random.Random(seed)
    selected: dict[str, list[str]] = {}
    for language in languages:
        common = sorted(
            set.intersection(*(set(by_language[language]) for by_language in indexed.values()))
        )
        if len(common) < sample_per_language:
            raise ValueError(
                f"{language}: shared teacher intersection {len(common)} is smaller than requested "
                f"{sample_per_language}"
            )
        selected[language] = rng.sample(common, sample_per_language)

    result: dict[str, list[dict[str, Any]]] = {}
    cutoff = int(reasoning_off_fraction * 100)
    for teacher, by_language in indexed.items():
        rows: list[dict[str, Any]] = []
        for language, query_ids in selected.items():
            for query_id in query_ids:
                source = by_language[language][query_id]
                user = next(message["content"] for message in source["messages"] if message["role"] == "user")
                assistant_source = next(message for message in source["messages"] if message["role"] == "assistant")
                is_off = int(query_id[:8], 16) % 100 < cutoff
                content = assistant_source["content"]
                if answer_variant == "stripped":
                    content = _final_answer(content, language, source["metadata"]["voted_letter"])
                elif answer_variant != "full":
                    raise ValueError(f"Unsupported answer variant: {answer_variant!r}")
                assistant = {"role": "assistant", "content": content}
                if not is_off:
                    assistant["reasoning_content"] = assistant_source["reasoning_content"]
                metadata = {**source["metadata"], "reasoning_mode": "off" if is_off else "on"}
                rows.append(
                    {
                        "messages": [{"role": "system", "content": ""}, {"role": "user", "content": user}, assistant],
                        "metadata": metadata,
                    }
                )
        random.Random(seed).shuffle(rows)
        result[teacher] = rows
    return result, {"selected_by_language": {language: len(ids) for language, ids in selected.items()}}
