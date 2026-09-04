# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused offline tests for the persona QASynth step and reusable plugin."""

from __future__ import annotations

import json
from importlib.metadata import entry_points
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from nemotron.steps.sdg.plugins.qasynth.parsing import parse_question
from nemotron.steps.sdg.qasynth.runtime.answers import parse_answer_letter
from nemotron.steps.sdg.qasynth.runtime.lexical import deduplicate, strip_latin_gloss
from nemotron.steps.sdg.qasynth.runtime.pipeline import QASynthPipeline, validate_config
from nemotron.steps.sdg.qasynth.runtime.semantic import deduplicate_embeddings, greedy_keep_indices
from nemotron.steps.sdg.qasynth.runtime.sft import (
    build_sft_records,
    prepare_answer_seed,
    sample_aligned_datasets,
    vote,
)

from .._step_helpers import assert_step_static, step_dir

STEP = step_dir(__file__, "sdg", "qasynth")


def _conversation(question: str, choices: list[str]) -> dict:
    return {
        "conversation": json.dumps(
            {
                "messages": [],
                "metadata": {
                    "difficulty": "hard",
                    "facet": "geography",
                    "parsed_question": {"question": question, "choices": choices, "success": True},
                },
            }
        )
    }


def _answer(query_id: str, model: str, letter: str = "A") -> dict:
    return {
        "query_id": query_id,
        "question": "Which river is longest?",
        "choices": ["Ganga", "Yamuna", "Kaveri", "Godavari"],
        "answer_model": model,
        "answer": f"The evidence supports Ganga.\nAnswer: {letter}",
        "reasoning": "Compare the known river lengths.",
        "parsed_letter": letter,
        "finish_reason": "stop",
        "difficulty": "hard",
        "topic": "geography",
        "facet": "geography",
    }


def test_step_static() -> None:
    assert_step_static(
        STEP,
        expected_name="steps/sdg/qasynth",
        expected_launch="python",
        expected_default_config="default",
    )


def test_qasynth_entry_point_is_installed() -> None:
    pytest.importorskip("data_designer")
    matches = {point.name: point for point in entry_points(group="data_designer.plugins")}
    assert "qasynth-mcq" in matches
    plugin = matches["qasynth-mcq"].load()
    assert plugin.name == "qasynth-mcq"


def test_parser_requires_exactly_four_distinct_options() -> None:
    valid = "<question>Which river?</question><options>A) Ganga\nB) Yamuna\nC) Kaveri\nD) Godavari</options>"
    assert parse_question(valid)["success"] is True
    assert parse_question(valid.replace("D) Godavari", "C) Godavari"))["success"] is False
    assert parse_question(valid.replace("D) Godavari", "D) Ganga"))["success"] is False


@pytest.mark.parametrize("new_shape", [True, False])
def test_model_facade_response_compatibility(new_shape: bool) -> None:
    pytest.importorskip("data_designer")
    from nemotron.steps.sdg.plugins.qasynth.llm import completion_text

    message = SimpleNamespace(content="authored", role="assistant")
    choice = SimpleNamespace(message=message)
    response = SimpleNamespace(choices=[choice], raw={"choices": [choice]})
    if not new_shape:
        response.choices = []

    class Facade:
        def completion(self, messages, **kwargs):
            assert len(messages) == 2
            assert kwargs["allow_multiple_choices"] is False
            return response

    assert completion_text(Facade(), system_prompt="system", user_prompt="user") == "authored"


def test_lexical_filters_and_deduplicates() -> None:
    choices = ["गंगा", "यमुना", "कावेरी", "गोदावरी"]
    records = [
        _conversation("भारत की सबसे लंबी नदी कौन सी है?", choices),
        _conversation("भारत की सबसे लंबी नदी कौन सी है?", choices),
        _conversation("Which river is longest?", ["Ganga", "Yamuna", "Kaveri", "Godavari"]),
        _conversation("यह mostly English question with one Hindi token?", choices),
    ]
    output, stats = deduplicate(
        records,
        language="hindi",
        source_model="oss",
        permutations=16,
        bands=4,
    )
    assert len(output) == 1
    assert stats["drop_exact"] == 1
    assert stats["drop_wrong_language"] == 2
    assert strip_latin_gloss("गेंदा (Marigold)") == "गेंदा"


def test_semantic_greedy_does_not_chain() -> None:
    # A~B and B~C, while A and C are not duplicates. Greedy with this seed keeps
    # two representatives rather than collapsing the transitive component.
    pairs = [(0, 1), (1, 2)]
    assert len(greedy_keep_indices(pairs, 3, seed=0)) == 2
    records = [{"question": str(index)} for index in range(3)]
    embeddings = np.asarray([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]])
    output, stats = deduplicate_embeddings(
        records,
        embeddings,
        threshold=0.75,
        method="greedy",
        seed=0,
    )
    assert len(output) == 2
    assert stats["accepted"] == 2


def test_answer_seed_shuffle_is_stable() -> None:
    record = {
        "query_id": "a" * 64,
        "question": "Question",
        "choices": ["A", "B", "C", "D"],
        "language": "english",
        "metadata": {"source_model": "oss"},
    }
    prepared = prepare_answer_seed([record], 42)
    assert prepared == prepare_answer_seed([record], 99)
    assert prepared[0]["query_id"] == "7e455300a746e992c8b068df"
    assert prepared[0]["shuffle_permutation"] == [3, 1, 2, 0]
    assert sorted(prepared[0]["choices"]) == record["choices"]


def test_answer_parser_and_vote() -> None:
    assert parse_answer_letter("Reasoning\n**Answer: $C$**") == "C"
    assert parse_answer_letter("उत्तर: (B)") == "B"
    assert vote(["A", "A", "A"], "unanimous") == "A"
    assert vote(["A", "A", "B"], "majority") == "A"
    assert vote(["A", "A", "B"], "unanimous") is None


def test_sft_quality_gates_and_aligned_sampling() -> None:
    query_id = "f" * 64
    answers = {model: [_answer(query_id, model)] for model in ("qwen", "oss", "gemma")}
    rows, stats = build_sft_records(
        answers,
        response_model="oss",
        language="english",
        agreement="unanimous",
        min_devanagari_fraction=0.0,
        max_devanagari_fraction=0.15,
    )
    assert stats["kept"] == 1
    datasets = {teacher: {"english": rows} for teacher in answers}
    sampled, summary = sample_aligned_datasets(
        datasets,
        sample_per_language=1,
        seed=42,
        reasoning_off_fraction=1.0,
        answer_variant="stripped",
    )
    assert summary["selected_by_language"] == {"english": 1}
    assert all("reasoning_content" not in data[0]["messages"][-1] for data in sampled.values())
    assert all(data[0]["messages"][-1]["content"] == "Answer: A" for data in sampled.values())


def test_shipped_configs_validate() -> None:
    for path in sorted((STEP / "config").glob("*.yaml")):
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        validate_config(config)


def test_cli_stage_list_string_is_normalized(tmp_path) -> None:
    config = yaml.safe_load((STEP / "config" / "tiny.yaml").read_text(encoding="utf-8"))
    config["run"].update(output_root=str(tmp_path), stages="[answers,build_sft,sample]")
    pipeline = QASynthPipeline(config)
    assert pipeline.config["run"]["stages"] == ["answers", "build_sft", "sample"]


def test_stage_selection_does_not_change_experiment_identity(tmp_path) -> None:
    base = yaml.safe_load((STEP / "config" / "tiny.yaml").read_text(encoding="utf-8"))
    base["run"].update(output_root=str(tmp_path), experiment_name="resume-test", stages=["questions"])
    QASynthPipeline(base)._prepare_experiment()

    resumed = yaml.safe_load((STEP / "config" / "tiny.yaml").read_text(encoding="utf-8"))
    resumed["run"].update(
        output_root=str(tmp_path),
        experiment_name="resume-test",
        stages="[answers,build_sft]",
        resume=False,
    )
    QASynthPipeline(resumed)._prepare_experiment()

    resumed["question_generation"]["num_records"] += 1
    with pytest.raises(ValueError, match="different config"):
        QASynthPipeline(resumed)._prepare_experiment()
