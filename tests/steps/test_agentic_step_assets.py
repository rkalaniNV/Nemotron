# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for the agent-facing skill assets under steps/."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


REPO_ROOT = Path(__file__).resolve().parents[2]
STEPS_ROOT = REPO_ROOT / "src" / "nemotron" / "steps"


def _load_frontmatter(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    match = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    assert match, f"Missing YAML frontmatter in {path}"
    return yaml.safe_load(match.group(1))


def _assert_skill_frontmatter(skill_dir: Path) -> None:
    frontmatter = _load_frontmatter(skill_dir / "SKILL.md")
    assert frontmatter["name"] == skill_dir.name
    assert re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", frontmatter["name"])
    assert frontmatter["description"]
    assert len(frontmatter["description"]) <= 1024
    assert frontmatter["when_to_use"]
    assert len(frontmatter["description"] + frontmatter["when_to_use"]) <= 1536
    if "compatibility" in frontmatter:
        assert len(frontmatter["compatibility"]) <= 500


def test_steps_root_files_exist() -> None:
    assert (STEPS_ROOT / "README.md").exists()
    assert (STEPS_ROOT / "SKILL_CHECKLIST.md").exists()
    assert (STEPS_ROOT / "types.toml").exists()


def test_translation_step_assets_exist() -> None:
    expected = [
        STEPS_ROOT / "translation" / "SKILL.md",
        STEPS_ROOT / "translation" / "scripts" / "run.py",
        STEPS_ROOT / "translation" / "scripts" / "runtime.py",
        STEPS_ROOT / "translation" / "references" / "STEP.md",
        STEPS_ROOT / "translation" / "references" / "guide.md",
        STEPS_ROOT / "translation" / "assets" / "default.yaml",
        STEPS_ROOT / "translation" / "patterns" / "index.yaml",
        STEPS_ROOT / "translation" / "eval" / "golden_cases.yaml",
    ]
    for path in expected:
        assert path.exists(), f"Missing translation asset: {path}"


def test_byob_step_assets_exist() -> None:
    expected = [
        STEPS_ROOT / "byob" / "SKILL.md",
        STEPS_ROOT / "byob" / "adapter.py",
        STEPS_ROOT / "byob" / "references" / "STEP.md",
        STEPS_ROOT / "byob" / "references" / "guide.md",
        STEPS_ROOT / "byob" / "patterns" / "index.yaml",
        STEPS_ROOT / "byob" / "eval" / "golden_cases.yaml",
        STEPS_ROOT / "byob" / "eval" / "skill_cases.yaml",
    ]
    for path in expected:
        assert path.exists(), f"Missing BYOB asset: {path}"


def test_skill_frontmatter_is_present_and_valid() -> None:
    _assert_skill_frontmatter(STEPS_ROOT / "translation")
    _assert_skill_frontmatter(STEPS_ROOT / "byob")


def test_skill_docs_stay_short_and_have_key_sections() -> None:
    for skill_name in ("translation", "byob"):
        skill_path = STEPS_ROOT / skill_name / "SKILL.md"
        text = skill_path.read_text(encoding="utf-8")
        assert len(text.splitlines()) < 500
        assert "## Gotchas" in text
        assert "## Validate" in text
        assert "## Load More Only If Needed" in text


def test_step_frontmatter_references_known_types() -> None:
    type_graph = tomllib.loads((STEPS_ROOT / "types.toml").read_text(encoding="utf-8"))
    known_types = set(type_graph["types"])

    for rel_path in ("translation/references/STEP.md", "byob/references/STEP.md"):
        frontmatter = _load_frontmatter(STEPS_ROOT / rel_path)
        consumed = {entry["type"] for entry in frontmatter["consumes"]}
        produced = {entry["type"] for entry in frontmatter["produces"]}
        assert consumed <= known_types
        assert produced <= known_types


def test_skill_pattern_indexes_point_to_real_pattern_files() -> None:
    for skill_name in ("translation", "byob"):
        patterns_dir = STEPS_ROOT / skill_name / "patterns"
        index_data = yaml.safe_load((patterns_dir / "index.yaml").read_text(encoding="utf-8"))
        for pattern in index_data["patterns"]:
            pattern_path = patterns_dir / f"{pattern['id']}.md"
            assert pattern_path.exists(), f"Missing pattern file: {pattern_path}"


def test_step_modules_import() -> None:
    from nemotron.steps.byob import flatten_mcq_records, format_mcq_for_metrics, restore_mcq_records
    from nemotron.steps.translation import translate_data

    assert callable(translate_data)
    assert callable(flatten_mcq_records)
    assert callable(restore_mcq_records)
    assert callable(format_mcq_for_metrics)


def test_byob_adapter_round_trip() -> None:
    from nemotron.steps.byob import flatten_mcq_records, restore_mcq_records

    source_records = [
        {
            "id": "mcq-1",
            "question": "What is grouped-query attention?",
            "options": {"A": "A decoder trick", "B": "A tokenizer", "C": "A dataset"},
            "answer": "A",
        }
    ]

    staged_rows, index = flatten_mcq_records(source_records)
    assert staged_rows == [
        {"text": "What is grouped-query attention?"},
        {"text": "A decoder trick"},
        {"text": "A tokenizer"},
        {"text": "A dataset"},
    ]

    translated_rows = [
        {"translated_text": "समूहित-क्वेरी अटेंशन क्या है?", "faith_avg": 4.0},
        {"translated_text": "एक डिकोडर ट्रिक", "faith_avg": 4.5},
        {"translated_text": "एक टोकनाइज़र", "faith_avg": 5.0},
        {"translated_text": "एक डेटासेट", "faith_avg": 4.5},
    ]
    restored = restore_mcq_records(
        source_records,
        index,
        translated_rows,
        target_lang="hi",
    )

    assert restored[0]["answer"] == "A"
    assert restored[0]["question"] == "समूहित-क्वेरी अटेंशन क्या है?"
    assert restored[0]["options"]["A"] == "एक डिकोडर ट्रिक"
    assert restored[0]["translation_metadata"]["target_lang"] == "hi"
    assert restored[0]["faith_avg"] == 4.5
