from __future__ import annotations

import shutil
from pathlib import Path

from nemotron.steps.byob.scripts.validate import validate_skill_dir

REPO_ROOT = Path(__file__).resolve().parents[3]
BYOB_DIR = REPO_ROOT / "src" / "nemotron" / "steps" / "byob"


def test_checked_in_byob_assets_validate() -> None:
    assert validate_skill_dir(BYOB_DIR) == []


def test_validator_reports_missing_family_asset(tmp_path: Path) -> None:
    skill_dir = tmp_path / "byob"
    shutil.copytree(BYOB_DIR, skill_dir)
    (skill_dir / "mcq" / "config" / "tiny.yaml").unlink()

    errors = validate_skill_dir(skill_dir)

    assert "missing required file: mcq/config/tiny.yaml" in errors


def test_validator_discovers_additional_family(tmp_path: Path) -> None:
    skill_dir = tmp_path / "byob"
    shutil.copytree(BYOB_DIR, skill_dir)
    shutil.copytree(skill_dir / "mcq", skill_dir / "bfcl")
    shutil.copytree(
        skill_dir / "runtime" / "benchmark_families" / "mcq",
        skill_dir / "runtime" / "benchmark_families" / "bfcl",
    )

    assert validate_skill_dir(skill_dir) == []
