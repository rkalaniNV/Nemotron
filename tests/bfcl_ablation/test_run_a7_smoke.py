from __future__ import annotations

import json
import shutil
from pathlib import Path

from bfcl_ablation import common
from bfcl_ablation.run_a7 import main


def _copy_inputs(tmp_path: Path) -> Path:
    destination = tmp_path / "results"
    shutil.copytree(
        common.RESULTS,
        destination,
        ignore=shutil.ignore_patterns("A7"),
    )
    return destination


def test_run_a7_writes_deterministic_report_only_outputs(tmp_path: Path) -> None:
    results = _copy_inputs(tmp_path)
    template = tmp_path / "human_labels.template.yaml"
    args = [
        "--results-dir",
        str(results),
        "--emit-label-template",
        str(template),
    ]
    assert main(args) == 0
    output = results / "A7"
    for name in ("metrics.json", "checks.json", "label_coverage.json", "report.md"):
        assert (output / name).is_file()
    assert template.is_file()

    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["rollup"] == {
        "integrity": "PASS",
        "release_readiness": "FAIL",
        "study_validity": "INCONCLUSIVE",
    }
    assert metrics["publication_decision"] == "NOT_READY"
    before = {
        name: (output / name).read_bytes()
        for name in ("metrics.json", "checks.json", "label_coverage.json", "report.md")
    }
    assert main(args) == 0
    assert before == {
        name: (output / name).read_bytes()
        for name in ("metrics.json", "checks.json", "label_coverage.json", "report.md")
    }


def test_strict_mode_fails_current_release_policy(tmp_path: Path) -> None:
    results = _copy_inputs(tmp_path)
    assert main(["--results-dir", str(results), "--strict"]) == 1


def test_invalid_label_schema_is_an_input_error(tmp_path: Path) -> None:
    results = _copy_inputs(tmp_path)
    labels = tmp_path / "bad-labels.yaml"
    labels.write_text("schema_version: '9.9'\n", encoding="utf-8")
    assert main(["--results-dir", str(results), "--labels", str(labels)]) == 2
