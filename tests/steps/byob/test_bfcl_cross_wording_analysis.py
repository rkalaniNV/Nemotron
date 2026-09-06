"""Tests for the BFCL cross-wording stability contract (SOV-862)."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

from nemotron.steps.byob.runtime.benchmark_families.bfcl.ablation_statistics import mcnemar_exact_p
from nemotron.steps.byob.runtime.benchmark_families.bfcl.cross_wording_analysis import (
    CROSS_WORDING_CONTRACT_VERSION,
    STABILITY_CONCLUSIONS,
    STRATUM_FIELDS,
    WORDING_SOURCES,
    CrossWordingError,
    CrossWordingInputs,
    ScoredRun,
    build_cross_wording_report,
    render_cross_wording_markdown,
    validate_cross_wording_report,
    write_cross_wording_report,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE = REPO_ROOT.parent
RELEASE = WORKSPACE / "releases" / "banking-vn-gold-v1-1392"
EVALUATIONS = WORKSPACE / "release-candidate" / "sov867-clean-52907cc" / "evaluations"


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def _published(path: Path, count: int = 12) -> Path:
    return _write_parquet(
        path,
        [
            {
                "task_id": f"task-{index:03d}",
                "turn_policy": "single_turn" if index % 2 else "dependent_call",
                "category": "transfer",
                "difficulty": "easy" if index < count // 2 else "hard",
            }
            for index in range(count)
        ],
    )


def _rendered(path: Path, count: int = 12, *, paired: bool = False) -> Path:
    rows: list[dict[str, Any]] = []
    for index in range(count):
        rows.append(
            {
                "task_id": f"task-{index:03d}",
                "base_task_id": f"base-{index:03d}",
                "source": "template" if index % 3 == 0 else "model",
            }
        )
        if paired:
            # A second published task sharing the skeleton under the other wording.
            rows.append(
                {
                    "task_id": f"task-{index:03d}",
                    "base_task_id": f"base-{index:03d}",
                    "source": "template" if index % 3 == 0 else "model",
                }
            )
    if paired:
        # Give one skeleton both wordings among the published tasks.
        rows.append({"task_id": "task-001", "base_task_id": "base-000", "source": "model"})
        rows = [row for row in rows if not (row["task_id"] == "task-001" and row["base_task_id"] == "base-001")]
    return _write_parquet(path, rows)


def _scored(path: Path, count: int = 12, *, passing: set[int] | None = None) -> Path:
    passing = passing if passing is not None else set(range(0, count, 2))
    return _write_parquet(
        path,
        [
            {
                "task_id": f"task-{index:03d}",
                "task_success": index in passing,
                "non_candidate_stop": False,
                "failure_codes": [] if index in passing else ["assertion_failed"],
            }
            for index in range(count)
        ],
    )


def _inputs(tmp_path: Path, **overrides: Any) -> CrossWordingInputs:
    defaults: dict[str, Any] = {
        "published": _published(tmp_path / "benchmark.parquet"),
        "rendered_conversations": _rendered(tmp_path / "rendered.parquet"),
        "primary_run": ScoredRun("primary", _scored(tmp_path / "primary.parquet"), "primary"),
        "output_dir": tmp_path / "out",
    }
    return CrossWordingInputs(**(defaults | overrides))


def test_report_is_deterministic_and_content_addressed(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    first = build_cross_wording_report(inputs)
    second = build_cross_wording_report(inputs)
    assert first == second
    assert first["report_hash"].startswith("sha256:")
    validate_cross_wording_report(first)


def test_tampering_with_a_conclusion_breaks_the_hash(tmp_path: Path) -> None:
    report = build_cross_wording_report(_inputs(tmp_path))
    tampered = copy.deepcopy(report)
    tampered["stability_conclusion"]["conclusion"] = "stable"
    with pytest.raises(CrossWordingError, match="report_hash mismatch"):
        validate_cross_wording_report(tampered)


def test_a_stability_verdict_requires_the_paired_design(tmp_path: Path) -> None:
    """Re-signing a forged verdict must still be refused, not just detected."""
    report = build_cross_wording_report(_inputs(tmp_path))
    assert report["paired_wording_design"]["status"] == "not_available"
    forged = copy.deepcopy(report)
    forged["stability_conclusion"]["conclusion"] = "stable"
    unsigned = {key: value for key, value in forged.items() if key != "report_hash"}
    import hashlib

    canonical = json.dumps(
        unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    forged["report_hash"] = f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"
    with pytest.raises(CrossWordingError, match="requires the paired wording design"):
        validate_cross_wording_report(forged)


def test_unpaired_release_is_reported_as_underpowered(tmp_path: Path) -> None:
    report = build_cross_wording_report(_inputs(tmp_path))
    conclusion = report["stability_conclusion"]
    assert conclusion["conclusion"] == "underpowered"
    assert report["paired_wording_design"]["missing_artifact"]
    assert conclusion["required_to_conclude"]
    assert report["policy"]["causal_claim"] is False


def test_paired_availability_is_detected_when_present(tmp_path: Path) -> None:
    inputs = _inputs(
        tmp_path,
        rendered_conversations=_rendered(tmp_path / "rendered_paired.parquet", paired=True),
    )
    report = build_cross_wording_report(inputs)
    assert report["paired_wording_design"]["status"] == "available"
    assert report["paired_wording_design"]["paired_base_task_count"] >= 1
    # Availability alone is not a verdict; the paired run still has to be supplied.
    assert report["stability_conclusion"]["conclusion"] == "not_measured"


def test_replicate_floor_measures_verdict_flips(tmp_path: Path) -> None:
    replicate = _scored(tmp_path / "replicate.parquet", passing={0, 1, 2, 4, 6, 8, 10})
    report = build_cross_wording_report(
        _inputs(
            tmp_path,
            replicate_runs=(ScoredRun("replicate", replicate, "replicate"),),
        )
    )
    floor = report["replicate_floor"]
    assert floor["status"] == "measured"
    paired = floor["runs"][0]["paired"]
    assert paired["paired_tasks"] == 12
    assert paired["discordant_pairs"] == 1
    assert paired["flipped_task_ids"] == ["task-001"]
    assert paired["right_only_pass"] == 1
    assert paired["agreement"] == pytest.approx(11 / 12)


def test_identical_replicate_is_refused(tmp_path: Path) -> None:
    primary = _scored(tmp_path / "primary.parquet")
    duplicate = _write_parquet(tmp_path / "copy.parquet", [])
    duplicate.write_bytes(primary.read_bytes())
    with pytest.raises(CrossWordingError, match="byte-identical"):
        build_cross_wording_report(
            _inputs(
                tmp_path,
                primary_run=ScoredRun("primary", primary, "primary"),
                replicate_runs=(ScoredRun("replicate", duplicate, "replicate"),),
            )
        )


def test_duplicate_replicate_run_id_is_refused(tmp_path: Path) -> None:
    replicate = _scored(tmp_path / "replicate.parquet", passing={1, 3})
    with pytest.raises(CrossWordingError, match="not unique"):
        build_cross_wording_report(
            _inputs(
                tmp_path,
                replicate_runs=(
                    ScoredRun("primary", replicate, "replicate"),
                ),
            )
        )


def test_partial_scoring_is_refused(tmp_path: Path) -> None:
    partial = _write_parquet(
        tmp_path / "partial.parquet",
        [
            {
                "task_id": f"task-{index:03d}",
                "task_success": True,
                "non_candidate_stop": False,
                "failure_codes": [],
            }
            for index in range(6)
        ],
    )
    with pytest.raises(CrossWordingError, match="does not score exactly the published task set"):
        build_cross_wording_report(
            _inputs(tmp_path, primary_run=ScoredRun("primary", partial, "primary"))
        )


def test_unknown_wording_source_is_refused(tmp_path: Path) -> None:
    rendered = _write_parquet(
        tmp_path / "bad_rendered.parquet",
        [
            {"task_id": f"task-{index:03d}", "base_task_id": f"base-{index:03d}", "source": "human"}
            for index in range(12)
        ],
    )
    with pytest.raises(CrossWordingError, match="unknown wording source"):
        build_cross_wording_report(_inputs(tmp_path, rendered_conversations=rendered))


def test_uncovered_wording_provenance_is_refused(tmp_path: Path) -> None:
    rendered = _write_parquet(
        tmp_path / "short_rendered.parquet",
        [
            {"task_id": f"task-{index:03d}", "base_task_id": f"base-{index:03d}", "source": "model"}
            for index in range(9)
        ],
    )
    with pytest.raises(CrossWordingError, match="do not cover every published task"):
        build_cross_wording_report(_inputs(tmp_path, rendered_conversations=rendered))


def test_single_wording_group_is_not_comparable(tmp_path: Path) -> None:
    rendered = _write_parquet(
        tmp_path / "one_wording.parquet",
        [
            {"task_id": f"task-{index:03d}", "base_task_id": f"base-{index:03d}", "source": "model"}
            for index in range(12)
        ],
    )
    report = build_cross_wording_report(_inputs(tmp_path, rendered_conversations=rendered))
    overall = report["unpaired_wording_contrast"]["overall"]
    assert overall["status"] == "not_comparable"
    assert overall["absolute_delta"] is None
    assert "only one wording" in overall["reason"]


def test_saturated_groups_are_flagged_as_uninformative(tmp_path: Path) -> None:
    all_failing = _scored(tmp_path / "all_fail.parquet", passing=set())
    report = build_cross_wording_report(
        _inputs(tmp_path, primary_run=ScoredRun("primary", all_failing, "primary"))
    )
    findings = " ".join(report["stability_conclusion"]["findings"])
    assert "saturated at a floor or ceiling" in findings


def test_writer_refuses_to_replace_a_different_report(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    report = build_cross_wording_report(inputs)
    json_path, markdown_path = write_cross_wording_report(report, inputs.output_dir)
    # Rewriting the same report is a no-op rather than an error.
    assert write_cross_wording_report(report, inputs.output_dir) == (json_path, markdown_path)
    json_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(CrossWordingError, match="refusing to replace"):
        write_cross_wording_report(report, inputs.output_dir)
    assert markdown_path.is_file()


def test_markdown_states_the_claim_boundary(tmp_path: Path) -> None:
    markdown = render_cross_wording_markdown(build_cross_wording_report(_inputs(tmp_path)))
    assert "Causal claim: no." in markdown
    assert "unpaired_observational" in markdown
    assert "Stability conclusion: **underpowered**" in markdown


def test_contract_document_matches_the_implemented_contract() -> None:
    text = (
        REPO_ROOT
        / "src"
        / "nemotron"
        / "steps"
        / "byob"
        / "references"
        / "bfcl-cross-wording-contract.md"
    ).read_text(encoding="utf-8")

    assert f"contract version is `{CROSS_WORDING_CONTRACT_VERSION}`" in text
    for conclusion in STABILITY_CONCLUSIONS:
        assert f"`{conclusion}`" in text, conclusion
    for stratum in STRATUM_FIELDS:
        assert f"`{stratum}`" in text, stratum
    for source in WORDING_SOURCES:
        assert f"`{source}`" in text, source


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        (0, 0, 1.0),
        (1, 1, 1.0),
        (0, 10, 0.001953125),
        (23, 28, 0.5758493477),
    ],
)
def test_mcnemar_exact_p_matches_the_binomial_tail(left: int, right: int, expected: float) -> None:
    assert mcnemar_exact_p(left, right) == pytest.approx(expected, abs=1e-9)


@pytest.mark.skipif(
    not (RELEASE / "benchmark" / "benchmark.parquet").is_file()
    or not (EVALUATIONS / "gptoss-120b-8k" / "eval_task_results.parquet").is_file(),
    reason="frozen Banking VN release artifacts are not present in this checkout",
)
def test_frozen_release_report_hash_is_pinned(tmp_path: Path) -> None:
    """Golden test: the published SOV-862 readout must stay reproducible."""
    report = build_cross_wording_report(
        CrossWordingInputs(
            published=RELEASE / "benchmark" / "benchmark.parquet",
            rendered_conversations=RELEASE / "stage_cache" / "rendered_conversations.parquet",
            primary_run=ScoredRun(
                "gptoss-120b-8k",
                EVALUATIONS / "gptoss-120b-8k" / "eval_task_results.parquet",
                "primary",
            ),
            replicate_runs=(
                ScoredRun(
                    "gptoss-structural-2",
                    EVALUATIONS / "gptoss-structural-2" / "eval_task_results.parquet",
                    "replicate",
                ),
            ),
            output_dir=tmp_path,
        )
    )
    assert report["report_hash"] == (
        "sha256:bf420299bbbf6ac9baf8e005705822496493abed4d3e4d77eea754d4979d5882"
    )
    assert report["stability_conclusion"]["conclusion"] == "underpowered"
    assert report["wording_distribution"] == {"template": 305, "model": 1087}
    assert report["replicate_floor"]["runs"][0]["paired"]["discordant_pairs"] == 51
