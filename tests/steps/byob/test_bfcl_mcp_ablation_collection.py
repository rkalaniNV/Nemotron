from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from nemotron.steps.byob.runtime.mcp.ablation import AblationError
from nemotron.steps.byob.runtime.mcp.ablation_collection import (
    assemble_ablation_input,
    begin_collection,
    digest_artifact_tree,
    finish_collection,
    mark_review_started,
    stop_collection,
)

_DIGEST = "sha256:" + "a" * 64
_START = datetime(2026, 8, 27, 1, 0, tzinfo=timezone.utc)


def _artifact(path: Path, value: str) -> Path:
    path.mkdir()
    (path / "run_manifest.json").write_text(value, encoding="utf-8")
    return path


def test_collection_measures_phases_and_digests_immutable_run_artifact(
    tmp_path: Path,
) -> None:
    state = tmp_path / "state.json"
    output = tmp_path / "observation.json"
    artifact = _artifact(tmp_path / "artifact", "run-one")
    begin_collection(
        state,
        flow="manual",
        repetition=1,
        sequence=1,
        now=_START,
    )
    mark_review_started(state, now=_START + timedelta(minutes=30))
    stop_collection(state, now=_START + timedelta(minutes=42))
    observation = finish_collection(
        state,
        output,
        run_artifact=artifact,
        user_authored_fields=12,
        validation_pass_rate=1.0,
        tool_coverage=1.0,
        replay_stability=1.0,
        benchmark_rows=20,
        excluded_authoring_minutes=5,
        excluded_review_minutes=2,
        now=_START + timedelta(minutes=90),
    )

    assert observation.authoring_minutes == 25
    assert observation.review_minutes == 10
    assert observation.run_digest == digest_artifact_tree(artifact)
    assert output.is_file()
    with pytest.raises(AblationError, match="already written"):
        finish_collection(
            state,
            tmp_path / "again.json",
            run_artifact=artifact,
            user_authored_fields=12,
            validation_pass_rate=1.0,
            tool_coverage=1.0,
            replay_stability=1.0,
            benchmark_rows=20,
            now=_START + timedelta(minutes=43),
        )


def test_collection_assembles_only_complete_nine_run_protocol(
    tmp_path: Path,
) -> None:
    schedule = [
        ("manual", 1),
        ("llm_backend", 1),
        ("llm_mcp", 1),
        ("llm_mcp", 2),
        ("manual", 2),
        ("llm_backend", 2),
        ("llm_backend", 3),
        ("llm_mcp", 3),
        ("manual", 3),
    ]
    observations: list[Path] = []
    for sequence, (flow, repetition) in enumerate(schedule, start=1):
        state = tmp_path / f"state-{sequence}.json"
        output = tmp_path / f"observation-{sequence}.json"
        artifact = _artifact(
            tmp_path / f"artifact-{sequence}",
            f"run-{sequence}",
        )
        started = _START + timedelta(hours=sequence)
        begin_collection(
            state,
            flow=flow,  # type: ignore[arg-type]
            repetition=repetition,
            sequence=sequence,
            now=started,
        )
        mark_review_started(state, now=started + timedelta(minutes=10))
        finish_collection(
            state,
            output,
            run_artifact=artifact,
            user_authored_fields=sequence,
            validation_pass_rate=1.0,
            tool_coverage=1.0,
            replay_stability=1.0,
            benchmark_rows=20,
            now=started + timedelta(minutes=15),
        )
        observations.append(output)

    with pytest.raises(AblationError, match="repetitions 1, 2, and 3"):
        assemble_ablation_input(
            observations[:-1],
            tmp_path / "incomplete.json",
            experiment_id="tiny-pilot",
            domain_artifact_digest=_DIGEST,
            evaluator_model="not_run",
            evaluation_config_digest=_DIGEST,
            held_out_policy_digest=_DIGEST,
        )

    source = assemble_ablation_input(
        list(reversed(observations)),
        tmp_path / "input.json",
        experiment_id="tiny-pilot",
        domain_artifact_digest=_DIGEST,
        evaluator_model="not_run",
        evaluation_config_digest=_DIGEST,
        held_out_policy_digest=_DIGEST,
    )
    assert [item.sequence for item in source.observations] == list(range(1, 10))
