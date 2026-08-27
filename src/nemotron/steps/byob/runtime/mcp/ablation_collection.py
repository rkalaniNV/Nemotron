"""Real-observation collection helpers for the BFCL onboarding ablation."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from nemotron.steps.byob.runtime.mcp.ablation import (
    ABLATION_INPUT_VERSION,
    AblationError,
    AblationInput,
    FlowName,
    FlowObservation,
)
from nemotron.steps.byob.runtime.pack_authoring.artifacts import (
    sha256_json,
    write_canonical_json,
)

COLLECTION_STATE_VERSION = "bfcl-onboarding-ablation-collection-v1"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_time(value: Any, field: str) -> datetime:
    if not isinstance(value, str):
        raise AblationError(f"collection {field} must be an ISO timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise AblationError(f"collection {field} must be an ISO timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise AblationError(f"collection {field} must include a UTC offset")
    return parsed


def _load_json(path: Path) -> Any:
    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        document: dict[str, Any] = {}
        for key, value in pairs:
            if key in document:
                raise AblationError(f"{path} repeats JSON key {key!r}")
            document[key] = value
        return document

    def reject_constant(token: str) -> None:
        raise AblationError(f"{path} contains non-finite constant {token}")

    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=unique_pairs,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AblationError(f"cannot load collection file {path}: {exc}") from exc


def _load_state(path: Path) -> dict[str, Any]:
    document = _load_json(path)
    if not isinstance(document, dict):
        raise AblationError("collection state must be a JSON object")
    expected = {
        "schema_version",
        "flow",
        "repetition",
        "sequence",
        "started_at",
        "review_started_at",
        "finished_at",
        "observation_written",
    }
    if set(document) != expected:
        raise AblationError("collection state has an invalid field set")
    if document["schema_version"] != COLLECTION_STATE_VERSION:
        raise AblationError("collection state version is unsupported")
    if document["flow"] not in {"manual", "llm_backend", "llm_mcp"}:
        raise AblationError("collection flow is invalid")
    if (
        not isinstance(document["repetition"], int)
        or isinstance(document["repetition"], bool)
        or document["repetition"] not in {1, 2, 3}
    ):
        raise AblationError("collection repetition must be 1, 2, or 3")
    if (
        not isinstance(document["sequence"], int)
        or isinstance(document["sequence"], bool)
        or not 1 <= document["sequence"] <= 9
    ):
        raise AblationError("collection sequence must be between 1 and 9")
    _parse_time(document["started_at"], "started_at")
    for field in ("review_started_at", "finished_at"):
        if document[field] is not None:
            _parse_time(document[field], field)
    if not isinstance(document["observation_written"], bool):
        raise AblationError("collection observation_written must be boolean")
    return document


def begin_collection(
    path: Path,
    *,
    flow: FlowName,
    repetition: int,
    sequence: int,
    now: datetime | None = None,
) -> Path:
    if path.exists():
        raise AblationError(f"collection state already exists: {path}")
    if flow not in {"manual", "llm_backend", "llm_mcp"}:
        raise AblationError("flow is invalid")
    if repetition not in {1, 2, 3}:
        raise AblationError("repetition must be 1, 2, or 3")
    if not 1 <= sequence <= 9:
        raise AblationError("sequence must be between 1 and 9")
    started = now or _now()
    if started.tzinfo is None or started.utcoffset() is None:
        raise AblationError("collection timestamps must include a UTC offset")
    return cast(
        Path,
        write_canonical_json(
            {
                "schema_version": COLLECTION_STATE_VERSION,
                "flow": flow,
                "repetition": repetition,
                "sequence": sequence,
                "started_at": started.isoformat(),
                "review_started_at": None,
                "finished_at": None,
                "observation_written": False,
            },
            path,
        ),
    )


def mark_review_started(
    path: Path,
    *,
    now: datetime | None = None,
) -> Path:
    state = _load_state(path)
    if state["review_started_at"] is not None:
        raise AblationError("review was already started")
    if state["finished_at"] is not None:
        raise AblationError("collection is already finished")
    marked = now or _now()
    started = _parse_time(state["started_at"], "started_at")
    if marked <= started:
        raise AblationError("review start must be after authoring start")
    state["review_started_at"] = marked.isoformat()
    return cast(Path, write_canonical_json(state, path))


def stop_collection(
    path: Path,
    *,
    now: datetime | None = None,
) -> Path:
    """Stop review timing without requiring metrics to be available yet."""
    state = _load_state(path)
    if state["review_started_at"] is None:
        raise AblationError("mark review start before stopping a collection")
    if state["finished_at"] is not None:
        raise AblationError("collection timer is already stopped")
    stopped = now or _now()
    review_started = _parse_time(
        state["review_started_at"],
        "review_started_at",
    )
    if stopped <= review_started:
        raise AblationError("finish time must be after review start")
    state["finished_at"] = stopped.isoformat()
    return cast(Path, write_canonical_json(state, path))


def digest_artifact_tree(root: Path) -> str:
    """Digest every regular file by relative path, size, and content hash."""
    resolved = root.resolve()
    if not resolved.is_dir():
        raise AblationError(f"run artifact must be a directory: {resolved}")
    records: list[dict[str, Any]] = []
    for path in sorted(resolved.rglob("*")):
        if path.is_symlink():
            raise AblationError(f"run artifact contains a symbolic link: {path}")
        if not path.is_file():
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        records.append(
            {
                "path": path.relative_to(resolved).as_posix(),
                "size": path.stat().st_size,
                "sha256": digest,
            }
        )
    if not records:
        raise AblationError("run artifact directory contains no files")
    return sha256_json({"files": records})


def finish_collection(
    state_path: Path,
    output_path: Path,
    *,
    run_artifact: Path,
    user_authored_fields: int,
    validation_pass_rate: float,
    tool_coverage: float,
    replay_stability: float,
    benchmark_rows: int,
    excluded_authoring_minutes: float = 0,
    excluded_review_minutes: float = 0,
    evaluation_score: float | None = None,
    evaluation_score_stderr: float | None = None,
    now: datetime | None = None,
) -> FlowObservation:
    state = _load_state(state_path)
    if state["review_started_at"] is None:
        raise AblationError("mark review start before finishing a collection")
    if state["observation_written"]:
        raise AblationError("collection observation was already written")
    finished = (
        _parse_time(state["finished_at"], "finished_at")
        if state["finished_at"] is not None
        else (now or _now())
    )
    started = _parse_time(state["started_at"], "started_at")
    review_started = _parse_time(state["review_started_at"], "review_started_at")
    if finished <= review_started:
        raise AblationError("finish time must be after review start")
    if excluded_authoring_minutes < 0 or excluded_review_minutes < 0:
        raise AblationError("excluded minutes cannot be negative")
    authoring_minutes = (
        (review_started - started).total_seconds() / 60
        - excluded_authoring_minutes
    )
    review_minutes = (
        (finished - review_started).total_seconds() / 60
        - excluded_review_minutes
    )
    if authoring_minutes < 0 or review_minutes < 0:
        raise AblationError("excluded minutes exceed measured elapsed time")

    observation = FlowObservation.model_validate(
        {
            "flow": state["flow"],
            "repetition": state["repetition"],
            "sequence": state["sequence"],
            "run_digest": digest_artifact_tree(run_artifact),
            "user_authored_fields": user_authored_fields,
            "authoring_minutes": authoring_minutes,
            "review_minutes": review_minutes,
            "validation_pass_rate": validation_pass_rate,
            "tool_coverage": tool_coverage,
            "replay_stability": replay_stability,
            "benchmark_rows": benchmark_rows,
            "evaluation_score": evaluation_score,
            "evaluation_score_stderr": evaluation_score_stderr,
        }
    )
    if output_path.exists():
        raise AblationError(f"observation output already exists: {output_path}")
    write_canonical_json(observation.model_dump(mode="json"), output_path)
    state["finished_at"] = finished.isoformat()
    state["observation_written"] = True
    write_canonical_json(state, state_path)
    return observation


def assemble_ablation_input(
    observation_paths: list[Path],
    output_path: Path,
    *,
    experiment_id: str,
    domain_artifact_digest: str,
    evaluator_model: str,
    evaluation_config_digest: str,
    held_out_policy_digest: str,
) -> AblationInput:
    try:
        observations = [
            FlowObservation.model_validate(_load_json(path))
            for path in observation_paths
        ]
        source = AblationInput.model_validate(
            {
                "schema_version": ABLATION_INPUT_VERSION,
                "experiment_id": experiment_id,
                "domain_artifact_digest": domain_artifact_digest,
                "evaluator_model": evaluator_model,
                "evaluation_config_digest": evaluation_config_digest,
                "held_out_policy_digest": held_out_policy_digest,
                "repetitions_per_flow": 3,
                "observations": sorted(
                    observations,
                    key=lambda item: item.sequence,
                ),
            }
        )
    except ValueError as exc:
        raise AblationError(f"cannot assemble comparable observations: {exc}") from exc
    if output_path.exists():
        raise AblationError(f"ablation input already exists: {output_path}")
    write_canonical_json(source.model_dump(mode="json"), output_path)
    return source
