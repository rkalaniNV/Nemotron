"""Strict, reproducible comparison contract for the three BFCL onboarding flows."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    StrictStr,
    model_validator,
)

from nemotron.steps.byob.runtime.pack_authoring.artifacts import (
    sha256_json,
    write_canonical_json,
)

ABLATION_INPUT_VERSION = "bfcl-onboarding-ablation-input-v2"
ABLATION_REPORT_VERSION = "bfcl-onboarding-ablation-report-v2"
FlowName = Literal["manual", "llm_backend", "llm_mcp"]
_FLOWS = frozenset({"manual", "llm_backend", "llm_mcp"})
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


class AblationError(ValueError):
    """Raised when three runs are not comparable."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FlowObservation(_StrictModel):
    flow: FlowName
    repetition: StrictInt = Field(ge=1, le=3)
    sequence: StrictInt = Field(ge=1, le=9)
    run_digest: StrictStr
    user_authored_fields: StrictInt = Field(ge=0)
    authoring_minutes: StrictFloat = Field(ge=0)
    review_minutes: StrictFloat = Field(ge=0)
    validation_pass_rate: StrictFloat = Field(ge=0, le=1)
    tool_coverage: StrictFloat = Field(ge=0, le=1)
    replay_stability: StrictFloat = Field(ge=0, le=1)
    benchmark_rows: StrictInt = Field(gt=0)
    evaluation_score: StrictFloat | None = None
    evaluation_score_stderr: StrictFloat | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_observation(self) -> FlowObservation:
        if _DIGEST.fullmatch(self.run_digest) is None:
            raise ValueError("run_digest must be sha256:<64 lowercase hex>")
        numeric = (
            self.authoring_minutes,
            self.review_minutes,
            self.validation_pass_rate,
            self.tool_coverage,
            self.replay_stability,
            self.evaluation_score,
            self.evaluation_score_stderr,
        )
        if any(value is not None and not math.isfinite(value) for value in numeric):
            raise ValueError("ablation metrics must be finite")
        if self.evaluation_score is None and self.evaluation_score_stderr is not None:
            raise ValueError("evaluation_score_stderr requires evaluation_score")
        return self


class AblationInput(_StrictModel):
    schema_version: Literal["bfcl-onboarding-ablation-input-v2"]
    experiment_id: StrictStr
    domain_artifact_digest: StrictStr
    evaluator_model: StrictStr
    evaluation_config_digest: StrictStr
    held_out_policy_digest: StrictStr
    repetitions_per_flow: Literal[3]
    observations: tuple[FlowObservation, ...]

    @model_validator(mode="after")
    def validate_comparability(self) -> AblationInput:
        if not self.experiment_id.strip() or not self.evaluator_model.strip():
            raise ValueError("experiment_id and evaluator_model must be non-empty")
        for value, label in (
            (self.domain_artifact_digest, "domain_artifact_digest"),
            (self.evaluation_config_digest, "evaluation_config_digest"),
            (self.held_out_policy_digest, "held_out_policy_digest"),
        ):
            if _DIGEST.fullmatch(value) is None:
                raise ValueError(f"{label} must be sha256:<64 lowercase hex>")
        pairs = {(item.flow, item.repetition) for item in self.observations}
        expected_pairs = {
            (flow, repetition)
            for flow in _FLOWS
            for repetition in range(1, 4)
        }
        if len(self.observations) != 9 or pairs != expected_pairs:
            raise ValueError(
                "observations must contain repetitions 1, 2, and 3 for every flow"
            )
        sequences = {item.sequence for item in self.observations}
        if sequences != set(range(1, 10)):
            raise ValueError("observation sequence must contain each value from 1 through 9")
        run_digests = [item.run_digest for item in self.observations]
        if len(set(run_digests)) != len(run_digests):
            raise ValueError("every ablation run_digest must be unique")
        scored = [item.evaluation_score is not None for item in self.observations]
        if any(scored) and not all(scored):
            raise ValueError(
                "evaluation_score must be present for all nine runs or omitted for all"
            )
        return self


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AblationError(f"ablation input repeats JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(token: str) -> None:
    raise AblationError(f"ablation input contains non-finite constant {token}")


def load_ablation_input(path: Path) -> AblationInput:
    try:
        raw = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_pairs,
            parse_constant=_reject_constant,
        )
        return cast(AblationInput, AblationInput.model_validate(raw))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise AblationError(f"cannot load comparable ablation input {path.resolve()}: {exc}") from exc


def build_ablation_report(source: AblationInput) -> dict[str, Any]:
    """Aggregate three repetitions per flow without inventing statistical claims."""
    by_flow = {
        flow: sorted(
            (item for item in source.observations if item.flow == flow),
            key=lambda item: item.repetition,
        )
        for flow in ("manual", "llm_backend", "llm_mcp")
    }

    def mean(items: list[FlowObservation], field: str) -> float:
        return sum(float(getattr(item, field)) for item in items) / len(items)

    def aggregate(items: list[FlowObservation]) -> dict[str, Any]:
        metrics: dict[str, Any] = {
            field: mean(items, field)
            for field in (
                "user_authored_fields",
                "authoring_minutes",
                "review_minutes",
                "validation_pass_rate",
                "tool_coverage",
                "replay_stability",
                "benchmark_rows",
            )
        }
        metrics["total_human_minutes"] = (
            metrics["authoring_minutes"] + metrics["review_minutes"]
        )
        metrics["quality_index"] = (
            metrics["validation_pass_rate"]
            + metrics["tool_coverage"]
            + metrics["replay_stability"]
        ) / 3
        scores = [item.evaluation_score for item in items]
        metrics["evaluation_score"] = (
            None
            if any(score is None for score in scores)
            else sum(cast(float, score) for score in scores) / len(scores)
        )
        return metrics

    aggregates = {flow: aggregate(items) for flow, items in by_flow.items()}
    manual = aggregates["manual"]
    rows: list[dict[str, Any]] = []
    for flow in ("manual", "llm_backend", "llm_mcp"):
        aggregate_row = aggregates[flow]
        rows.append(
            {
                "flow": flow,
                "repetitions": len(by_flow[flow]),
                "runs": [
                    item.model_dump(mode="json") for item in by_flow[flow]
                ],
                "mean": aggregate_row,
                "delta_vs_manual": {
                    field: aggregate_row[field] - manual[field]
                    for field in (
                        "user_authored_fields",
                        "total_human_minutes",
                        "validation_pass_rate",
                        "tool_coverage",
                        "replay_stability",
                        "benchmark_rows",
                        "quality_index",
                    )
                }
                | {
                    "evaluation_score": (
                        None
                        if aggregate_row["evaluation_score"] is None
                        or manual["evaluation_score"] is None
                        else aggregate_row["evaluation_score"]
                        - manual["evaluation_score"]
                    ),
                },
            }
        )
    report: dict[str, Any] = {
        "schema_version": ABLATION_REPORT_VERSION,
        "experiment_id": source.experiment_id,
        "comparison_contract": {
            "evaluator_model": source.evaluator_model,
            "domain_artifact_digest": source.domain_artifact_digest,
            "evaluation_config_digest": source.evaluation_config_digest,
            "held_out_policy_digest": source.held_out_policy_digest,
            "repetitions_per_flow": source.repetitions_per_flow,
            "run_order_recorded_by": "sequence",
            "baseline_flow": "manual",
            "quality_index": "mean(validation_pass_rate, tool_coverage, replay_stability)",
            "causal_claim": False,
        },
        "flows": rows,
        "warnings": [
            "Quality index is descriptive, not a publication score.",
            "Score deltas are comparable only when all three runs use the pinned evaluator and held-out policy.",
        ],
    }
    report["report_digest"] = sha256_json(report)
    return report


def write_ablation_report(report: dict[str, Any], path: Path) -> Path:
    claimed = report.get("report_digest")
    unsigned = {key: value for key, value in report.items() if key != "report_digest"}
    if claimed != sha256_json(unsigned):
        raise AblationError("ablation report_digest mismatch")
    return cast(Path, write_canonical_json(report, path))
