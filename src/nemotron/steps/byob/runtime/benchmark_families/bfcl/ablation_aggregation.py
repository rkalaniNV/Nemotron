"""Deterministic, read-only aggregation of the BFCL ablation ladder.

The ladder opens one degree of freedom per arm, so a conclusion is only worth
publishing when the arm that produced it is identifiable. This module compares
every arm against the declared baseline, separates material change from noise,
and derives a release recommendation from a fixed policy rather than prose.

Two properties matter more than the arithmetic. Missing evidence is never
represented as a zero, because an unmeasured metric and a metric measured at
zero support opposite decisions. And a truth-preservation gate only counts as
evidence when it is capable of failing under the intervention it guards: a
verdict computed over fields the arm does not touch carries no information, and
recording it as a pass is how an ablation launders an unchecked risk into a
green report.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

from nemotron.steps.byob.runtime.benchmark_families.bfcl.ablation_statistics import (
    StatisticsError,
    critical_z,
    newcombe_difference_interval,
    round_statistic,
    two_proportion_test,
    welch_test,
)

ABLATION_AGGREGATION_VERSION: Final = "1.2"

MEASUREMENT_KINDS: Final = ("deterministic", "proportion", "repeated")
ARM_STATUSES: Final = ("measured", "partially_measured", "deferred", "blocked")

# An arm that was executed only in a degraded design. It carries real evidence
# and real arm-local measurements, but the design does not license a delta
# against the baseline, so the contract records the numbers and withholds the
# comparison instead of choosing between publishing a misleading delta and
# discarding the measurement.
PARTIAL_STATUS: Final = "partially_measured"
GATE_VERDICTS: Final = ("passed", "failed", "not_run")

COMPARISON_VERDICTS: Final = (
    "material_improvement",
    "material_regression",
    "no_material_change",
    "inconclusive",
    "not_measured",
)
TRADE_OFF_VERDICTS: Final = (
    "gain_against_measured_cost",
    "gain_with_unpriced_risk",
    "gain_with_no_observed_cost",
    "cost_without_gain",
    "no_gain_observed",
)
RECOMMENDATIONS: Final = (
    "adopt",
    "adopt_with_conditions",
    "retain_baseline",
    "reject",
    "insufficient_evidence",
)

_MIN_TEST_SAMPLE: Final = 5


class AblationAggregationError(ValueError):
    """An input cannot support a trustworthy ablation conclusion."""


@dataclass(frozen=True)
class MetricDefinition:
    """A metric the contract knows how to compare, and in which direction."""

    family: str
    direction: str
    unit: str
    relative_threshold: float


# The registry is fixed on purpose. An operator who could declare a metric's own
# direction could turn a regression into an improvement by editing one field.
METRIC_DEFINITIONS: Final[Mapping[str, MetricDefinition]] = {
    "authoring_lines": MetricDefinition("effort", "lower_is_better", "lines", 0.05),
    "authoring_minutes": MetricDefinition("effort", "lower_is_better", "minutes", 0.10),
    "review_minutes": MetricDefinition("effort", "lower_is_better", "minutes", 0.10),
    "validation_pass_rate": MetricDefinition("quality", "higher_is_better", "ratio", 0.02),
    "replay_stability": MetricDefinition("quality", "higher_is_better", "ratio", 0.02),
    "gold_eligible_rate": MetricDefinition("quality", "higher_is_better", "ratio", 0.02),
    "intent_preservation_recall": MetricDefinition("quality", "higher_is_better", "ratio", 0.05),
    "tool_coverage": MetricDefinition("coverage", "higher_is_better", "ratio", 0.02),
    "entity_coverage": MetricDefinition("coverage", "higher_is_better", "ratio", 0.05),
    "joint_cell_coverage": MetricDefinition("coverage", "higher_is_better", "ratio", 0.05),
    "distinct_surface_count": MetricDefinition("coverage", "higher_is_better", "count", 0.10),
    "published_task_count": MetricDefinition("coverage", "higher_is_better", "count", 0.05),
    "task_success_rate": MetricDefinition("task_success", "higher_is_better", "ratio", 0.02),
    "cost_amount": MetricDefinition("cost", "lower_is_better", "currency", 0.10),
    "input_tokens": MetricDefinition("cost", "lower_is_better", "tokens", 0.10),
    "output_tokens": MetricDefinition("cost", "lower_is_better", "tokens", 0.10),
    "latency_p50_ms": MetricDefinition("latency", "lower_is_better", "milliseconds", 0.10),
    "latency_p95_ms": MetricDefinition("latency", "lower_is_better", "milliseconds", 0.10),
    "wall_clock_seconds": MetricDefinition("latency", "lower_is_better", "seconds", 0.10),
}

METRIC_FAMILIES: Final = (
    "effort",
    "quality",
    "coverage",
    "task_success",
    "failure_codes",
    "cost",
    "latency",
)

# Families SOV-866 requires the summary to speak to. A family with nothing
# measured anywhere is reported as a gap rather than omitted.
REQUIRED_FAMILIES: Final = (
    "quality",
    "coverage",
    "task_success",
    "failure_codes",
    "cost",
    "latency",
)


@dataclass(frozen=True)
class AggregationInputs:
    ablation_input: Path
    output_dir: Path


def _round(value: float) -> float:
    try:
        return round_statistic(value)
    except StatisticsError as exc:
        raise AblationAggregationError(str(exc)) from exc


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _semantic_hash(value: Any) -> str:
    payload = _canonical_json(value).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AblationAggregationError(f"{label} must be a mapping")
    return value


def _sequence(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise AblationAggregationError(f"{label} must be a list")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AblationAggregationError(f"{label} must be a non-empty string")
    return value


def _count(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise AblationAggregationError(f"{label} must be a non-negative integer")
    return value


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AblationAggregationError(f"{label} must be a number")
    if not math.isfinite(float(value)):
        raise AblationAggregationError(f"{label} must be finite")
    return float(value)


def _load_json_document(path: Path, label: str) -> dict[str, Any]:
    """Load JSON, refusing duplicate keys and non-finite constants."""

    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        document: dict[str, Any] = {}
        for key, value in pairs:
            if key in document:
                raise AblationAggregationError(f"{label} repeats JSON key {key!r}")
            document[key] = value
        return document

    def reject_constant(token: str) -> None:
        raise AblationAggregationError(f"{label} contains non-finite constant {token}")

    if not path.is_file():
        raise AblationAggregationError(f"{label} does not exist: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=unique_pairs,
            parse_constant=reject_constant,
        )
    except AblationAggregationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AblationAggregationError(f"{label} is not valid JSON: {path}") from exc
    return _mapping(value, label)


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _load_policy(raw: Any) -> dict[str, Any]:
    policy = _mapping(raw, "policy")
    unknown = set(policy) - {
        "confidence_level",
        "multiple_comparison",
        "practical_threshold_overrides",
    }
    if unknown:
        raise AblationAggregationError(f"policy has unknown fields: {sorted(unknown)}")
    confidence = _number(policy.get("confidence_level", 0.95), "policy.confidence_level")
    if not 0.5 < confidence < 1.0:
        raise AblationAggregationError("policy.confidence_level must be between 0.5 and 1.0")
    correction = policy.get("multiple_comparison", "holm")
    if correction != "holm":
        raise AblationAggregationError("policy.multiple_comparison must be 'holm'")
    thresholds = {
        metric_id: definition.relative_threshold for metric_id, definition in METRIC_DEFINITIONS.items()
    }
    overrides: list[dict[str, Any]] = []
    declared_overrides = _sequence(
        policy.get("practical_threshold_overrides", []),
        "policy.practical_threshold_overrides",
    )
    for index, item in enumerate(declared_overrides):
        override = _mapping(item, f"practical_threshold_overrides[{index}]")
        if set(override) != {"metric_id", "relative_threshold", "rationale"}:
            raise AblationAggregationError(
                f"practical_threshold_overrides[{index}] must declare metric_id, relative_threshold and rationale"
            )
        metric_id = _text(override["metric_id"], "override metric_id")
        if metric_id not in METRIC_DEFINITIONS:
            raise AblationAggregationError(f"override names an unknown metric: {metric_id}")
        value = _number(override["relative_threshold"], "override relative_threshold")
        if not 0.0 < value < 1.0:
            raise AblationAggregationError("override relative_threshold must be between 0 and 1")
        thresholds[metric_id] = value
        overrides.append(
            {
                "metric_id": metric_id,
                "relative_threshold": value,
                "rationale": _text(override["rationale"], "override rationale"),
            }
        )
    if len({item["metric_id"] for item in overrides}) != len(overrides):
        raise AblationAggregationError("practical_threshold_overrides repeat a metric")
    return {
        "confidence_level": confidence,
        "multiple_comparison": correction,
        "practical_threshold_overrides": overrides,
        "resolved_thresholds": thresholds,
    }


def _load_measurement(raw: Any, label: str) -> dict[str, Any]:
    measurement = _mapping(raw, label)
    metric_id = _text(measurement.get("metric_id"), f"{label}.metric_id")
    if metric_id not in METRIC_DEFINITIONS:
        raise AblationAggregationError(f"{label} names an unregistered metric: {metric_id}")
    kind = _text(measurement.get("kind"), f"{label}.kind")
    if kind not in MEASUREMENT_KINDS:
        raise AblationAggregationError(f"{label}.kind must be one of {list(MEASUREMENT_KINDS)}")
    if kind == "deterministic":
        expected = {"metric_id", "kind", "value"}
        if set(measurement) != expected:
            raise AblationAggregationError(f"{label} must declare exactly {sorted(expected)}")
        return {
            "metric_id": metric_id,
            "kind": kind,
            "value": _number(measurement["value"], f"{label}.value"),
        }
    if kind == "proportion":
        expected = {"metric_id", "kind", "numerator", "denominator"}
        if set(measurement) != expected:
            raise AblationAggregationError(f"{label} must declare exactly {sorted(expected)}")
        numerator = _count(measurement["numerator"], f"{label}.numerator")
        denominator = _count(measurement["denominator"], f"{label}.denominator")
        if denominator == 0:
            raise AblationAggregationError(f"{label}.denominator must be positive")
        if numerator > denominator:
            raise AblationAggregationError(f"{label}.numerator cannot exceed its denominator")
        return {
            "metric_id": metric_id,
            "kind": kind,
            "numerator": numerator,
            "denominator": denominator,
            "value": numerator / denominator,
        }
    expected = {"metric_id", "kind", "observations"}
    if set(measurement) != expected:
        raise AblationAggregationError(f"{label} must declare exactly {sorted(expected)}")
    observations = [
        _number(item, f"{label}.observations[{index}]")
        for index, item in enumerate(_sequence(measurement["observations"], f"{label}.observations"))
    ]
    if len(observations) < 2:
        raise AblationAggregationError(f"{label}.observations needs at least two values")
    return {
        "metric_id": metric_id,
        "kind": kind,
        "observations": observations,
        "value": statistics.fmean(observations),
    }


def _load_gate(raw: Any, label: str) -> dict[str, Any]:
    gate = _mapping(raw, label)
    expected = {"gate_id", "verdict", "sensitive_to_intervention", "rationale"}
    if set(gate) != expected:
        raise AblationAggregationError(f"{label} must declare exactly {sorted(expected)}")
    verdict = _text(gate["verdict"], f"{label}.verdict")
    if verdict not in GATE_VERDICTS:
        raise AblationAggregationError(f"{label}.verdict must be one of {list(GATE_VERDICTS)}")
    sensitive = gate["sensitive_to_intervention"]
    if not isinstance(sensitive, bool):
        raise AblationAggregationError(f"{label}.sensitive_to_intervention must be boolean")
    return {
        "gate_id": _text(gate["gate_id"], f"{label}.gate_id"),
        "verdict": verdict,
        "sensitive_to_intervention": sensitive,
        "rationale": _text(gate["rationale"], f"{label}.rationale"),
    }


def _load_evidence(raw: Any, label: str, root: Path) -> dict[str, Any]:
    evidence = _mapping(raw, label)
    expected = {"kind", "locator", "content_hash"}
    if set(evidence) != expected:
        raise AblationAggregationError(f"{label} must declare exactly {sorted(expected)}")
    locator = _text(evidence["locator"], f"{label}.locator")
    declared = _text(evidence["content_hash"], f"{label}.content_hash")
    if not declared.startswith("sha256:") or len(declared) != 71:
        raise AblationAggregationError(f"{label}.content_hash must be sha256:<64 hex>")
    record = {
        "kind": _text(evidence["kind"], f"{label}.kind"),
        "locator": locator,
        "content_hash": declared,
        "verified": False,
    }
    candidate = Path(locator)
    if not candidate.is_absolute():
        candidate = root / candidate
    if candidate.is_file():
        if _file_hash(candidate) != declared:
            raise AblationAggregationError(
                f"{label} content hash does not match the file on disk: {locator}"
            )
        record["verified"] = True
    return record


def _load_arm(raw: Any, index: int, root: Path) -> dict[str, Any]:
    arm = _mapping(raw, f"arms[{index}]")
    label = f"arms[{index}]"
    status = _text(arm.get("status"), f"{label}.status")
    if status not in ARM_STATUSES:
        raise AblationAggregationError(f"{label}.status must be one of {list(ARM_STATUSES)}")
    common = {"arm_id", "title", "ticket", "status", "intervention"}
    carries_evidence = status in ("measured", PARTIAL_STATUS)
    if carries_evidence:
        allowed = common | {
            "evidence",
            "measurements",
            "task_set_hash",
            "failure_codes",
            "cost_context",
            "truth_preservation_gates",
            "limitations",
        }
        required = {"evidence", "measurements"}
        if status == PARTIAL_STATUS:
            # A withheld comparison has to say why, in the same field a deferred
            # arm uses, so no arm can go quiet about a missing design.
            allowed |= {"deferral_reason"}
            required |= {"deferral_reason"}
    else:
        allowed = common | {"deferral_reason", "limitations"}
        required = {"deferral_reason"}
    unknown = set(arm) - allowed
    if unknown:
        raise AblationAggregationError(f"{label} has unsupported fields: {sorted(unknown)}")
    missing = (common | required) - set(arm)
    if missing:
        raise AblationAggregationError(f"{label} is missing required fields: {sorted(missing)}")

    record: dict[str, Any] = {
        "arm_id": _text(arm["arm_id"], f"{label}.arm_id"),
        "title": _text(arm["title"], f"{label}.title"),
        "ticket": _text(arm["ticket"], f"{label}.ticket"),
        "status": status,
        "intervention": _text(arm["intervention"], f"{label}.intervention"),
        "limitations": [
            _text(item, f"{label}.limitations[{position}]")
            for position, item in enumerate(_sequence(arm.get("limitations", []), f"{label}.limitations"))
        ],
    }
    if status == PARTIAL_STATUS:
        record["deferral_reason"] = _text(arm["deferral_reason"], f"{label}.deferral_reason")
    if not carries_evidence:
        record["deferral_reason"] = _text(arm["deferral_reason"], f"{label}.deferral_reason")
        record["measurements"] = {}
        record["evidence"] = []
        record["failure_codes"] = None
        record["cost_context"] = None
        record["truth_preservation_gates"] = []
        record["task_set_hash"] = None
        return record

    evidence = [
        _load_evidence(item, f"{label}.evidence[{position}]", root)
        for position, item in enumerate(_sequence(arm["evidence"], f"{label}.evidence"))
    ]
    if not evidence:
        raise AblationAggregationError(f"{label} must cite at least one evidence artifact")
    measurements: dict[str, dict[str, Any]] = {}
    for position, item in enumerate(_sequence(arm["measurements"], f"{label}.measurements")):
        measurement = _load_measurement(item, f"{label}.measurements[{position}]")
        if measurement["metric_id"] in measurements:
            raise AblationAggregationError(f"{label} repeats metric {measurement['metric_id']}")
        measurements[measurement["metric_id"]] = measurement
    if not measurements:
        raise AblationAggregationError(f"{label} must declare at least one measurement")

    failure_codes: dict[str, int] | None = None
    if "failure_codes" in arm:
        raw_codes = _mapping(arm["failure_codes"], f"{label}.failure_codes")
        failure_codes = {
            _text(code, f"{label}.failure_codes key"): _count(value, f"{label}.failure_codes[{code}]")
            for code, value in raw_codes.items()
        }

    cost_context: dict[str, Any] | None = None
    if "cost_context" in arm:
        context = _mapping(arm["cost_context"], f"{label}.cost_context")
        if set(context) != {"currency", "pricing_snapshot"}:
            raise AblationAggregationError(
                f"{label}.cost_context must declare currency and pricing_snapshot"
            )
        cost_context = {
            "currency": _text(context["currency"], f"{label}.cost_context.currency"),
            "pricing_snapshot": _text(context["pricing_snapshot"], f"{label}.cost_context.pricing_snapshot"),
        }
    # Only a currency-denominated cost needs a pricing snapshot. A token count
    # is provider-independent, and demanding a snapshot for it would push an
    # operator into inventing one, which is worse than having none.
    if cost_context is None and any(
        METRIC_DEFINITIONS[metric_id].unit == "currency" for metric_id in measurements
    ):
        raise AblationAggregationError(f"{label} reports a priced cost without cost_context")

    record.update(
        {
            "evidence": evidence,
            "measurements": measurements,
            "failure_codes": failure_codes,
            "cost_context": cost_context,
            "task_set_hash": (
                _text(arm["task_set_hash"], f"{label}.task_set_hash") if "task_set_hash" in arm else None
            ),
            "truth_preservation_gates": [
                _load_gate(item, f"{label}.truth_preservation_gates[{position}]")
                for position, item in enumerate(
                    _sequence(arm.get("truth_preservation_gates", []), f"{label}.truth_preservation_gates")
                )
            ],
        }
    )
    if "task_success_rate" in measurements and record["task_set_hash"] is None:
        raise AblationAggregationError(f"{label} reports task success without a task_set_hash")
    gate_ids = [gate["gate_id"] for gate in record["truth_preservation_gates"]]
    if len(gate_ids) != len(set(gate_ids)):
        raise AblationAggregationError(f"{label} repeats a truth-preservation gate id")
    return record


def load_ablation_aggregation_input(path: Path) -> dict[str, Any]:
    """Load and fully validate a ladder input, refusing incomparable arms."""
    resolved = path.expanduser().resolve()
    document = _load_json_document(resolved, "ablation aggregation input")
    expected = {"schema_version", "experiment_id", "baseline_arm_id", "policy", "arms"}
    if set(document) != expected:
        raise AblationAggregationError(f"ablation input must declare exactly {sorted(expected)}")
    if document["schema_version"] != ABLATION_AGGREGATION_VERSION:
        raise AblationAggregationError(
            f"unsupported ablation input schema_version: {document['schema_version']!r}"
        )
    arms = [
        _load_arm(item, index, resolved.parent)
        for index, item in enumerate(_sequence(document["arms"], "arms"))
    ]
    if len(arms) < 2:
        raise AblationAggregationError("an ablation needs a baseline and at least one arm")
    arm_ids = [arm["arm_id"] for arm in arms]
    if len(arm_ids) != len(set(arm_ids)):
        raise AblationAggregationError("arm_id values must be unique")
    baseline_id = _text(document["baseline_arm_id"], "baseline_arm_id")
    if baseline_id not in arm_ids:
        raise AblationAggregationError(f"baseline_arm_id {baseline_id!r} is not among the arms")
    baseline = next(arm for arm in arms if arm["arm_id"] == baseline_id)
    if baseline["status"] != "measured":
        raise AblationAggregationError("the baseline arm must be measured")
    return {
        "schema_version": document["schema_version"],
        "experiment_id": _text(document["experiment_id"], "experiment_id"),
        "baseline_arm_id": baseline_id,
        "policy": _load_policy(document["policy"]),
        "arms": arms,
        "input_hash": _file_hash(resolved),
        "input_path": resolved,
    }


def _oriented(delta: float, direction: str) -> float:
    return delta if direction == "higher_is_better" else -delta


def _classify(
    *,
    oriented_delta: float,
    material: bool,
    significant: bool | None,
) -> str:
    if significant is None:
        # A deterministic measurement is exact: there is no noise to rule out.
        if not material:
            return "no_material_change"
        return "material_improvement" if oriented_delta > 0 else "material_regression"
    if not significant:
        return "no_material_change" if not material else "inconclusive"
    if not material:
        return "no_material_change"
    return "material_improvement" if oriented_delta > 0 else "material_regression"


def _compare_metric(
    metric_id: str,
    arm: dict[str, Any] | None,
    baseline: dict[str, Any] | None,
    *,
    threshold: float,
    z: float,
) -> dict[str, Any]:
    definition = METRIC_DEFINITIONS[metric_id]
    record: dict[str, Any] = {
        "metric_id": metric_id,
        "family": definition.family,
        "direction": definition.direction,
        "unit": definition.unit,
        "practical_relative_threshold": threshold,
        "baseline_value": None,
        "arm_value": None,
        "absolute_delta": None,
        "relative_delta": None,
        "confidence_interval": None,
        "p_value": None,
        "adjusted_p_value": None,
        "practically_material": None,
        "verdict": "not_measured",
        "notes": [],
    }
    if arm is None or baseline is None:
        # Keep whichever side exists. A reviewer has to see what the missing
        # measurement would have been compared against.
        if baseline is not None:
            record["baseline_value"] = _round(float(baseline["value"]))
            record["notes"].append("measured only in the baseline")
        else:
            record["arm_value"] = _round(float(cast(dict[str, Any], arm)["value"]))
            record["notes"].append("measured only in the arm")
        return record
    if arm["kind"] != baseline["kind"]:
        raise AblationAggregationError(
            f"{metric_id} is measured as {baseline['kind']} in the baseline and {arm['kind']} in the arm"
        )

    baseline_value = float(baseline["value"])
    arm_value = float(arm["value"])
    delta = arm_value - baseline_value
    record["baseline_value"] = _round(baseline_value)
    record["arm_value"] = _round(arm_value)
    record["absolute_delta"] = _round(delta)
    if baseline_value != 0.0:
        record["relative_delta"] = _round(delta / abs(baseline_value))
    else:
        record["notes"].append("baseline is zero, so relative change is undefined")

    if record["relative_delta"] is None:
        material = delta != 0.0
    else:
        material = abs(cast(float, record["relative_delta"])) >= threshold
    record["practically_material"] = material

    significant: bool | None = None
    if arm["kind"] == "proportion":
        arm_pair = (arm["numerator"], arm["denominator"])
        base_pair = (baseline["numerator"], baseline["denominator"])
        low, high = newcombe_difference_interval(arm_pair, base_pair, z)
        record["confidence_interval"] = [_round(low), _round(high)]
        record["p_value"] = _round(two_proportion_test(arm_pair, base_pair))
        significant = not (low <= 0.0 <= high)
    elif arm["kind"] == "repeated":
        if len(arm["observations"]) < _MIN_TEST_SAMPLE or len(baseline["observations"]) < _MIN_TEST_SAMPLE:
            record["notes"].append(
                f"underpowered: a repeated-measure test needs at least {_MIN_TEST_SAMPLE} observations per arm"
            )
            record["verdict"] = "inconclusive"
            return record
        _, p_value, low, high = welch_test(arm["observations"], baseline["observations"])
        record["confidence_interval"] = [_round(low), _round(high)]
        record["p_value"] = _round(p_value)
        significant = not (low <= 0.0 <= high)
    else:
        record["notes"].append("deterministic measurement: the delta is exact, not sampled")

    record["verdict"] = _classify(
        oriented_delta=_oriented(delta, definition.direction),
        material=material,
        significant=significant,
    )
    return record


def _compare_failure_codes(
    arm: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    if arm["failure_codes"] is None or baseline["failure_codes"] is None:
        return {
            "status": "not_measured",
            "missing_in": [
                name
                for name, record in (("baseline", baseline), ("arm", arm))
                if record["failure_codes"] is None
            ],
            "codes": [],
        }
    codes = sorted(set(arm["failure_codes"]) | set(baseline["failure_codes"]))
    return {
        "status": "measured",
        "missing_in": [],
        "codes": [
            {
                "code": code,
                "baseline_count": baseline["failure_codes"].get(code, 0),
                "arm_count": arm["failure_codes"].get(code, 0),
                "delta": arm["failure_codes"].get(code, 0) - baseline["failure_codes"].get(code, 0),
            }
            for code in codes
        ],
    }


def _arm_local_failure_codes(arm: dict[str, Any]) -> dict[str, Any]:
    """Report an arm's own failure codes when no baseline delta is licensed.

    A degraded design still produces a real failure profile, and that profile is
    what tells a reviewer *how* the arm fails. Suppressing it because no delta
    exists would discard the only failure-code evidence in the ladder.
    """
    if arm["status"] != PARTIAL_STATUS or arm["failure_codes"] is None:
        return {"status": "not_measured", "missing_in": ["arm"], "codes": []}
    return {
        "status": "arm_only",
        "missing_in": ["baseline"],
        "codes": [
            {"code": code, "baseline_count": None, "arm_count": count, "delta": None}
            for code, count in sorted(arm["failure_codes"].items())
        ],
    }


def _gate_findings(arm: dict[str, Any]) -> dict[str, Any]:
    """Separate gates that could have failed from gates that could not."""
    informative: list[str] = []
    vacuous: list[str] = []
    failed: list[str] = []
    not_run: list[str] = []
    for gate in arm["truth_preservation_gates"]:
        if gate["verdict"] == "failed":
            failed.append(gate["gate_id"])
            continue
        if gate["verdict"] == "not_run":
            not_run.append(gate["gate_id"])
            continue
        if gate["sensitive_to_intervention"]:
            informative.append(gate["gate_id"])
        else:
            vacuous.append(gate["gate_id"])
    return {
        "informative_pass_gate_ids": informative,
        "vacuous_pass_gate_ids": vacuous,
        "failed_gate_ids": failed,
        "not_run_gate_ids": not_run,
    }


def _holm_adjust(comparisons: list[dict[str, Any]], alpha: float) -> None:
    """Apply Holm correction across every metric that produced a p-value."""
    tested = [item for item in comparisons if item["p_value"] is not None]
    order = sorted(
        range(len(tested)),
        key=lambda index: (tested[index]["p_value"], tested[index]["metric_id"]),
    )
    total = len(order)
    running = 0.0
    for rank, position in enumerate(order):
        item = tested[position]
        adjusted = min(1.0, cast(float, item["p_value"]) * (total - rank))
        running = max(running, adjusted)
        item["adjusted_p_value"] = _round(running)
    for item in tested:
        significant = cast(float, item["adjusted_p_value"]) < alpha
        if item["verdict"] in ("material_improvement", "material_regression") and not significant:
            item["verdict"] = "inconclusive"
            item["notes"].append("not significant after Holm correction across metric families")
        elif item["verdict"] == "inconclusive" and significant and item["practically_material"]:
            item["verdict"] = (
                "material_improvement"
                if _oriented(cast(float, item["absolute_delta"]), item["direction"]) > 0
                else "material_regression"
            )


def _recommend(arm: dict[str, Any], comparisons: list[dict[str, Any]], gates: dict[str, Any]) -> dict[str, Any]:
    """Derive a recommendation from a fixed policy, never from prose."""
    reasons: list[str] = []
    conditions: list[str] = []

    if arm["status"] != "measured":
        return {
            "recommendation": "insufficient_evidence",
            "reasons": [f"arm status is {arm['status']}: {arm['deferral_reason']}"],
            "conditions": [],
        }

    if gates["failed_gate_ids"]:
        reasons.append(f"truth-preservation gates failed: {gates['failed_gate_ids']}")
        return {"recommendation": "reject", "reasons": reasons, "conditions": []}

    regressions = [item["metric_id"] for item in comparisons if item["verdict"] == "material_regression"]
    improvements = [item["metric_id"] for item in comparisons if item["verdict"] == "material_improvement"]
    inconclusive = [item["metric_id"] for item in comparisons if item["verdict"] == "inconclusive"]
    unmeasured = [item["metric_id"] for item in comparisons if item["verdict"] == "not_measured"]

    if regressions:
        reasons.append(f"material regressions: {regressions}")
        return {"recommendation": "reject", "reasons": reasons, "conditions": []}

    if gates["vacuous_pass_gate_ids"]:
        conditions.append(
            "replace gates that cannot fail under this intervention: "
            f"{gates['vacuous_pass_gate_ids']}"
        )
    if gates["not_run_gate_ids"]:
        conditions.append(f"run the declared gates: {gates['not_run_gate_ids']}")
    if unmeasured:
        conditions.append(f"measure metrics missing from the comparison: {unmeasured}")
    if inconclusive:
        conditions.append(f"resolve inconclusive metrics: {inconclusive}")
    if not arm["truth_preservation_gates"]:
        conditions.append("declare a truth-preservation gate for this intervention")

    if not improvements:
        reasons.append("no metric improved materially against the baseline")
        return {
            "recommendation": "retain_baseline",
            "reasons": reasons,
            "conditions": conditions,
        }

    reasons.append(f"material improvements: {improvements}")
    if conditions:
        return {
            "recommendation": "adopt_with_conditions",
            "reasons": reasons,
            "conditions": conditions,
        }
    return {"recommendation": "adopt", "reasons": reasons, "conditions": []}


def _trade_offs(
    arm: dict[str, Any],
    comparisons: list[dict[str, Any]],
    gates: dict[str, Any],
    measured_families: set[str],
) -> dict[str, Any]:
    """State what each arm's gains were purchased with.

    SOV-866 asks for trade-offs, and a trade-off is not a limitation bullet: it
    is a gain named alongside its price. The price has three forms a reviewer
    keeps confusing — a measured regression, a gain whose truth was never
    verified, and a family nobody measured at all — so each is reported
    separately rather than merged into one prose sentence.
    """
    gains = [
        {"metric_id": item["metric_id"], "family": item["family"], "delta": item["absolute_delta"]}
        for item in comparisons
        if item["verdict"] == "material_improvement"
    ]
    measured_costs = [
        {"metric_id": item["metric_id"], "family": item["family"], "delta": item["absolute_delta"]}
        for item in comparisons
        if item["verdict"] == "material_regression"
    ]
    unverified = sorted(set(gates["not_run_gate_ids"]) | set(gates["vacuous_pass_gate_ids"]))
    unpriced = sorted(set(REQUIRED_FAMILIES) - measured_families)
    if gains and (measured_costs or unverified or unpriced):
        verdict = "gain_with_unpriced_risk" if not measured_costs else "gain_against_measured_cost"
    elif gains:
        verdict = "gain_with_no_observed_cost"
    elif measured_costs:
        verdict = "cost_without_gain"
    else:
        verdict = "no_gain_observed"
    return {
        "verdict": verdict,
        "gains": gains,
        "measured_costs": measured_costs,
        "unverified_by_gates": unverified,
        "unpriced_families": unpriced,
    }


def build_ablation_summary(inputs: AggregationInputs) -> dict[str, Any]:
    """Aggregate every arm against the baseline into a content-addressed report."""
    source = load_ablation_aggregation_input(inputs.ablation_input)
    policy = source["policy"]
    alpha = 1.0 - cast(float, policy["confidence_level"])
    z = critical_z(cast(float, policy["confidence_level"]))
    arms: list[dict[str, Any]] = source["arms"]
    baseline = next(arm for arm in arms if arm["arm_id"] == source["baseline_arm_id"])

    # A family only counts as covered when some arm can be compared to the
    # baseline in it. Arm-local numbers from a degraded design are reported
    # separately, because letting them close a coverage gap would turn a
    # withheld comparison into apparent readiness.
    measured_families = {
        METRIC_DEFINITIONS[metric_id].family
        for arm in arms
        if arm["status"] == "measured"
        for metric_id in arm["measurements"]
    }
    if any(arm["failure_codes"] is not None for arm in arms if arm["status"] == "measured"):
        measured_families.add("failure_codes")
    uncompared_families = {
        METRIC_DEFINITIONS[metric_id].family
        for arm in arms
        if arm["status"] == PARTIAL_STATUS
        for metric_id in arm["measurements"]
    }
    if any(arm["failure_codes"] is not None for arm in arms if arm["status"] == PARTIAL_STATUS):
        uncompared_families.add("failure_codes")
    uncompared_families -= measured_families

    results: list[dict[str, Any]] = []
    for arm in arms:
        if arm["arm_id"] == baseline["arm_id"]:
            continue
        comparisons: list[dict[str, Any]] = []
        if arm["status"] == "measured":
            if (
                "task_success_rate" in arm["measurements"]
                and arm["task_set_hash"] != baseline["task_set_hash"]
            ):
                raise AblationAggregationError(
                    f"{arm['arm_id']} scores a different task set than the baseline; "
                    "declare a common intersection before comparing task success"
                )
            if arm["cost_context"] is not None and baseline["cost_context"] is not None:
                if arm["cost_context"] != baseline["cost_context"]:
                    raise AblationAggregationError(
                        f"{arm['arm_id']} cost is priced differently than the baseline"
                    )
            for metric_id in sorted(set(arm["measurements"]) | set(baseline["measurements"])):
                comparisons.append(
                    _compare_metric(
                        metric_id,
                        arm["measurements"].get(metric_id),
                        baseline["measurements"].get(metric_id),
                        threshold=policy["resolved_thresholds"][metric_id],
                        z=z,
                    )
                )
            _holm_adjust(comparisons, alpha)
        gates = _gate_findings(arm)
        results.append(
            {
                "arm_id": arm["arm_id"],
                "title": arm["title"],
                "ticket": arm["ticket"],
                "status": arm["status"],
                "intervention": arm["intervention"],
                "evidence": arm["evidence"],
                "limitations": arm["limitations"],
                "truth_preservation": gates | {"declared": arm["truth_preservation_gates"]},
                "comparisons": comparisons,
                "arm_local_measurements": (
                    [
                        {
                            "metric_id": metric_id,
                            "family": METRIC_DEFINITIONS[metric_id].family,
                            "kind": measurement["kind"],
                            "value": _round(float(measurement["value"])),
                            "comparison_withheld_because": arm["deferral_reason"],
                        }
                        for metric_id, measurement in sorted(arm["measurements"].items())
                    ]
                    if arm["status"] == PARTIAL_STATUS
                    else []
                ),
                "failure_codes": (
                    _compare_failure_codes(arm, baseline)
                    if arm["status"] == "measured"
                    else _arm_local_failure_codes(arm)
                ),
                "recommendation": _recommend(arm, comparisons, gates),
                "trade_offs": _trade_offs(arm, comparisons, gates, measured_families),
            }
        )

    recommendations = {item["arm_id"]: item["recommendation"]["recommendation"] for item in results}
    report: dict[str, Any] = {
        "schema_version": ABLATION_AGGREGATION_VERSION,
        "report_hash": None,
        "source": {
            "experiment_id": source["experiment_id"],
            "baseline_arm_id": baseline["arm_id"],
            "input_file": source["input_path"].name,
            "input_hash": source["input_hash"],
            "arm_ids": [arm["arm_id"] for arm in arms],
        },
        "policy": {
            "confidence_level": policy["confidence_level"],
            "multiple_comparison": policy["multiple_comparison"],
            "practical_threshold_overrides": policy["practical_threshold_overrides"],
            "causal_claim": False,
            "claim_boundary": (
                "Deltas are measured against one declared baseline arm. A material change "
                "identifies the arm that produced it, not a causal mechanism."
            ),
        },
        "baseline": {
            "arm_id": baseline["arm_id"],
            "title": baseline["title"],
            "ticket": baseline["ticket"],
            "evidence": baseline["evidence"],
            "measurements": [
                {
                    "metric_id": metric_id,
                    "family": METRIC_DEFINITIONS[metric_id].family,
                    "kind": measurement["kind"],
                    "value": _round(float(measurement["value"])),
                }
                for metric_id, measurement in sorted(baseline["measurements"].items())
            ],
            "limitations": baseline["limitations"],
        },
        "arms": results,
        "coverage": {
            "required_families": list(REQUIRED_FAMILIES),
            "measured_families": sorted(measured_families & set(REQUIRED_FAMILIES)),
            "unmeasured_families": sorted(set(REQUIRED_FAMILIES) - measured_families),
            "families_measured_without_comparison": sorted(
                uncompared_families & set(REQUIRED_FAMILIES)
            ),
        },
        "summary": {
            "arms_compared": len(results),
            "arms_measured": sum(item["status"] == "measured" for item in results),
            "arms_partially_measured": sum(item["status"] == PARTIAL_STATUS for item in results),
            "recommendations": recommendations,
            "adopt_arm_ids": sorted(k for k, v in recommendations.items() if v == "adopt"),
            "conditional_arm_ids": sorted(
                k for k, v in recommendations.items() if v == "adopt_with_conditions"
            ),
            "rejected_arm_ids": sorted(k for k, v in recommendations.items() if v == "reject"),
            "insufficient_evidence_arm_ids": sorted(
                k for k, v in recommendations.items() if v == "insufficient_evidence"
            ),
            "release_readiness": (
                "blocked"
                if any(v == "reject" for v in recommendations.values())
                else "incomplete"
                if any(v == "insufficient_evidence" for v in recommendations.values())
                or set(REQUIRED_FAMILIES) - measured_families
                else "ready"
            ),
        },
    }
    report["report_hash"] = _semantic_hash({key: value for key, value in report.items() if key != "report_hash"})
    validate_ablation_summary(report)
    return report


def validate_ablation_summary(report: Mapping[str, Any]) -> None:
    """Re-derive every claim a summary makes about itself."""
    document = _mapping(dict(report), "ablation summary")
    if document.get("schema_version") != ABLATION_AGGREGATION_VERSION:
        raise AblationAggregationError("ablation summary schema_version is unsupported")
    claimed = document.get("report_hash")
    if not isinstance(claimed, str) or not claimed.startswith("sha256:"):
        raise AblationAggregationError("ablation summary report_hash must be sha256:<64 hex>")
    unsigned = {key: value for key, value in document.items() if key != "report_hash"}
    if claimed != _semantic_hash(unsigned):
        raise AblationAggregationError("ablation summary report_hash mismatch")
    summary = _mapping(document.get("summary"), "ablation summary.summary")
    arms = _sequence(document.get("arms"), "ablation summary.arms")
    recommendations = _mapping(summary.get("recommendations"), "summary.recommendations")
    if recommendations != {
        arm["arm_id"]: arm["recommendation"]["recommendation"] for arm in arms
    }:
        raise AblationAggregationError("summary recommendations disagree with the arm records")
    for arm in arms:
        recommendation = arm["recommendation"]["recommendation"]
        if recommendation not in RECOMMENDATIONS:
            raise AblationAggregationError(f"unsupported recommendation: {recommendation}")
        for comparison in arm["comparisons"]:
            if comparison["verdict"] not in COMPARISON_VERDICTS:
                raise AblationAggregationError(f"unsupported verdict: {comparison['verdict']}")
        if arm["status"] == PARTIAL_STATUS:
            # The whole point of the status is that no delta was licensed, so a
            # partial arm must never carry one, nor a recommendation earned from one.
            if arm["comparisons"]:
                raise AblationAggregationError(
                    f"{arm['arm_id']} is partially measured but publishes a baseline comparison"
                )
            if not arm["arm_local_measurements"]:
                raise AblationAggregationError(
                    f"{arm['arm_id']} is partially measured but reports no arm-local measurement"
                )
            if recommendation != "insufficient_evidence":
                raise AblationAggregationError(
                    f"{arm['arm_id']} is partially measured and cannot recommend {recommendation}"
                )
        elif arm["arm_local_measurements"]:
            raise AblationAggregationError(
                f"{arm['arm_id']} is {arm['status']} and must not report arm-local measurements"
            )
        trade = _mapping(arm.get("trade_offs"), f"{arm['arm_id']}.trade_offs")
        if trade.get("verdict") not in TRADE_OFF_VERDICTS:
            raise AblationAggregationError(f"unsupported trade-off verdict: {trade.get('verdict')}")
        # A recommendation that claims a gain has to name one, or the summary is
        # asserting value the comparisons never produced.
        if recommendation in ("adopt", "adopt_with_conditions") and not trade["gains"]:
            raise AblationAggregationError(
                f"{arm['arm_id']} recommends {recommendation} without a single material gain"
            )
    if _mapping(document.get("policy"), "summary.policy").get("causal_claim") is not False:
        raise AblationAggregationError("this contract cannot publish a causal claim")


def _display(value: Any) -> str:
    """Render a metric value without turning a count into a float."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def render_ablation_summary_markdown(report: Mapping[str, Any]) -> str:
    """Render a reviewer-facing recommendation from a validated summary."""
    validate_ablation_summary(report)
    source = _mapping(report["source"], "summary source")
    summary = _mapping(report["summary"], "summary")
    coverage = _mapping(report["coverage"], "summary coverage")
    lines = [
        "# BFCL ablation summary and release recommendation",
        "",
        f"- Report hash: `{report['report_hash']}`",
        f"- Experiment: `{source['experiment_id']}`",
        f"- Baseline arm: `{source['baseline_arm_id']}`",
        f"- Input hash: `{source['input_hash']}`",
        f"- Release readiness: **{summary['release_readiness']}**",
        "",
        "## Recommendation by arm",
        "",
        "| Arm | Title | Ticket | Status | Recommendation |",
        "|---|---|---|---|---|",
    ]
    for arm in report["arms"]:
        lines.append(
            "| {arm_id} | {title} | {ticket} | {status} | **{recommendation}** |".format(
                arm_id=arm["arm_id"],
                title=str(arm["title"]).replace("|", "\\|"),
                ticket=arm["ticket"],
                status=arm["status"],
                recommendation=arm["recommendation"]["recommendation"],
            )
        )
    lines.extend(
        [
            "",
            "## Metric families",
            "",
            "| Family | Compared to baseline | Numbers on record |",
            "|---|---|---|",
        ]
    )
    for family in coverage["required_families"]:
        compared = family in coverage["measured_families"]
        arm_only = family in coverage["families_measured_without_comparison"]
        lines.append(
            "| `{family}` | {compared} | {numbers} |".format(
                family=family,
                compared="yes" if compared else "no",
                numbers=(
                    "yes"
                    if compared
                    else "yes, arm-local only"
                    if arm_only
                    else "none"
                ),
            )
        )
    lines.extend(["", "## Arm detail", ""])
    for arm in report["arms"]:
        lines.extend([f"### {arm['arm_id']} — {arm['title']}", "", f"- Intervention: {arm['intervention']}"])
        for reason in arm["recommendation"]["reasons"]:
            lines.append(f"- Reason: {reason}")
        for condition in arm["recommendation"]["conditions"]:
            lines.append(f"- Condition: {condition}")
        truth = arm["truth_preservation"]
        if truth["vacuous_pass_gate_ids"]:
            lines.append(
                "- Gates that passed but could not fail under this intervention: "
                f"{truth['vacuous_pass_gate_ids']}"
            )
        if truth["failed_gate_ids"]:
            lines.append(f"- Failed gates: {truth['failed_gate_ids']}")
        if arm["comparisons"]:
            lines.extend(
                [
                    "",
                    "| Metric | Family | Baseline | Arm | Delta | 95% CI | Adjusted p | Verdict |",
                    "|---|---|---:|---:|---:|---|---:|---|",
                ]
            )
            for item in arm["comparisons"]:
                interval = item["confidence_interval"]
                lines.append(
                    "| `{metric}` | {family} | {baseline} | {arm} | {delta} | {ci} | {p} | {verdict} |".format(
                        metric=item["metric_id"],
                        family=item["family"],
                        baseline="n/a" if item["baseline_value"] is None else item["baseline_value"],
                        arm="n/a" if item["arm_value"] is None else item["arm_value"],
                        delta="n/a" if item["absolute_delta"] is None else item["absolute_delta"],
                        ci="n/a" if interval is None else f"[{interval[0]}, {interval[1]}]",
                        p="n/a" if item["adjusted_p_value"] is None else item["adjusted_p_value"],
                        verdict=item["verdict"],
                    )
                )
        if arm["arm_local_measurements"]:
            lines.extend(
                [
                    "",
                    "Measured in this arm. The design does not license a delta against the "
                    "baseline, because "
                    f"{arm['arm_local_measurements'][0]['comparison_withheld_because']}",
                    "",
                    "| Metric | Family | Unit | Arm value |",
                    "|---|---|---|---:|",
                ]
            )
            for item in arm["arm_local_measurements"]:
                lines.append(
                    "| `{metric}` | {family} | {unit} | {value} |".format(
                        metric=item["metric_id"],
                        family=item["family"],
                        unit=METRIC_DEFINITIONS[item["metric_id"]].unit,
                        value=_display(item["value"]),
                    )
                )
        failure = arm["failure_codes"]
        if failure["codes"]:
            if failure["status"] == "arm_only":
                lines.append("")
                lines.append(
                    "Failure profile measured in this arm; the baseline has none to compare "
                    "against:"
                )
            lines.extend(["", "| Failure code | Baseline | Arm | Delta |", "|---|---:|---:|---:|"])
            for code in failure["codes"]:
                lines.append(
                    "| `{code}` | {baseline} | {arm} | {delta} |".format(
                        code=code["code"],
                        baseline="n/a" if code["baseline_count"] is None else code["baseline_count"],
                        arm=code["arm_count"],
                        delta="n/a" if code["delta"] is None else code["delta"],
                    )
                )
        trade = arm["trade_offs"]
        lines.extend(["", f"Trade-off: **{trade['verdict']}**", ""])
        for label, entries in (("Gain", trade["gains"]), ("Measured cost", trade["measured_costs"])):
            for entry in entries:
                definition = METRIC_DEFINITIONS[entry["metric_id"]]
                lines.append(
                    f"- {label}: `{entry['metric_id']}` ({entry['family']}, "
                    f"{definition.direction}) by {_display(entry['delta'])} "
                    f"{definition.unit}"
                )
        if trade["unverified_by_gates"]:
            lines.append(
                "- Bought with unverified truth: gates that did not run or could not fail "
                f"under this intervention: {trade['unverified_by_gates']}"
            )
        if trade["unpriced_families"]:
            lines.append(
                "- Unpriced: no arm in this ladder can be compared to the baseline in "
                f"{trade['unpriced_families']}, so any price paid there is invisible"
            )
        if arm["limitations"]:
            lines.append("")
            for limitation in arm["limitations"]:
                lines.append(f"- Limitation: {limitation}")
        lines.append("")
    lines.extend(
        [
            "## Claim boundary",
            "",
            f"- {report['policy']['claim_boundary']}",
            f"- Confidence level: {report['policy']['confidence_level']}",
            f"- Multiple-comparison correction: {report['policy']['multiple_comparison']}",
            "- Causal claim: no.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_ablation_summary(
    report: Mapping[str, Any],
    output_dir: Path,
) -> tuple[Path, Path]:
    """Write JSON and Markdown atomically; never replace different summary bytes."""
    validate_ablation_summary(report)
    root = output_dir.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    json_bytes = (
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    markdown_bytes = render_ablation_summary_markdown(report).encode("utf-8")
    json_path = root / "ablation_summary.json"
    markdown_path = root / "ablation_summary.md"
    for path, content in ((json_path, json_bytes), (markdown_path, markdown_bytes)):
        if path.exists():
            if not path.is_file() or path.read_bytes() != content:
                raise AblationAggregationError(f"refusing to replace a different summary: {path}")
            continue
        temporary = path.with_suffix(path.suffix + ".tmp")
        try:
            temporary.write_bytes(content)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    return json_path, markdown_path


__all__ = [
    "ABLATION_AGGREGATION_VERSION",
    "AblationAggregationError",
    "AggregationInputs",
    "METRIC_DEFINITIONS",
    "REQUIRED_FAMILIES",
    "build_ablation_summary",
    "load_ablation_aggregation_input",
    "render_ablation_summary_markdown",
    "validate_ablation_summary",
    "write_ablation_summary",
]
