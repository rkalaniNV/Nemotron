"""Artifact discovery and provenance for the A7 meta-evaluation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from bfcl_ablation import common
from bfcl_ablation.quality_gate.schema import ThresholdPolicy


@dataclass(frozen=True)
class ArtifactSpec:
    key: str
    relative_path: str
    required: bool = True


ARTIFACT_SPECS = (
    ArtifactSpec("a0_metrics", "A0/metrics.json"),
    ArtifactSpec("budget_sweep", "budget_sweep.json", required=False),
    ArtifactSpec("a1_metrics", "A1/metrics.json"),
    ArtifactSpec("a1_equivalence", "A1/vs_a0_equivalence.json"),
    ArtifactSpec("a2_metrics", "A2/metrics.json"),
    ArtifactSpec("a3_metrics", "A3/metrics.json"),
    ArtifactSpec("a4_metrics", "A4/metrics.json"),
    ArtifactSpec("a4_trials", "A4/trials.json"),
    ArtifactSpec("a5_metrics", "A5/metrics.json"),
    ArtifactSpec("a5_trials", "A5/trials.json"),
    ArtifactSpec("a6_metrics", "A6/metrics.json"),
    ArtifactSpec("a6_trials", "A6/trials.json"),
    ArtifactSpec("a6_triage", "A6/triage.json"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _schema_errors(spec: ArtifactSpec, data: Any) -> list[str]:
    errors: list[str] = []
    if spec.key.endswith("_metrics"):
        if not isinstance(data, dict):
            return ["metrics root must be an object"]
        expected_arm = spec.key.removesuffix("_metrics")
        if data.get("arm") != expected_arm:
            errors.append(f"arm must be {expected_arm!r}")
        if not isinstance(data.get("metrics_version"), str):
            errors.append("metrics_version must be a string")
    elif spec.key == "a1_equivalence":
        if not isinstance(data, dict):
            errors.append("equivalence root must be an object")
        elif data.get("baseline_arm") != "a0" or data.get("candidate_arm") != "a1":
            errors.append("equivalence lineage must be a0 -> a1")
    elif spec.key == "a4_trials":
        if not isinstance(data, dict) or not isinstance(data.get("human"), list):
            errors.append("A4 trials must be an object containing a human row list")
    elif spec.key in {"a5_trials", "a6_trials", "budget_sweep"} and not isinstance(data, list):
        errors.append(f"{spec.key} must be a row list")
    elif spec.key == "a6_triage":
        if not isinstance(data, dict):
            errors.append("A6 triage root must be an object")
        elif not isinstance(data.get("counts"), dict) or not isinstance(data.get("verdicts"), list):
            errors.append("A6 triage requires counts and verdicts")
    return errors


def load_artifacts(results_root: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Load every frozen input without treating a missing file as a pass."""
    payloads: dict[str, Any] = {}
    inventory: dict[str, dict[str, Any]] = {}
    for spec in ARTIFACT_SPECS:
        path = results_root / spec.relative_path
        record: dict[str, Any] = {
            "path": common.rel(path),
            "required": spec.required,
            "present": path.is_file(),
        }
        if not path.is_file():
            record["error"] = "missing"
            inventory[spec.key] = record
            payloads[spec.key] = None
            continue
        try:
            payloads[spec.key] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            payloads[spec.key] = None
            record["error"] = f"{type(exc).__name__}: {exc}"
        else:
            record["sha256"] = sha256_file(path)
            data = payloads[spec.key]
            schema_errors = _schema_errors(spec, data)
            record["schema_valid"] = not schema_errors
            if schema_errors:
                record["error"] = "schema: " + "; ".join(schema_errors)
            if spec.key.endswith("_metrics") and isinstance(data, dict):
                record["arm"] = data.get("arm")
                record["metrics_version"] = data.get("metrics_version")
        inventory[spec.key] = record
    return payloads, inventory


def load_thresholds(path: Path) -> tuple[ThresholdPolicy, dict[str, Any]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    policy = ThresholdPolicy.model_validate(raw)
    provenance = {
        "path": common.rel(path),
        "sha256": sha256_file(path),
        "canonical_sha256": canonical_hash(policy.model_dump(mode="json")),
        "contract_version": policy.contract_version,
    }
    return policy, provenance


def reported_path_exists(value: Any) -> bool | None:
    """Resolve a path recorded by an arm against the repository root."""
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value)
    if not path.is_absolute():
        path = common.REPO_ROOT / path
    return path.exists()
