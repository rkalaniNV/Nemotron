"""One declaration of the raw files that make up a domain's ablation evidence.

The reviewer and the operator have to be looking at the same thing, and neither
can be trusted to describe it in prose. A bundle manifest names every raw file by
a path relative to itself, so the reviewer verifies a directory and signs its
digests, and publication later resolves the same manifest and recomputes them.

Paths are relative on purpose: an absolute path records where a checkout happened
to live, which is not part of the evidence and would make the same bundle
unverifiable on another machine. Escaping the manifest's directory is refused for
the same reason a run artifact may not contain a symlink — evidence has to be a
self-contained tree someone else can re-verify.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, StrictStr, model_validator

from nemotron.steps.byob.runtime.mcp.ablation import AblationError
from nemotron.steps.byob.runtime.mcp.ablation_evaluator_pin import EvaluatorPin, load_evaluator_pin
from nemotron.steps.byob.runtime.mcp.ablation_rollout import RunExclusion

DOMAIN_BUNDLE_VERSION = "bfcl-onboarding-domain-bundle-v1"
_SCHEDULE: tuple[tuple[str, int], ...] = (
    ("manual", 1),
    ("llm_backend", 1),
    ("llm_mcp", 1),
    ("llm_mcp", 2),
    ("manual", 2),
    ("llm_backend", 2),
    ("llm_backend", 3),
    ("llm_mcp", 3),
    ("manual", 3),
)


class DomainBundleError(AblationError):
    """Raised when a bundle manifest does not describe a verifiable evidence tree."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class BundleRun(_StrictModel):
    sequence: StrictInt = Field(ge=1, le=9)
    observation: StrictStr
    state: StrictStr
    run_artifact: StrictStr
    excluded_authoring_minutes: StrictFloat = Field(default=0.0, ge=0)
    excluded_review_minutes: StrictFloat = Field(default=0.0, ge=0)
    exclusion_reason: StrictStr | None = None


class DomainBundleManifest(_StrictModel):
    schema_version: Literal["bfcl-onboarding-domain-bundle-v1"]
    domain_id: StrictStr
    operator_identity: StrictStr
    protocol: StrictStr
    ablation_input: StrictStr
    ablation_report: StrictStr
    evaluator_pin: StrictStr
    runs: tuple[BundleRun, ...]
    # The signed attestation may be named here because a signature vouches for
    # itself. Which reviewer key to trust may not: that decision belongs to
    # whoever publishes the rollout, not to the operator who wrote this manifest.
    review_attestation: StrictStr | None = None

    @model_validator(mode="after")
    def validate_manifest(self) -> DomainBundleManifest:
        sequences = [run.sequence for run in self.runs]
        if sequences != list(range(1, 10)):
            raise ValueError("bundle runs must list sequences 1 through 9 in ascending order")
        return self


@dataclass(frozen=True)
class LoadedDomainBundle:
    """A manifest with every declared path resolved and every exclusion parsed."""

    domain_id: str
    operator_identity: str
    protocol_path: Path
    ablation_input_path: Path
    ablation_report_path: Path
    observation_paths: list[Path]
    state_paths: list[Path]
    run_artifact_paths: list[Path]
    exclusions: list[RunExclusion]
    evaluator_pin: EvaluatorPin
    review_attestation_path: Path | None


def load_domain_bundle(path: Path) -> LoadedDomainBundle:
    """Resolve a bundle manifest, refusing paths that leave its own directory."""
    manifest_path = path.resolve()
    root = manifest_path.parent

    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        document: dict[str, Any] = {}
        for key, value in pairs:
            if key in document:
                raise DomainBundleError(f"bundle manifest repeats JSON key {key!r}")
            document[key] = value
        return document

    def reject_constant(token: str) -> None:
        raise DomainBundleError(f"bundle manifest contains non-finite constant {token}")

    try:
        document = json.loads(
            manifest_path.read_text(encoding="utf-8"),
            object_pairs_hook=unique_pairs,
            parse_constant=reject_constant,
        )
        manifest = cast(DomainBundleManifest, DomainBundleManifest.model_validate(document))
    except DomainBundleError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise DomainBundleError(f"cannot load bundle manifest {manifest_path}: {exc}") from exc

    exclusions: list[RunExclusion] = []
    for run in manifest.runs:
        flow, repetition = _SCHEDULE[run.sequence - 1]
        try:
            exclusions.append(
                cast(
                    RunExclusion,
                    RunExclusion.model_validate(
                        {
                            "flow": flow,
                            "repetition": repetition,
                            "sequence": run.sequence,
                            "excluded_authoring_minutes": run.excluded_authoring_minutes,
                            "excluded_review_minutes": run.excluded_review_minutes,
                            "reason": run.exclusion_reason,
                        }
                    ),
                )
            )
        except ValueError as exc:
            raise DomainBundleError(f"invalid exclusion for sequence {run.sequence}: {exc}") from exc

    return LoadedDomainBundle(
        domain_id=manifest.domain_id,
        operator_identity=manifest.operator_identity,
        protocol_path=_resolve(root, manifest.protocol, "protocol"),
        ablation_input_path=_resolve(root, manifest.ablation_input, "ablation_input"),
        ablation_report_path=_resolve(root, manifest.ablation_report, "ablation_report"),
        observation_paths=[_resolve(root, run.observation, "observation") for run in manifest.runs],
        state_paths=[_resolve(root, run.state, "state") for run in manifest.runs],
        run_artifact_paths=[_resolve(root, run.run_artifact, "run_artifact") for run in manifest.runs],
        exclusions=exclusions,
        evaluator_pin=_load_pin(_resolve(root, manifest.evaluator_pin, "evaluator_pin")),
        review_attestation_path=(
            None
            if manifest.review_attestation is None
            else _resolve(root, manifest.review_attestation, "review_attestation")
        ),
    )


def _resolve(root: Path, declared: str, label: str) -> Path:
    candidate = Path(declared)
    if candidate.is_absolute():
        raise DomainBundleError(f"bundle {label} path must be relative to the manifest")
    resolved = (root / candidate).resolve()
    if not resolved.is_relative_to(root):
        raise DomainBundleError(f"bundle {label} path must stay inside the manifest directory")
    if not resolved.exists():
        raise DomainBundleError(f"bundle {label} path does not exist: {resolved}")
    return resolved


def _load_pin(path: Path) -> EvaluatorPin:
    try:
        return load_evaluator_pin(path)
    except ValueError as exc:
        raise DomainBundleError(f"bundle evaluator pin is unusable: {exc}") from exc
