"""Which weights produced an ablation score, or an explicit record that none did.

A rollout that compares three authoring flows on the same domain only carries a
score claim when every run was scored by the same weights. A serving route names
where a request went, not which bytes answered it, so a route alone cannot close
a comparison: the same route serves different weights next month and the same
config would report a different number.

The authority for "these weights cannot move" already exists in the evaluation
config contract, so this module does not restate it. A pinned evaluator is
validated by constructing :class:`CandidateModelIdentity`, which refuses moving
pointers such as ``main`` or ``latest``. That contract also allows an identity
that pins nothing, since an evaluation may knowingly score a provider-managed
route; a record whose status says ``pinned`` may not, so this module requires the
constructed identity to be ``weights_pinned``. On top of that it adds what the
ablation needs: the non-secret serving route the operator actually called, the
name of the environment variable that held the credential, and a digest of the
provider evidence the pin was read from.

An unpinned evaluator is a first-class record rather than a gap to be filled in
later with a plausible string. It states why no pinned identity exists and keeps
the route visible, so the rollout can publish ``target_model_pin_missing``
instead of implying a score it cannot support.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal, cast
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, StrictStr, model_validator

from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.errors import EvalConfigError
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.schemas import (
    CandidateModelIdentity,
)
from nemotron.steps.byob.runtime.mcp.ablation import AblationError
from nemotron.steps.byob.runtime.pack_authoring.artifacts import (
    sha256_json,
    write_canonical_json,
)

EVALUATOR_PIN_VERSION = "bfcl-onboarding-evaluator-pin-v1"

# The literal ``evaluator_model`` an ablation input carries when the target model
# was never called. It is not a model name, and it must never be mistaken for one.
NOT_RUN_EVALUATOR = "not_run"

_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]{2,63}$")
_ROUTE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@/+~-]{2,255}$")

# Credential shapes that must never reach evidence bytes, because a pin record is
# published and a leaked key would be leaked permanently and in public. Each
# alternative requires key-shaped material rather than a substring, so an ordinary
# model or route name is not mistaken for a secret.
_SECRET_PATTERN = re.compile(
    r"nvapi-[0-9A-Za-z_-]{8}"
    r"|sk-[0-9A-Za-z]{16}"
    r"|ghp_[0-9A-Za-z]{16}"
    r"|-----BEGIN"
    r"|(?:api[-_]?key|access[-_]?token|authorization|bearer)\s*[=:]\s*\S",
    re.IGNORECASE,
)


class EvaluatorPinError(AblationError):
    """Raised when an evaluator identity cannot support a comparable score."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _reject_secret_material(value: Any, field: str) -> None:
    if isinstance(value, str) and _SECRET_PATTERN.search(value) is not None:
        raise ValueError(f"{field} looks like credential material and cannot enter evidence")


class PinnedEvaluator(_StrictModel):
    """A serving route plus the immutable weight identity that answered on it."""

    schema_version: Literal["bfcl-onboarding-evaluator-pin-v1"]
    status: Literal["pinned"]
    provider: StrictStr
    served_model: StrictStr
    api_base: StrictStr
    credential_env_var: StrictStr
    weight_source: StrictStr
    weight_model: StrictStr
    revision: StrictStr | None
    weights_digest: StrictStr | None
    pin_evidence_digest: StrictStr
    canonical_id: StrictStr
    pin_digest: StrictStr

    @model_validator(mode="after")
    def validate_pin(self) -> PinnedEvaluator:
        for field in (
            "provider",
            "served_model",
            "api_base",
            "credential_env_var",
            "weight_source",
            "weight_model",
            "revision",
            "weights_digest",
            "canonical_id",
        ):
            _reject_secret_material(getattr(self, field), field)
        if _ENV_NAME.fullmatch(self.credential_env_var) is None:
            raise ValueError("credential_env_var must be an environment variable name, never its value")
        _validate_api_base(self.api_base)
        for field in ("provider", "served_model"):
            if _ROUTE.fullmatch(str(getattr(self, field))) is None:
                raise ValueError(f"{field} must be a stable non-secret route component")
        if _DIGEST.fullmatch(self.pin_evidence_digest) is None:
            raise ValueError("pin_evidence_digest must be sha256:<64 lowercase hex>")
        identity = _immutable_identity(
            weight_source=self.weight_source,
            weight_model=self.weight_model,
            revision=self.revision,
            weights_digest=self.weights_digest,
        )
        if self.canonical_id != identity.canonical_id:
            raise ValueError("canonical_id must be derived from the immutable weight identity")
        _verify_self_digest(self)
        return self

    @property
    def evaluator_model(self) -> str:
        """The ``evaluator_model`` string an ablation input must carry for this pin."""
        return self.canonical_id

    @property
    def route(self) -> str:
        return f"{self.provider}/{self.served_model}"


class UnpinnedEvaluator(_StrictModel):
    """An explicit record that no immutable evaluator identity backs this domain."""

    schema_version: Literal["bfcl-onboarding-evaluator-pin-v1"]
    status: Literal["unpinned"]
    reason_code: Literal["target_evaluation_not_run", "immutable_pin_unavailable"]
    declared_route: StrictStr | None
    detail: StrictStr
    pin_digest: StrictStr

    @model_validator(mode="after")
    def validate_unpinned(self) -> UnpinnedEvaluator:
        for field in ("declared_route", "detail"):
            _reject_secret_material(getattr(self, field), field)
        if not self.detail.strip():
            raise ValueError("an unpinned evaluator must state why no pin exists")
        if self.reason_code == "target_evaluation_not_run":
            if self.declared_route is not None:
                raise ValueError("a target model that was never called cannot declare a serving route")
        else:
            if self.declared_route is None:
                raise ValueError("immutable_pin_unavailable must name the route that lacks a pin")
            if _ROUTE.fullmatch(self.declared_route) is None:
                raise ValueError("declared_route must be a stable non-secret route")
        _verify_self_digest(self)
        return self

    @property
    def evaluator_model(self) -> str:
        """The ``evaluator_model`` string an ablation input must carry for this record."""
        if self.reason_code == "target_evaluation_not_run":
            return NOT_RUN_EVALUATOR
        return cast(str, self.declared_route)


EvaluatorPin = PinnedEvaluator | UnpinnedEvaluator


def _validate_api_base(value: str) -> None:
    parts = urlsplit(value)
    if parts.scheme != "https":
        raise ValueError("api_base must be an https endpoint")
    if not parts.hostname:
        raise ValueError("api_base must name a host")
    if parts.username or parts.password:
        raise ValueError("api_base must not embed credentials")
    if parts.query or parts.fragment:
        raise ValueError("api_base must not carry a query string or fragment")


def _immutable_identity(
    *,
    weight_source: str,
    weight_model: str,
    revision: str | None,
    weights_digest: str | None,
) -> CandidateModelIdentity:
    """Validate weight identity through the evaluation config contract, not a local rule."""
    try:
        identity = cast(
            CandidateModelIdentity,
            CandidateModelIdentity.model_validate(
                {
                    "source": weight_source,
                    "model": weight_model,
                    "revision": revision,
                    "weights_digest": weights_digest,
                }
            ),
        )
    except (EvalConfigError, ValueError) as exc:
        raise EvaluatorPinError(f"evaluator weights are not immutably pinned: {exc}") from exc
    # The eval contract accepts a route no provider pins and records it as
    # provider_managed, because scoring a hosted model is a legitimate run that
    # simply cannot publish. A *pinned* evaluator has no such middle ground: the
    # record's own status would be a lie, and the unpinned record below is the
    # place that case belongs.
    if identity.assurance != "weights_pinned":
        raise EvaluatorPinError(
            "evaluator weights are not immutably pinned: neither revision nor weights_digest is set; "
            "record an unpinned evaluator with reason immutable_pin_unavailable instead"
        )
    return identity


def _verify_self_digest(pin: PinnedEvaluator | UnpinnedEvaluator) -> None:
    unsigned = pin.model_dump(mode="json", exclude={"pin_digest"})
    if pin.pin_digest != sha256_json(unsigned):
        raise ValueError("evaluator pin_digest mismatch")


def build_pinned_evaluator(
    *,
    provider: str,
    served_model: str,
    api_base: str,
    credential_env_var: str,
    weight_source: str,
    weight_model: str,
    pin_evidence_digest: str,
    revision: str | None = None,
    weights_digest: str | None = None,
) -> PinnedEvaluator:
    """Bind a serving route to weights that cannot move, or refuse to bind at all."""
    identity = _immutable_identity(
        weight_source=weight_source,
        weight_model=weight_model,
        revision=revision,
        weights_digest=weights_digest,
    )
    payload: dict[str, Any] = {
        "schema_version": EVALUATOR_PIN_VERSION,
        "status": "pinned",
        "provider": provider,
        "served_model": served_model,
        "api_base": api_base,
        "credential_env_var": credential_env_var,
        "weight_source": weight_source,
        "weight_model": weight_model,
        "revision": revision,
        "weights_digest": weights_digest,
        "pin_evidence_digest": pin_evidence_digest,
        "canonical_id": identity.canonical_id,
    }
    payload["pin_digest"] = sha256_json(payload)
    try:
        return cast(PinnedEvaluator, PinnedEvaluator.model_validate(payload))
    except ValueError as exc:
        raise EvaluatorPinError(f"invalid pinned evaluator: {exc}") from exc


def build_unpinned_evaluator(
    *,
    reason_code: Literal["target_evaluation_not_run", "immutable_pin_unavailable"],
    detail: str,
    declared_route: str | None = None,
) -> UnpinnedEvaluator:
    """Record the absence of a pin as evidence instead of leaving it to be guessed."""
    payload: dict[str, Any] = {
        "schema_version": EVALUATOR_PIN_VERSION,
        "status": "unpinned",
        "reason_code": reason_code,
        "declared_route": declared_route,
        "detail": detail,
    }
    payload["pin_digest"] = sha256_json(payload)
    try:
        return cast(UnpinnedEvaluator, UnpinnedEvaluator.model_validate(payload))
    except ValueError as exc:
        raise EvaluatorPinError(f"invalid unpinned evaluator record: {exc}") from exc


def parse_evaluator_pin(document: Any) -> EvaluatorPin:
    """Read a pin record without trusting the caller to say which kind it is."""
    if not isinstance(document, dict):
        raise EvaluatorPinError("evaluator pin must be a JSON object")
    status = document.get("status")
    model: type[PinnedEvaluator] | type[UnpinnedEvaluator]
    if status == "pinned":
        model = PinnedEvaluator
    elif status == "unpinned":
        model = UnpinnedEvaluator
    else:
        raise EvaluatorPinError("evaluator pin status must be 'pinned' or 'unpinned'")
    try:
        return cast(EvaluatorPin, model.model_validate(document))
    except ValueError as exc:
        raise EvaluatorPinError(f"invalid evaluator pin: {exc}") from exc


def load_evaluator_pin(path: Path) -> EvaluatorPin:
    """Read a pin record from disk, rejecting duplicate keys and non-finite constants."""

    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        document: dict[str, Any] = {}
        for key, value in pairs:
            if key in document:
                raise EvaluatorPinError(f"evaluator pin repeats JSON key {key!r}")
            document[key] = value
        return document

    def reject_constant(token: str) -> None:
        raise EvaluatorPinError(f"evaluator pin contains non-finite constant {token}")

    try:
        document = json.loads(
            path.resolve().read_text(encoding="utf-8"),
            object_pairs_hook=unique_pairs,
            parse_constant=reject_constant,
        )
    except EvaluatorPinError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvaluatorPinError(f"cannot load evaluator pin {path}: {exc}") from exc
    return parse_evaluator_pin(document)


def write_evaluator_pin(pin: EvaluatorPin, path: Path) -> Path:
    return cast(Path, write_canonical_json(pin.model_dump(mode="json"), path))


def verify_evaluator_model_binding(
    pin: EvaluatorPin,
    *,
    evaluator_model: str,
    evaluation_scores_complete: bool,
) -> None:
    """Refuse a pin that does not describe the evaluator the observations were scored by."""
    if pin.evaluator_model != evaluator_model:
        raise EvaluatorPinError(
            "evaluator pin does not describe the ablation input's evaluator_model: "
            f"pin expects {pin.evaluator_model!r}, input records {evaluator_model!r}"
        )
    if (
        isinstance(pin, UnpinnedEvaluator)
        and pin.reason_code == "target_evaluation_not_run"
        and evaluation_scores_complete
    ):
        raise EvaluatorPinError(
            "the domain carries evaluation scores, so the target model cannot be recorded as never run"
        )
