"""Typed failures for the contamination gate.

These say something narrower than "the source is wrong" or "the config is
wrong": the source and the config are both valid, and the candidate is
still not allowed to be scored on this benchmark — or not on all of it. A
publication that cannot state which models read its rows fails earlier, as
:class:`...source_errors.ModelExposureError`, because that is a defect in the
benchmark rather than in the comparison being asked for.

Every message names the model and the rows involved, because a contamination
refusal is only actionable if the operator can see which candidate collided with
which role over how many tasks. Model names, roles, and task ids are all
pipeline- or operator-declared identifiers, never credentials: a
:class:`...identity.ModelIdentityClaim` is built from provider, model, revision,
digest, and label, and there is no field on it that can hold a secret. Hashes and
counts are reported verbatim through :func:`...source_errors.render_evidence`;
anything else is reduced to its shape.
"""

from __future__ import annotations

from typing import Any, Final

from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.errors import redact_value
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.source_errors import render_evidence

_UNSET: Final = object()


class ContaminationError(Exception):
    """A candidate may not be scored on this benchmark as configured."""

    code: str = "eval_contamination_invalid"

    def __init__(
        self,
        subject: str,
        problem: str,
        *,
        expected: str,
        recovery: str,
        actual: Any = _UNSET,
    ) -> None:
        self.subject = subject
        self.problem = problem
        self.expected = expected
        self.recovery = recovery
        self.rendered_actual = render_evidence(actual) if actual is not _UNSET else "<missing>"
        super().__init__(
            f"{subject}: {problem} (observed {self.rendered_actual}); "
            f"expected {expected}. Fix: {recovery}"
        )

    def as_report(self) -> dict[str, str]:
        return {
            "code": self.code,
            "subject": self.subject,
            "problem": self.problem,
            "actual": self.rendered_actual,
            "expected": self.expected,
            "recovery": self.recovery,
        }


class CandidateContaminationError(ContaminationError):
    """A candidate is the same model that helped build the tasks it would be scored on."""

    code = "eval_contamination_candidate_exposed"


class UnresolvedContaminationError(ContaminationError):
    """Separation between a candidate and an exposed model cannot be proven."""

    code = "eval_contamination_unresolved"


class EmptyEvaluationTaskSetError(ContaminationError):
    """Excluding contaminated rows left nothing to score."""

    code = "eval_contamination_empty_task_set"


class TaskSetConsistencyError(ContaminationError):
    """The task sets derived for the candidates cannot be compared to each other."""

    code = "eval_contamination_task_set_inconsistent"


class ContaminationPlanDriftError(ContaminationError):
    """The eligible-task plan is not the one the run was authorized with."""

    code = "eval_contamination_plan_drift"


def describe_contamination_error(exc: Exception) -> str:
    """One-line, secret-free summary for a CLI or a step report."""
    if isinstance(exc, ContaminationError):
        report = exc.as_report()
        return f"[{report['code']}] {report['subject']}: {report['problem']}"
    return f"[eval_contamination_invalid] {type(exc).__name__}: {redact_value(str(exc))}"
