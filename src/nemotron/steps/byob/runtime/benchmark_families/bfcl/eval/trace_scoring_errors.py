"""Secret-free typed failures for trace scoring.

Scoring is the one stage that turns evidence into a published number, so it
raises rather than degrades. A score derived from an episode that does not belong
to the script, or under a policy this scorer cannot honour, would be a number
whose meaning nobody can state — worse than no number at all.
"""

from __future__ import annotations

from typing import Any, Final

from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.errors import redact_value

_UNSET: Final = object()


class TraceScoringError(Exception):
    """A trace this scorer refuses to turn into a number."""

    code = "eval_trace_scoring_invalid"

    def __init__(
        self,
        subject: str,
        problem: str,
        *,
        expected: str,
        recovery: str,
        actual: Any = _UNSET,
        secret: bool = False,
    ) -> None:
        self.subject = subject
        self.problem = problem
        self.expected = expected
        self.recovery = recovery
        self.rendered_actual = redact_value(actual, secret=secret) if actual is not _UNSET else "<missing>"
        super().__init__(
            f"{subject}: {problem} (observed {self.rendered_actual}); expected {expected}. Fix: {recovery}"
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


class TraceEvidenceError(TraceScoringError):
    """The episode and the script are not two halves of the same replay."""

    code = "eval_trace_evidence_mismatch"


class TraceScoringPolicyError(TraceScoringError):
    """The pinned config asks for scoring this component will not perform."""

    code = "eval_trace_scoring_policy_unsupported"


class TraceAggregationError(TraceScoringError):
    """The task scores offered do not form one candidate's authorized task set."""

    code = "eval_trace_aggregation_invalid"


def describe_trace_scoring_error(exc: Exception) -> str:
    if isinstance(exc, TraceScoringError):
        report = exc.as_report()
        return f"[{report['code']}] {report['subject']}: {report['problem']}"
    return f"[eval_trace_scoring_invalid] {type(exc).__name__}: {redact_value(str(exc))}"


__all__ = [
    "TraceAggregationError",
    "TraceEvidenceError",
    "TraceScoringError",
    "TraceScoringPolicyError",
    "describe_trace_scoring_error",
]
