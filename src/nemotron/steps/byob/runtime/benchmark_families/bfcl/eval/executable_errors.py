"""Secret-free typed failures for executable evaluation setup and live I/O."""

from __future__ import annotations

from typing import Any, Final

from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.errors import redact_value

_UNSET: Final = object()


class ExecutableEvalError(Exception):
    """Executable evaluation could not be authorized, prepared, or driven."""

    code = "eval_executable_invalid"

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
        self.rendered_actual = (
            redact_value(actual, secret=secret) if actual is not _UNSET else "<missing>"
        )
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


class ExecutableProjectionError(ExecutableEvalError):
    """A source-bound task could not be projected without losing lineage."""

    code = "eval_executable_projection_invalid"


class ExecutableAuthorizationError(ExecutableEvalError):
    """The candidate, task, source, plan, or oracle identity is unauthorized."""

    code = "eval_executable_unauthorized"


class OracleSessionError(ExecutableEvalError):
    """A verified oracle session could not be opened, used, or closed."""

    code = "eval_oracle_session_failed"


class OracleResetError(OracleSessionError):
    """The oracle did not establish a clean task-local initial state."""

    code = "eval_oracle_reset_failed"


class OracleCallError(OracleSessionError):
    """A live oracle tool call failed outside a valid business result."""

    code = "eval_oracle_call_failed"


class OracleStateError(OracleSessionError):
    """The oracle could not return a canonical final state."""

    code = "eval_oracle_state_failed"


class OracleAssertionError(OracleSessionError):
    """Pack assertion infrastructure failed rather than returning a verdict."""

    code = "eval_assertion_infrastructure_failed"


def describe_executable_error(exc: Exception) -> str:
    if isinstance(exc, ExecutableEvalError):
        report = exc.as_report()
        return f"[{report['code']}] {report['subject']}: {report['problem']}"
    return f"[eval_executable_invalid] {type(exc).__name__}: {redact_value(str(exc))}"


__all__ = [
    "ExecutableAuthorizationError",
    "ExecutableEvalError",
    "ExecutableProjectionError",
    "OracleAssertionError",
    "OracleCallError",
    "OracleResetError",
    "OracleSessionError",
    "OracleStateError",
    "describe_executable_error",
]
