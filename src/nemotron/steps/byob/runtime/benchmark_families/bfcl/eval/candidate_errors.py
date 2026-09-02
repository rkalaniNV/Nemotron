"""Secret-free typed failures for the native candidate client."""

from __future__ import annotations

from typing import Any, Final

from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.errors import redact_value

_UNSET: Final = object()


class CandidateClientError(Exception):
    """A request could not be constructed, transported, parsed, or replayed."""

    code = "eval_candidate_client_invalid"

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


class CandidateCredentialMissingError(CandidateClientError):
    code = "eval_candidate_credentials_missing"


class CandidateAuthenticationError(CandidateClientError):
    """The endpoint rejected the credential every task in the run shares."""

    code = "eval_candidate_authentication_failed"


class CandidateRequestError(CandidateClientError):
    code = "eval_candidate_request_invalid"


class CandidateProviderExtensionError(CandidateRequestError):
    code = "eval_candidate_provider_extension_invalid"


class CandidateResponseError(CandidateClientError):
    code = "eval_candidate_response_invalid"


class CandidateCacheError(CandidateClientError):
    code = "eval_candidate_cache_invalid"


class CandidateCacheConflictError(CandidateCacheError):
    code = "eval_candidate_cache_conflict"


def describe_candidate_client_error(exc: Exception) -> str:
    if isinstance(exc, CandidateClientError):
        report = exc.as_report()
        return f"[{report['code']}] {report['subject']}: {report['problem']}"
    return f"[eval_candidate_client_invalid] {type(exc).__name__}: {redact_value(str(exc))}"


__all__ = [
    "CandidateAuthenticationError",
    "CandidateCacheConflictError",
    "CandidateCacheError",
    "CandidateClientError",
    "CandidateCredentialMissingError",
    "CandidateProviderExtensionError",
    "CandidateRequestError",
    "CandidateResponseError",
    "describe_candidate_client_error",
]
