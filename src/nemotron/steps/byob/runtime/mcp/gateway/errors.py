"""HTTP-safe failures emitted by the MCP-to-BFCL gateway."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GatewayError(RuntimeError):
    """One infrastructure failure with a stable BFCL gateway error code."""

    code: str
    message: str
    http_status: int = 500
    poison_session: bool = False

    def __post_init__(self) -> None:
        RuntimeError.__init__(self, self.message)

    def as_dict(self) -> dict[str, dict[str, str]]:
        return {"error": {"code": self.code, "message": self.message}}


def bad_request(code: str, message: str) -> GatewayError:
    return GatewayError(code=code, message=message, http_status=400)


def unavailable(code: str, message: str) -> GatewayError:
    return GatewayError(code=code, message=message, http_status=503)


def upstream_failure(
    code: str,
    message: str,
    *,
    timeout: bool = False,
) -> GatewayError:
    return GatewayError(
        code=code,
        message=message,
        http_status=504 if timeout else 502,
        poison_session=True,
    )
