"""HTTPS client and configuration for BFCL Oracle HTTP v1 endpoints."""

from __future__ import annotations

import json
import os
import re
import ssl
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from nemotron.steps.byob.runtime.benchmark_families.bfcl.isolation import (
    assert_pack_allowed,
)

PROTOCOL_VERSION = "bfcl-oracle-http-v1"
_CONFIG_KEYS = frozenset(
    {
        "protocol_version",
        "base_url",
        "auth",
        "expected",
        "tls",
        "max_request_bytes",
        "max_response_bytes",
    }
)
_AUTH_KEYS = frozenset({"bearer_token_env", "headers"})
_EXPECTED_KEYS = frozenset({"oracle_id", "oracle_version", "content_digest"})
_TLS_KEYS = frozenset({"ca_bundle_path"})
_HEADER_NAME = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")


@dataclass(frozen=True)
class EndpointIdentity:
    protocol_version: str
    oracle_id: str
    oracle_version: str
    content_digest: str

    @classmethod
    def from_mapping(cls, value: Any, *, source: str) -> EndpointIdentity:
        if not isinstance(value, dict):
            raise ValueError(f"{source} must be a JSON object")
        required = {
            "protocol_version",
            "oracle_id",
            "oracle_version",
            "content_digest",
        }
        unknown = sorted(set(value) - required)
        if unknown:
            raise ValueError(f"{source} has unknown fields: {', '.join(unknown)}")
        missing = sorted(required - set(value))
        if missing:
            raise ValueError(f"{source} is missing: {', '.join(missing)}")
        fields = {name: value[name] for name in required}
        if any(not isinstance(item, str) or not item.strip() for item in fields.values()):
            raise ValueError(f"{source} identity fields must be non-empty strings")
        if fields["protocol_version"] != PROTOCOL_VERSION:
            raise ValueError(
                f"{source} protocol_version must be {PROTOCOL_VERSION!r}, got {fields['protocol_version']!r}"
            )
        digest = str(fields["content_digest"])
        if not digest.startswith("sha256:") or len(digest) != 71:
            raise ValueError(f"{source} content_digest must be sha256:<64 hex characters>")
        try:
            int(digest.removeprefix("sha256:"), 16)
        except ValueError as exc:
            raise ValueError(f"{source} content_digest is not hexadecimal") from exc
        return cls(**fields)

    def as_dict(self) -> dict[str, str]:
        return {
            "protocol_version": self.protocol_version,
            "oracle_id": self.oracle_id,
            "oracle_version": self.oracle_version,
            "content_digest": self.content_digest,
        }


@dataclass(frozen=True)
class EndpointConfig:
    path: Path
    base_url: str
    expected: EndpointIdentity
    bearer_token_env: str | None = None
    header_env: tuple[tuple[str, str], ...] = ()
    ca_bundle_path: Path | None = None
    max_request_bytes: int = 10 * 1024 * 1024
    max_response_bytes: int = 10 * 1024 * 1024


def _reject_unknown(value: Mapping[str, Any], allowed: frozenset[str], source: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{source} has unknown keys: {', '.join(unknown)}")


def _nonempty_string(value: Any, source: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{source} must be a non-empty string")
    return value.strip()


def load_endpoint_config(
    path: Path,
    *,
    allowed_roots: tuple[Path, ...],
) -> EndpointConfig:
    """Load a strict, secret-free endpoint declaration."""
    path = assert_pack_allowed(path, allowed_roots)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"endpoint config must be a mapping: {path}")
    _reject_unknown(raw, _CONFIG_KEYS, "endpoint config")
    protocol = _nonempty_string(raw.get("protocol_version"), "endpoint protocol_version")
    if protocol != PROTOCOL_VERSION:
        raise ValueError(f"endpoint protocol_version must be {PROTOCOL_VERSION!r}, got {protocol!r}")

    base_url = _nonempty_string(raw.get("base_url"), "endpoint base_url").rstrip("/")
    parsed = urllib.parse.urlsplit(base_url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("endpoint base_url must be an HTTPS origin/path without credentials, query, or fragment")

    auth = raw.get("auth") or {}
    if not isinstance(auth, dict):
        raise ValueError("endpoint auth must be a mapping")
    _reject_unknown(auth, _AUTH_KEYS, "endpoint auth")
    bearer = auth.get("bearer_token_env")
    if bearer is not None:
        bearer = _nonempty_string(bearer, "endpoint auth.bearer_token_env")
    headers = auth.get("headers") or {}
    if not isinstance(headers, dict):
        raise ValueError("endpoint auth.headers must map HTTP header names to environment names")
    header_env: list[tuple[str, str]] = []
    for header, env_name in headers.items():
        header_name = _nonempty_string(header, "endpoint auth header name")
        if _HEADER_NAME.fullmatch(header_name) is None:
            raise ValueError(f"endpoint auth header {header_name!r} is not a valid HTTP field name")
        if header_name.lower() in {"authorization", "host", "content-length"}:
            raise ValueError(f"endpoint auth header {header_name!r} is reserved")
        header_env.append((header_name, _nonempty_string(env_name, f"endpoint auth.headers.{header_name}")))

    tls = raw.get("tls") or {}
    if not isinstance(tls, dict):
        raise ValueError("endpoint tls must be a mapping")
    _reject_unknown(tls, _TLS_KEYS, "endpoint tls")
    ca_bundle_path = None
    if tls.get("ca_bundle_path") is not None:
        candidate = Path(_nonempty_string(tls["ca_bundle_path"], "endpoint tls.ca_bundle_path"))
        if not candidate.is_absolute():
            candidate = path.parent / candidate
        ca_bundle_path = assert_pack_allowed(candidate, allowed_roots)
        if not ca_bundle_path.is_file():
            raise FileNotFoundError(f"endpoint CA bundle does not exist: {ca_bundle_path}")

    request_limit = raw.get("max_request_bytes", 10 * 1024 * 1024)
    if not isinstance(request_limit, int) or isinstance(request_limit, bool) or request_limit <= 0:
        raise ValueError("endpoint max_request_bytes must be a positive integer")
    response_limit = raw.get("max_response_bytes", 10 * 1024 * 1024)
    if not isinstance(response_limit, int) or isinstance(response_limit, bool) or response_limit <= 0:
        raise ValueError("endpoint max_response_bytes must be a positive integer")
    expected = raw.get("expected")
    if not isinstance(expected, dict):
        raise ValueError("endpoint expected must be a mapping")
    _reject_unknown(expected, _EXPECTED_KEYS, "endpoint expected")
    return EndpointConfig(
        path=path,
        base_url=base_url,
        expected=EndpointIdentity.from_mapping(
            {"protocol_version": protocol, **expected},
            source="endpoint expected",
        ),
        bearer_token_env=bearer,
        header_env=tuple(sorted(header_env)),
        ca_bundle_path=ca_bundle_path,
        max_request_bytes=request_limit,
        max_response_bytes=response_limit,
    )


def resolve_endpoint_headers(
    config: EndpointConfig,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Resolve configured secret references without retaining their values in config."""
    source = os.environ if environ is None else environ
    headers: dict[str, str] = {}
    if config.bearer_token_env is not None:
        token = source.get(config.bearer_token_env)
        if not token or "\r" in token or "\n" in token:
            raise ValueError(f"endpoint authentication environment variable {config.bearer_token_env!r} is missing")
        headers["Authorization"] = f"Bearer {token}"
    for name, env_name in config.header_env:
        value = source.get(env_name)
        if not value or "\r" in value or "\n" in value:
            raise ValueError(f"endpoint authentication environment variable {env_name!r} is missing")
        headers[name] = value
    return headers


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> None:
        return None


class EndpointOracleClient:
    """Stateful client implementing the same callable surface as a local backend."""

    def __init__(
        self,
        config: EndpointConfig,
        *,
        headers: Mapping[str, str],
        timeout_s: float,
    ) -> None:
        self.config = config
        self.headers = dict(headers)
        self.timeout_s = timeout_s
        self.session_id: str | None = None
        context = ssl.create_default_context(cafile=str(config.ca_bundle_path) if config.ca_bundle_path else None)
        self._opener = urllib.request.build_opener(
            _NoRedirect(),
            urllib.request.HTTPSHandler(context=context),
        )

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> Any:
        body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        if body is not None and len(body) > self.config.max_request_bytes:
            raise RuntimeError("endpoint request exceeds max_request_bytes")
        headers = {"Accept": "application/json", **self.headers}
        if body is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            f"{self.config.base_url}{path}",
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with self._opener.open(request, timeout=self.timeout_s) as response:
                declared = response.headers.get("Content-Length")
                if declared and int(declared) > self.config.max_response_bytes:
                    raise RuntimeError("endpoint response exceeds max_response_bytes")
                raw = response.read(self.config.max_response_bytes + 1)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"endpoint {method} {path} returned HTTP {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(f"endpoint {method} {path} failed: {type(exc).__name__}") from exc
        if len(raw) > self.config.max_response_bytes:
            raise RuntimeError("endpoint response exceeds max_response_bytes")
        if not raw and method == "DELETE":
            return None
        try:
            return json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"endpoint {method} {path} returned invalid JSON") from exc

    def metadata(self) -> dict[str, str]:
        identity = EndpointIdentity.from_mapping(
            self._request("GET", "/v1/metadata"),
            source="endpoint metadata",
        )
        if identity != self.config.expected:
            raise RuntimeError("endpoint metadata does not match the expected oracle identity or content digest")
        return identity.as_dict()

    def list_tools(self) -> list[str]:
        self.metadata()
        payload = self._request("GET", "/v1/tools")
        tools = payload.get("tools") if isinstance(payload, dict) else None
        if not isinstance(tools, list) or any(not isinstance(item, str) for item in tools):
            raise RuntimeError("endpoint /v1/tools must return {'tools': [<name>, ...]}")
        return tools

    def reset(self, *, ctx: Any, fixtures: dict[str, Any] | None = None) -> None:
        self.close()
        payload = self._request(
            "POST",
            "/v1/sessions",
            {
                "context": {
                    "clock": ctx.clock.isoformat(),
                    "seed": ctx.seed,
                    "timeout_s": ctx.timeout_s,
                    "task_id": ctx.task_id,
                },
                "fixtures": fixtures,
            },
        )
        if not isinstance(payload, dict):
            raise RuntimeError("endpoint session response must be an object")
        session_id = payload.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise RuntimeError("endpoint session response needs a non-empty session_id")
        self.session_id = session_id
        try:
            identity = EndpointIdentity.from_mapping(
                payload.get("oracle"),
                source="endpoint session oracle",
            )
            if identity != self.config.expected:
                raise RuntimeError("endpoint identity changed while creating a session")
        except Exception:
            self.close()
            raise

    def _session_path(self, suffix: str) -> str:
        if self.session_id is None:
            raise RuntimeError("endpoint session is not initialized; reset must run first")
        identifier = urllib.parse.quote(self.session_id, safe="")
        return f"/v1/sessions/{identifier}{suffix}"

    def call_tool(self, name: str, arguments: dict[str, Any], *, ctx: Any) -> Any:
        return self._request(
            "POST",
            self._session_path("/calls"),
            {
                "name": name,
                "arguments": arguments,
                "turn_index": ctx.turn_index,
            },
        )

    def get_state(self) -> dict[str, Any]:
        state = self._request("GET", self._session_path("/state"))
        if not isinstance(state, dict):
            raise RuntimeError("endpoint state response must be a JSON object")
        return state

    def close(self, *, suppress_errors: bool = False) -> None:
        if self.session_id is None:
            return
        path = self._session_path("")
        try:
            self._request("DELETE", path)
        except Exception:
            if not suppress_errors:
                raise
        else:
            self.session_id = None
