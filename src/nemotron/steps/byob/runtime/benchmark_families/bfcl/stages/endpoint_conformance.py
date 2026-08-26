"""The Gold Gate check that decides whether an attested endpoint may publish.

The check runs only when a pack actually pins an attestation. That is deliberate: pinning is
how a pack claims certifiable conformance, and a pack making no claim should not be measured
against one. It also means a hand-written endpoint pack that never pins anything is never
audited here — worth stating plainly, because the protection comes from intake always pinning,
not from this function being able to tell an MCP-backed endpoint from any other.

Three digests have to agree before anything is published: the one the pack pinned at intake,
the one the live endpoint reports at `GET /v1/metadata`, and the one inside the attestation
itself. Each is produced by a different party at a different time, so agreement is the only
evidence that the build being certified is the build answering calls.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from nemotron.steps.byob.runtime.benchmark_families.bfcl.conformance import (
    ConformanceVerdict,
    verify_conformance,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.endpoint import (
    EndpointConfig,
    EndpointOracleClient,
    resolve_endpoint_headers,
)

CHECK_ID = "A1"
CHECK_NAME = "endpoint_conformance"

AttestationFetcher = Callable[[EndpointConfig], Any]


def _fetch_over_https(config: EndpointConfig, *, timeout_s: float) -> Any:
    # Read-only and callable before any session exists, so it needs none of the process
    # isolation that importing pack code would: there is no pack code on this path.
    client = EndpointOracleClient(
        config,
        headers=resolve_endpoint_headers(config),
        timeout_s=timeout_s,
    )
    return client.conformance()


def run_endpoint_conformance_check(
    endpoint_config: EndpointConfig | None,
    endpoint_metadata: Mapping[str, str] | None,
    *,
    fetch: AttestationFetcher | None = None,
    timeout_s: float = 30.0,
    trusted_issuers: Sequence[str] = (),
    local_conformance_report_digest: str | None = None,
) -> dict[str, Any] | None:
    """Return the `A1` check entry, or None when the pack pins no attestation."""
    if endpoint_config is None or endpoint_config.attestation is None:
        return None

    def _entry(status: str, failures: list[dict[str, Any]], verdict: ConformanceVerdict | None) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "id": CHECK_ID,
            "name": CHECK_NAME,
            "status": status,
            "failures": failures,
        }
        if verdict is not None:
            entry["conformance"] = verdict.as_dict()
        return entry

    if endpoint_metadata is None:
        # Without live metadata there is nothing to compare the attestation against, and a
        # comparison against the pack's pin alone would accept a stale document.
        return _entry("fail", [{"reason": "endpoint_metadata_unavailable"}], None)

    try:
        document = (
            fetch(endpoint_config)
            if fetch is not None
            else _fetch_over_https(endpoint_config, timeout_s=timeout_s)
        )
    except Exception as exc:
        return _entry(
            "fail",
            [{"reason": "conformance_unavailable", "detail": f"{type(exc).__name__}: {exc}"}],
            None,
        )

    verdict = verify_conformance(
        document,
        expected_digest=endpoint_config.attestation.expected_digest,
        metadata_content_digest=str(endpoint_metadata.get("content_digest")),
        expected_identity={
            # Closes the chain: what the pack pinned must be what the attestation attests.
            "effective_content_digest": endpoint_config.expected.content_digest,
        },
        local_conformance_report_digest=local_conformance_report_digest,
        trusted_issuers=trusted_issuers,
    )
    if verdict.publishable:
        return _entry("pass", [], verdict)

    failures = [{"reason": finding} for finding in verdict.findings]
    failures.extend(
        # A cap is not a defect in the pack; it is missing evidence, and it still blocks
        # publication, so it has to appear as a reason rather than as a silent downgrade.
        {"reason": "level_capped", "detail": cap, "effective_level": verdict.effective_level}
        for cap in verdict.caps
    )
    if not failures:
        failures = [{"reason": "level_below_l2", "effective_level": verdict.effective_level}]
    return _entry("fail", failures, verdict)
