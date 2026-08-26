from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from nemotron.steps.byob.runtime.benchmark_families.bfcl.conformance import (
    ATTESTATION_KIND,
    AttestationError,
    ConformanceAttestation,
    attestation_digest,
    verify_conformance,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.endpoint import (
    load_endpoint_config,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.endpoint_conformance import (
    run_endpoint_conformance_check,
)

DIGESTS = {name: f"sha256:{chr(97 + index) * 64}" for index, name in enumerate("abcdefgh")}
EFFECTIVE = "sha256:" + "1" * 64
METADATA_DIGEST = EFFECTIVE


def _attestation(**overrides: Any) -> dict[str, Any]:
    document: dict[str, Any] = {
        "schema_version": ATTESTATION_KIND,
        "provider_kind": "mcp",
        "profile_version": "bfcl-mcp-oracle-v1",
        "level": "L2",
        "effective_content_digest": EFFECTIVE,
        "gateway_artifact_digest": DIGESTS["a"],
        "shim_artifact_digest": None,
        "tool_catalog_digest": DIGESTS["b"],
        "server_content_digest": DIGESTS["c"],
        "snapshot_digest": None,
        "probe_report_digest": DIGESTS["d"],
        "gateway_conformance_report_digest": DIGESTS["e"],
        "gateway_evidence_kind": "locally_verified",
        "gateway_evidence_issuer": "bfcl-mcp-conformance-v1",
        "state_observability": "complete",
        "read_only_boundary": None,
        "checks": [
            {"id": "P1", "requirement": "required", "status": "pass", "reason": None},
            {"id": "P7", "requirement": "conditional", "status": "not_applicable", "reason": "no mutating tool"},
        ],
    }
    document.update(overrides)
    return document


def _verify(document: dict[str, Any], **overrides: Any):
    kwargs: dict[str, Any] = {
        "expected_digest": attestation_digest(document),
        "metadata_content_digest": METADATA_DIGEST,
        "local_conformance_report_digest": DIGESTS["e"],
    }
    kwargs.update(overrides)
    return verify_conformance(document, **kwargs)


# --- Schema strictness -------------------------------------------------------------------


def test_an_unknown_field_is_refused_rather_than_ignored() -> None:
    document = _attestation(future_claim="trust me")
    with pytest.raises(AttestationError, match="unknown fields: future_claim"):
        ConformanceAttestation.from_mapping(document)


def test_optional_digests_must_be_explicit_nulls_not_omissions() -> None:
    document = _attestation()
    del document["snapshot_digest"]
    # Omitting a field would let two different deployments hash to the same document.
    with pytest.raises(AttestationError, match="is missing: snapshot_digest"):
        ConformanceAttestation.from_mapping(document)


def test_a_free_form_read_only_boundary_is_refused() -> None:
    document = _attestation(read_only_boundary="we promise not to write")
    with pytest.raises(AttestationError, match="read_only_boundary must be one of"):
        ConformanceAttestation.from_mapping(document)


def test_a_malformed_document_verifies_as_unreadable_rather_than_low() -> None:
    verdict = _verify({"schema_version": ATTESTATION_KIND})
    assert verdict.attested_level == "unknown"
    assert verdict.effective_level == "L0"
    assert verdict.publishable is False
    assert verdict.findings[0].startswith("attestation_malformed:")


def test_the_digest_covers_the_document_regardless_of_key_order() -> None:
    document = _attestation()
    reordered = dict(reversed(list(document.items())))
    assert attestation_digest(document) == attestation_digest(reordered)


# --- Verification ------------------------------------------------------------------------


def test_a_well_evidenced_l2_attestation_publishes() -> None:
    verdict = _verify(_attestation())
    assert verdict.attested_level == "L2"
    assert verdict.effective_level == "L2"
    assert verdict.publishable is True
    assert verdict.findings == ()
    assert verdict.caps == ()


def test_a_document_the_pack_did_not_pin_is_refused() -> None:
    verdict = _verify(_attestation(), expected_digest="sha256:" + "9" * 64)
    assert "attestation_digest_mismatch" in verdict.findings
    assert verdict.publishable is False


def test_an_attestation_describing_a_different_build_than_metadata_is_refused() -> None:
    # The live endpoint answers calls as one build while attesting to another.
    verdict = _verify(_attestation(), metadata_content_digest="sha256:" + "7" * 64)
    assert "effective_digest_differs_from_metadata" in verdict.findings
    assert verdict.effective_level == "L0"


def test_the_pack_pin_the_live_metadata_and_the_attestation_must_all_agree() -> None:
    document = _attestation()
    verdict = _verify(
        document,
        expected_identity={"effective_content_digest": "sha256:" + "5" * 64},
    )
    assert "identity_mismatch:effective_content_digest" in verdict.findings


def test_a_gateway_cannot_certify_itself() -> None:
    # locally_verified means BFCL ran the suite. Without BFCL's own report digest the claim
    # is self-reported, which caps the endpoint below publication.
    verdict = _verify(_attestation(), local_conformance_report_digest=None)
    assert verdict.attested_level == "L2"
    assert verdict.effective_level == "L1"
    assert verdict.publishable is False
    assert verdict.caps == ("gateway_evidence_self_reported",)
    assert verdict.findings == ()


def test_a_locally_verified_report_for_another_build_is_a_hard_failure() -> None:
    verdict = _verify(_attestation(), local_conformance_report_digest="sha256:" + "4" * 64)
    assert "gateway_conformance_report_digest_mismatch" in verdict.findings


def test_a_signed_release_needs_an_issuer_the_verifier_trusts() -> None:
    document = _attestation(
        gateway_evidence_kind="signed_release",
        gateway_evidence_issuer="vendor-self-signed",
    )
    untrusted = _verify(document, local_conformance_report_digest=None)
    assert untrusted.caps == ("gateway_evidence_issuer_untrusted",)
    assert untrusted.publishable is False

    trusted = _verify(
        document,
        local_conformance_report_digest=None,
        trusted_issuers=("vendor-self-signed",),
    )
    assert trusted.publishable is True


@pytest.mark.parametrize(
    ("overrides", "expected_cap"),
    [
        ({"server_content_digest": None}, "server_content_digest_absent"),
        ({"state_observability": "diagnostic"}, "state_observability_incomplete"),
    ],
)
def test_missing_evidence_caps_the_level_instead_of_failing_the_pack(
    overrides: dict[str, Any],
    expected_cap: str,
) -> None:
    verdict = _verify(_attestation(**overrides))
    assert verdict.attested_level == "L2"
    assert verdict.effective_level == "L1"
    assert verdict.caps == (expected_cap,)
    assert verdict.findings == ()
    assert verdict.publishable is False


def test_a_skipped_probe_never_counts_as_a_passing_one() -> None:
    document = _attestation(
        checks=[{"id": "P1", "requirement": "required", "status": "skipped", "reason": None}]
    )
    verdict = _verify(document)
    assert "check_skipped:P1" in verdict.findings
    assert verdict.publishable is False


def test_a_conditional_probe_may_be_inapplicable_only_with_a_stated_reason() -> None:
    document = _attestation(
        checks=[
            {"id": "P1", "requirement": "required", "status": "pass", "reason": None},
            {"id": "P7", "requirement": "conditional", "status": "not_applicable", "reason": None},
        ]
    )
    verdict = _verify(document)
    assert "conditional_check_without_reason:P7" in verdict.findings


def test_a_required_probe_cannot_be_declared_inapplicable() -> None:
    document = _attestation(
        checks=[
            {"id": "P1", "requirement": "required", "status": "not_applicable", "reason": "awkward"}
        ]
    )
    verdict = _verify(document)
    assert "required_check_not_passed:P1" in verdict.findings


def test_a_lower_attested_level_is_reported_as_itself() -> None:
    verdict = _verify(_attestation(level="L1"))
    assert verdict.attested_level == "L1"
    assert verdict.effective_level == "L1"
    assert verdict.publishable is False
    assert verdict.findings == ()


# --- Endpoint config pinning (MCP-401) ---------------------------------------------------


def _write_endpoint_config(root: Path, **overrides: Any) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    document: dict[str, Any] = {
        "protocol_version": "bfcl-oracle-http-v1",
        "base_url": "https://oracle.example.test",
        "expected": {
            "oracle_id": "catalog-oracle",
            "oracle_version": "1.0.0",
            "content_digest": EFFECTIVE,
        },
        "attestation": {"kind": ATTESTATION_KIND, "expected_digest": DIGESTS["f"]},
    }
    document.update(overrides)
    path = root / "endpoint_config.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=True), encoding="utf-8")
    return path


def test_an_endpoint_config_can_pin_an_attestation(tmp_path: Path) -> None:
    path = _write_endpoint_config(tmp_path / "pack")
    config = load_endpoint_config(path, allowed_roots=(tmp_path,))
    assert config.attestation is not None
    assert config.attestation.expected_digest == DIGESTS["f"]


def test_an_endpoint_config_without_an_attestation_stays_valid(tmp_path: Path) -> None:
    # Making no claim is legal; it simply cannot publish.
    path = _write_endpoint_config(tmp_path / "pack")
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    del document["attestation"]
    path.write_text(yaml.safe_dump(document, sort_keys=True), encoding="utf-8")
    assert load_endpoint_config(path, allowed_roots=(tmp_path,)).attestation is None


@pytest.mark.parametrize(
    ("attestation", "match"),
    [
        ({"kind": "something-else", "expected_digest": DIGESTS["f"]}, "attestation.kind must be"),
        ({"kind": ATTESTATION_KIND, "expected_digest": "not-a-digest"}, "expected_digest must be"),
        ({"kind": ATTESTATION_KIND, "expected_digest": DIGESTS["f"], "trust": "me"}, "unknown keys"),
    ],
)
def test_a_malformed_attestation_pin_is_refused_at_load(
    tmp_path: Path,
    attestation: dict[str, Any],
    match: str,
) -> None:
    path = _write_endpoint_config(tmp_path / "pack", attestation=attestation)
    with pytest.raises(ValueError, match=match):
        load_endpoint_config(path, allowed_roots=(tmp_path,))


# --- The Gold Gate check (MCP-403, 409, 410) ---------------------------------------------


class _Config:
    """Minimal stand-in for the parts of EndpointConfig this check reads."""

    def __init__(self, *, expected_digest: str | None, content_digest: str = EFFECTIVE) -> None:
        self.attestation = None if expected_digest is None else _Pin(expected_digest)
        self.expected = _Expected(content_digest)


class _Pin:
    def __init__(self, expected_digest: str) -> None:
        self.kind = ATTESTATION_KIND
        self.expected_digest = expected_digest


class _Expected:
    def __init__(self, content_digest: str) -> None:
        self.content_digest = content_digest


def test_a_pack_that_pins_nothing_is_not_audited_here() -> None:
    # A local Python oracle, or an endpoint making no conformance claim, must not acquire a
    # fabricated passing check just because this code ran.
    assert run_endpoint_conformance_check(None, {"content_digest": EFFECTIVE}) is None
    assert (
        run_endpoint_conformance_check(
            _Config(expected_digest=None),  # type: ignore[arg-type]
            {"content_digest": EFFECTIVE},
        )
        is None
    )


def test_a_self_reported_l2_endpoint_fails_the_gate_with_the_cap_named() -> None:
    document = _attestation()
    entry = run_endpoint_conformance_check(
        _Config(expected_digest=attestation_digest(document)),  # type: ignore[arg-type]
        {"content_digest": METADATA_DIGEST},
        fetch=lambda _config: document,
    )
    assert entry is not None
    assert entry["id"] == "A1"
    assert entry["status"] == "fail"
    assert entry["conformance"]["effective_level"] == "L1"
    assert entry["failures"] == [
        {
            "reason": "level_capped",
            "detail": "gateway_evidence_self_reported",
            "effective_level": "L1",
        }
    ]


def test_an_endpoint_verified_by_bfcl_itself_passes_the_gate() -> None:
    document = _attestation()
    entry = run_endpoint_conformance_check(
        _Config(expected_digest=attestation_digest(document)),  # type: ignore[arg-type]
        {"content_digest": METADATA_DIGEST},
        fetch=lambda _config: document,
        local_conformance_report_digest=DIGESTS["e"],
    )
    assert entry is not None
    assert entry["status"] == "pass"
    assert entry["failures"] == []
    assert entry["conformance"]["effective_level"] == "L2"


def test_an_unreachable_conformance_route_fails_closed() -> None:
    def _explode(_config: Any) -> Any:
        raise RuntimeError("endpoint GET /v1/conformance returned HTTP 404")

    entry = run_endpoint_conformance_check(
        _Config(expected_digest=DIGESTS["f"]),  # type: ignore[arg-type]
        {"content_digest": METADATA_DIGEST},
        fetch=_explode,
    )
    assert entry is not None
    assert entry["status"] == "fail"
    assert entry["failures"][0]["reason"] == "conformance_unavailable"
    assert "HTTP 404" in entry["failures"][0]["detail"]


def test_missing_live_metadata_fails_rather_than_trusting_the_pin_alone() -> None:
    entry = run_endpoint_conformance_check(
        _Config(expected_digest=DIGESTS["f"]),  # type: ignore[arg-type]
        None,
        fetch=lambda _config: _attestation(),
    )
    assert entry is not None
    assert entry["status"] == "fail"
    assert entry["failures"] == [{"reason": "endpoint_metadata_unavailable"}]


def test_the_check_blocks_gold_through_the_existing_tier_derivation() -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.oracle_validation import (
        derive_pack_tier,
    )

    document = _attestation()
    entry = run_endpoint_conformance_check(
        _Config(expected_digest=attestation_digest(document)),  # type: ignore[arg-type]
        {"content_digest": METADATA_DIGEST},
        fetch=lambda _config: document,
    )
    report = {
        "checks": [{"id": 1, "name": "template_tool_names", "status": "pass", "failures": []}],
        "extra_checks": [entry],
        "stats": {"has_oracle": True, "n_templates": 3, "n_assertions": 2, "n_tools": 4},
    }
    # No separate gate: a failing A1 flows through the tier rule the pipeline already uses.
    assert derive_pack_tier(report) == (False, "silver")

    passing = copy.deepcopy(report)
    passing["extra_checks"] = [{**entry, "status": "pass", "failures": []}]  # type: ignore[dict-item]
    assert derive_pack_tier(passing) == (True, "gold")


def test_the_report_entry_is_json_serialisable() -> None:
    document = _attestation()
    entry = run_endpoint_conformance_check(
        _Config(expected_digest=attestation_digest(document)),  # type: ignore[arg-type]
        {"content_digest": METADATA_DIGEST},
        fetch=lambda _config: document,
    )
    # It is written into oracle_validation_report.json, so it has to survive that trip.
    assert json.loads(json.dumps(entry, sort_keys=True)) == entry
