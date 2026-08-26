"""Secret-free MCP origin provenance carried by a frozen oracle pack.

The lineage record is one file inside the pack, so on its own it proves nothing: it can be
copied into any other endpoint pack. Publication therefore accepts it only when it names this
pack and agrees with the review records sealed beside it. Field names are read structurally so
this module stays free of any import from the MCP release path.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from nemotron.steps.byob.runtime.benchmark_families.bfcl.endpoint import EndpointConfig
from nemotron.steps.byob.runtime.benchmark_families.bfcl.pack_loader import (
    ResolvedPackPaths,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.row_schema import canonical_json

MCP_LINEAGE_VERSION = "bfcl-mcp-publication-lineage-v1"
MCP_LINEAGE_RELATIVE_PATH = Path("provenance") / "mcp_lineage.json"
_REVIEW_PACKET_RELATIVE_PATH = Path("provenance") / "review_packet.json"
_REVIEW_APPROVAL_RELATIVE_PATH = Path("provenance") / "review_approval.json"
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


class OriginProvenanceError(ValueError):
    """Raised when pack origin fields contradict executable pack identity."""


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise OriginProvenanceError(f"MCP lineage repeats JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(token: str) -> None:
    raise OriginProvenanceError(f"MCP lineage contains {token}")


def _digest(value: Any, label: str, *, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise OriginProvenanceError(f"{label} must be sha256:<64 lowercase hex>")
    return value


def _sealed_record(paths: ResolvedPackPaths, relative: Path, label: str) -> dict[str, Any]:
    """Read one provenance record from inside the fingerprinted tree."""
    path = paths.pack_root / relative
    if path.is_symlink() or not path.is_file():
        raise OriginProvenanceError(f"{label} must be a regular file inside the pack")
    try:
        document = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_pairs,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OriginProvenanceError(f"cannot read {label}: {exc}") from exc
    if not isinstance(document, dict):
        raise OriginProvenanceError(f"{label} must be a JSON object")
    return document


def _self_digest(document: Mapping[str, Any], field: str, label: str) -> str:
    """Recompute a record's own digest field so an edited record cannot keep its name."""
    claimed = _digest(document.get(field), f"{label} {field}")
    unsigned = {key: value for key, value in document.items() if key != field}
    observed = "sha256:" + hashlib.sha256(
        canonical_json(unsigned).encode("utf-8")
    ).hexdigest()
    if claimed != observed:
        raise OriginProvenanceError(f"{label} {field} mismatch")
    return str(claimed)


def load_mcp_origin(
    paths: ResolvedPackPaths,
    endpoint_config: EndpointConfig | None,
    *,
    pack_fingerprint: str,
    pack_id: str,
    pack_version: str,
) -> dict[str, Any] | None:
    """Return the publication-safe projection, or None for a non-MCP pack."""
    if not (paths.pack_root / MCP_LINEAGE_RELATIVE_PATH).exists():
        return None
    document = _sealed_record(paths, MCP_LINEAGE_RELATIVE_PATH, "MCP lineage")
    if document.get("schema_version") != MCP_LINEAGE_VERSION:
        raise OriginProvenanceError(
            f"MCP lineage must use schema {MCP_LINEAGE_VERSION!r}"
        )
    claimed = _self_digest(document, "record_digest", "MCP lineage")
    if document.get("provider_kind") != "mcp" or document.get("origin") != "mcp_backed_endpoint":
        raise OriginProvenanceError("MCP lineage has an invalid provider origin")
    if endpoint_config is None:
        raise OriginProvenanceError("MCP lineage is present on a non-endpoint pack")

    identity = document.get("identity")
    review = document.get("review")
    declared = document.get("pack")
    if (
        not isinstance(identity, Mapping)
        or not isinstance(review, Mapping)
        or not isinstance(declared, Mapping)
    ):
        raise OriginProvenanceError("MCP lineage pack, identity, and review must be objects")
    if str(declared.get("pack_id")) != pack_id or str(declared.get("version")) != pack_version:
        raise OriginProvenanceError(
            "MCP lineage names a different pack than manifest.yaml, so it was not "
            "produced by freezing this pack"
        )
    source_pack = _digest(
        identity.get("source_pack_fingerprint"),
        "MCP source_pack_fingerprint",
    )
    packet = _sealed_record(paths, _REVIEW_PACKET_RELATIVE_PATH, "MCP review packet")
    approval = _sealed_record(paths, _REVIEW_APPROVAL_RELATIVE_PATH, "MCP review approval")
    packet_digest = _self_digest(packet, "packet_digest", "MCP review packet")
    approval_digest = _self_digest(approval, "approval_digest", "MCP review approval")
    if review.get("packet_digest") != packet_digest:
        raise OriginProvenanceError("MCP lineage cites a review packet the pack does not carry")
    if review.get("approval_digest") != approval_digest:
        raise OriginProvenanceError("MCP lineage cites an approval the pack does not carry")
    if approval.get("review_packet_digest") != packet_digest:
        raise OriginProvenanceError("the sealed approval covers a different review packet")
    packet_sources = packet.get("source_digests")
    if (
        not isinstance(packet_sources, Mapping)
        or packet_sources.get("canonical_pack") != source_pack
    ):
        raise OriginProvenanceError(
            "the approved review packet covers a different pre-freeze pack than the lineage"
        )
    effective = _digest(
        identity.get("effective_content_digest"),
        "MCP effective_content_digest",
    )
    conformance = _digest(
        identity.get("conformance_digest"),
        "MCP conformance_digest",
    )
    catalog = _digest(identity.get("tool_catalog_digest"), "MCP tool_catalog_digest")
    if endpoint_config.expected.content_digest != effective:
        raise OriginProvenanceError(
            "MCP lineage effective digest differs from endpoint_config.yaml"
        )
    if (
        endpoint_config.attestation is None
        or endpoint_config.attestation.expected_digest != conformance
    ):
        raise OriginProvenanceError(
            "MCP lineage conformance digest differs from endpoint_config.yaml"
        )
    return {
        "provider_kind": "mcp",
        "origin": "mcp_backed_endpoint",
        "profile_version": document.get("profile_version"),
        "mode": document.get("mode"),
        "frozen_pack_fingerprint": (
            pack_fingerprint
            if pack_fingerprint.startswith("sha256:")
            else f"sha256:{pack_fingerprint}"
        ),
        "source_pack_fingerprint": source_pack,
        "effective_content_digest": effective,
        "conformance_digest": conformance,
        "tool_catalog_digest": catalog,
        "lineage_record_digest": claimed,
        "review_packet_digest": _digest(
            review.get("packet_digest"),
            "MCP review packet digest",
        ),
        "review_approval_digest": _digest(
            review.get("approval_digest"),
            "MCP review approval digest",
        ),
        # Reviewer identity is intentionally omitted. The frozen lineage retains it for
        # audit, while publication needs the decision digest rather than personal data.
    }
