"""Trust declarations travel with the reviewed pack, never just its workspace."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

TRUST_FIELDS = frozenset({"trust_mode", "release_status", "official_publishable"})


def trust_fields(mode: str) -> dict[str, Any]:
    if mode == "compliance":
        # Preserve the existing strict packet and signed-manifest formats.
        return {}
    if mode not in {"dev", "release"}:
        raise ValueError(f"unsupported authoring trust mode: {mode!r}")
    return {
        "trust_mode": mode,
        "release_status": "development_unsealed" if mode == "dev" else "release_candidate_unofficial",
        "official_publishable": False,
    }


def packet_trust(document: Mapping[str, Any]) -> dict[str, Any]:
    declared = document.get("adapter_review", {}).get("authoring_trust")
    if declared is None:
        return {}
    if not isinstance(declared, Mapping) or set(declared) != TRUST_FIELDS:
        raise ValueError("authoring trust declaration is incomplete")
    expected = trust_fields(str(declared.get("trust_mode")))
    if not expected or dict(declared) != expected or declared["official_publishable"] is not False:
        raise ValueError("workspace-local authoring releases must remain unofficial")
    return expected


def pack_trust(pack_root: Path) -> dict[str, Any]:
    from nemotron.steps.byob.runtime.authoring_release.review import load_review_packet

    packet_path = pack_root / "provenance" / "review_packet.json"
    if not packet_path.exists():
        return {}
    # The packet digest is also bound into the pack's verified origin provenance.
    return packet_trust(load_review_packet(packet_path).document)
