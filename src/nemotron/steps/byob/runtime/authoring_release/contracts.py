"""Typed adapter boundaries for the adapter-neutral release kernel."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class AdapterReviewContribution:
    """Adapter-owned review facts consumed without transport interpretation."""

    identity_digest: str
    certification_tier: str
    review_data: Mapping[str, Any]
    blockers: Sequence[Mapping[str, Any]] = ()
    risks: Sequence[Mapping[str, Any]] = ()


@dataclass(frozen=True)
class FreezeHookContext:
    adapter_kind: str
    packet: Mapping[str, Any]
    approval: Mapping[str, Any]
    source_digests: Mapping[str, str]


class ReleaseAdapter(Protocol):
    """Transport-specific operations required by review and freeze."""

    @property
    def kind(self) -> str: ...

    def validate_pack(self, pack_root: Path) -> str:
        """Return ``sha256:<hex>`` for one valid canonical candidate pack."""

    def review(
        self,
        pack_root: Path,
        source_digests: Mapping[str, str],
    ) -> AdapterReviewContribution: ...

    def freeze_sidecars(
        self,
        context: FreezeHookContext,
    ) -> Mapping[str, bytes]: ...


class PublicationAdapter(Protocol):
    """Adapter-specific validation around the existing generation pipeline."""

    @property
    def kind(self) -> str: ...

    def validate_pack(self, pack_root: Path) -> str: ...

    def bind_config(
        self,
        config_path: Path,
        pack_root: Path,
        frozen_pack_fingerprint: str,
    ) -> None: ...

    def prepare(self, config_path: Path) -> Path: ...

    def load_validation_report(self, report_path: Path) -> Mapping[str, Any]: ...

    def require_fresh_gold(
        self,
        report: Mapping[str, Any],
        frozen_pack_fingerprint: str,
    ) -> None: ...

    def generate(self, config_path: Path) -> Path: ...

    def verify_publication_manifest(
        self,
        manifest: Mapping[str, Any],
        frozen_pack_fingerprint: str,
    ) -> None: ...
