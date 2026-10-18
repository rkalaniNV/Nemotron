# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared contracts for safety evaluation adapters."""

from collections.abc import Mapping
from typing import Any, Protocol


class EvalInputError(ValueError):
    """An evaluation configuration or input cannot be scored."""


class GuardAdapter(Protocol):
    """Turn one canonical safety event into a canonical guard decision."""

    name: str
    max_concurrency: int

    @property
    def public_identity(self) -> Mapping[str, Any]:
        """Credential-free target identity."""

    def decide(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return decision, parse_status, latency_ms, and optional rail telemetry."""

    def close(self) -> None:
        """Release endpoint resources."""
