"""Interfaces for BYOB benchmark-family implementations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol


@dataclass(frozen=True)
class BenchmarkRunResult:
    """Machine-readable result returned by a family-specific CLI stage."""

    output_path: Path
    output_format: Literal["human", "json"]
    payload: dict[str, Any]

    def render(self) -> str:
        if self.output_format == "json":
            return json.dumps(self.payload, sort_keys=True)
        lines = [f"{key}: {value}" for key, value in self.payload.items()]
        return "\n".join(lines)


class PrepareHook(Protocol):
    def __call__(self, config: Path) -> Path | None: ...


class GenerateHook(Protocol):
    def __call__(
        self, config: Path, *, skip_until: str | None = None
    ) -> Path | None: ...


class EvaluateHook(Protocol):
    def __call__(self, config: Path) -> Path | BenchmarkRunResult: ...


@dataclass(frozen=True)
class BenchmarkFamilySpec:
    """Named hooks that let agents add new benchmark families without rewriting orchestration."""

    name: str
    description: str
    prepare_data: PrepareHook
    generate: GenerateHook
    translate: GenerateHook | None = None
    evaluate: EvaluateHook | None = None
