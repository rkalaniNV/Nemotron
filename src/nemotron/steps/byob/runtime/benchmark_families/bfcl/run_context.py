"""Shared RunContext for oracle backends (pipeline-constructed only)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class RunContext:
    """Injected clock/seed/timeout for one tool call or reset."""

    clock: datetime
    seed: int
    timeout_s: float
    task_id: str
    turn_index: int = 0
