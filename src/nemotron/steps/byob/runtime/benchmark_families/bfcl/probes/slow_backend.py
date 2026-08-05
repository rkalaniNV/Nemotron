"""A backend that never returns, used to prove the episode deadline is wired.

Pack tools, resets, and assertions all run through the persistent episode worker,
whose deadline is enforced per operation. Timing out a plain callable would exercise
different code, so the timeout check drives this stand-in through the real path.
"""

from __future__ import annotations

import time
from typing import Any

TOOL_NAME = "sleep_forever"


def list_tools() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": TOOL_NAME,
                "description": "Blocks until the worker is terminated.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }
    ]


def reset(*, ctx: Any, fixtures: dict[str, Any] | None = None) -> None:
    return None


def call_tool(name: str, arguments: dict[str, Any], *, ctx: Any) -> dict[str, Any]:
    if name != TOOL_NAME:
        return {"error": {"code": "unknown_tool"}}
    while True:
        time.sleep(3600)


def get_state() -> dict[str, Any]:
    return {}
