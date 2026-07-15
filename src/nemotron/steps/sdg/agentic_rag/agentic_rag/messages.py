"""Message / tool / theme / persona formatting helpers. Reused from the reference."""

from __future__ import annotations

import json
from typing import Any, Dict, List


def format_tools_for_prompt(tools: List[Dict[str, Any]]) -> str:
    lines = []
    for td in tools:
        tool = td.get("tool", {}) if "tool" in td else td
        fn = tool.get("function", {})
        lines.append(f"- {fn.get('name', 'Unknown')}: {fn.get('description', 'No description')}")
    return "\n".join(lines)


def format_theme_for_prompt(theme: Any) -> str:
    t = parse_theme(theme)
    return f"{t.get('type', 'general')}: {t.get('description', '')}"


def format_conversation_history_for_prompt(messages: List[Dict[str, Any]]) -> str:
    lines = []
    for msg in messages:
        role = msg.get("role")
        if role == "user":
            lines.append(f"User: {msg.get('content', '')}")
        elif role == "assistant":
            text = f"Assistant: {msg.get('content', '')}"
            tcs = msg.get("tool_calls") or []
            if tcs:
                rendered = []
                for tc in tcs:
                    fn = tc.get("function", {})
                    try:
                        args = json.loads(fn.get("arguments", "{}"))
                    except (json.JSONDecodeError, TypeError):
                        args = fn.get("arguments", "")
                    rendered.append({"name": fn.get("name"), "arguments": args})
                text += f" | Tool-Calls: {rendered}"
            lines.append(text)
        elif role == "tool":
            lines.append(f"Tool: {msg.get('content', '')}")
        elif role == "system" and msg.get("content"):
            lines.append(f"System: {msg.get('content', '')}")
    return "\n".join(lines)


def parse_theme(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {"type": raw, "description": ""}
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list) and parsed:
            return parsed[0] if isinstance(parsed[0], dict) else {"type": str(parsed[0]), "description": ""}
    return {"type": str(raw), "description": ""}
