"""Runtime tool contracts (the tool definitions the agent sees).

Three model tools — ``retrieve`` (the live retriever), ``memory_read``,
``memory_write``. During generation these are resolved by
:class:`mtsdg.runtime.LiveToolExecutor` against the real retriever + validated
memory store.

The ``context.compress`` schema is retained here only to document the app-level
compaction operation and to let the trajectory validator flag it if it ever leaks
into the chat: context compaction is automatic and internal (see
:mod:`mtsdg.generator`), never a tool the model calls.
"""

from __future__ import annotations

from typing import Any, Dict

# --- retrieve (the retriever) --------------------------------------------- #
RETRIEVE_SCHEMA: Dict[str, Any] = {
    "name": "retrieve",
    "description": (
        "Retrieve passages from the knowledge corpus for a query. Results may be "
        "imperfect; inspect them and, if they do not answer the need, rewrite the "
        "query (more specific terms / filters) and retrieve again."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "filters": {
                "type": "object",
                "properties": {
                    "domain": {"type": "string"},
                    "recency": {"type": "string"},
                },
            },
            "top_k": {"type": "integer", "minimum": 1, "maximum": 8},
        },
        "required": ["query"],
    },
}

# --- context.compress ----------------------------------------------------- #
CONTEXT_COMPRESS_SCHEMA: Dict[str, Any] = {
    "name": "context.compress",
    "description": (
        "Compress completed conversation turns into a source-linked rolling "
        "summary without adding new facts. Fires when the active context reaches "
        "the token threshold."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "conversation_id": {"type": "string"},
            "from_turn": {"type": "integer", "minimum": 1},
            "to_turn": {"type": "integer", "minimum": 1},
            "prior_summary_id": {"type": "string"},
            "preserve": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [
                        "user_stated_facts",
                        "constraints",
                        "authorities",
                        "tool_results",
                        "open_questions",
                        "memory_preferences",
                        "decisions",
                    ],
                },
            },
        },
        "required": ["conversation_id", "from_turn", "to_turn", "preserve"],
    },
}

# --- memory_read ---------------------------------------------------------- #
MEMORY_READ_SCHEMA: Dict[str, Any] = {
    "name": "memory_read",
    "description": "Read permitted saved user/conversation preferences.",
    "parameters": {
        "type": "object",
        "properties": {
            "keys": {"type": "array", "items": {"type": "string"}},
            "scope": {"type": "string", "enum": ["user", "conversation"]},
        },
        "required": ["scope"],
    },
}

# --- memory_write --------------------------------------------------------- #
MEMORY_WRITE_SCHEMA: Dict[str, Any] = {
    "name": "memory_write",
    "description": "Persist an allowed preference after a direct user request.",
    "parameters": {
        "type": "object",
        "properties": {
            "key": {"type": "string"},
            "value": {},
            "scope": {"type": "string", "enum": ["user", "conversation"]},
            "reason": {"type": "string"},
        },
        "required": ["key", "value", "scope", "reason"],
    },
}

TOOL_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "retrieve": RETRIEVE_SCHEMA,
    "context.compress": CONTEXT_COMPRESS_SCHEMA,
    "memory_read": MEMORY_READ_SCHEMA,
    "memory_write": MEMORY_WRITE_SCHEMA,
}
