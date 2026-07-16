"""The conversation-generation engine (fresh, minimal, config-driven).

Retrieval is HTTP (the external retrieval service). Tools come from config.
Judges are embedded. All prompts live in ``prompts.py``; every knob in ``config.py``.
"""
