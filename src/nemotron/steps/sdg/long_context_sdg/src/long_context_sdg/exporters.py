"""Lossless rich and trainer-oriented canonical record exporters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from .checkpoint import load_records


def export_records(
    source: Path,
    destination: Path,
    *,
    output_format: Literal["messages", "messages_and_tools", "rich"],
) -> int:
    records = [record for record in load_records(source) if record.status == "accepted"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as fh:
        for record in records:
            if output_format == "messages":
                value = {"messages": record.messages}
            elif output_format == "messages_and_tools":
                value = {"messages": record.messages, "tools": record.tools}
            else:
                value = record.model_dump()
            fh.write(json.dumps(value, ensure_ascii=False) + "\n")
    return len(records)
