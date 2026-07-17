"""Deterministic accept/reject split over generated trajectory rows.

Re-runs the structural + reasoning gates on each row's ``structured_messages`` and
partitions rows into accepted / rejected (Data Designer has no row-drop primitive).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from mtsdg.validators import validate_trajectory


def filter_accepted(rows: List[Dict[str, Any]], *, max_reasoning_tokens: int = 400) -> Dict[str, List[Dict[str, Any]]]:
    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for row in rows:
        try:
            messages = json.loads(row["structured_messages"]) if isinstance(row.get("structured_messages"), str) else row.get("structured_messages")
        except (json.JSONDecodeError, TypeError):
            messages = None
        if not messages:
            row["_reject_reasons"] = ["empty structured_messages"]
            rejected.append(row)
            continue
        report = validate_trajectory(messages, max_reasoning_tokens=max_reasoning_tokens)
        status = row.get("trajectory_status")
        if report.ok and status is not False:
            accepted.append(row)
        else:
            row["_reject_reasons"] = report.errors or ["trajectory_status=False"]
            rejected.append(row)
    return {"accepted": accepted, "rejected": rejected}
