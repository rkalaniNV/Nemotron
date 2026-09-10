"""Per-task progress beneath the existing verified generation-resume boundary."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from nemotron.steps.byob.runtime.pack_authoring.artifacts import sha256_json, write_canonical_json


class TaskProgress:
    """Store completed deterministic tasks; incomplete tasks are never accepted."""

    def __init__(self, root: Path, stage: str, identity: dict[str, Any]) -> None:
        self.stage = stage
        self.identity = sha256_json(identity)
        self.root = root / "partial" / stage / self.identity.removeprefix("sha256:")

    def run(
        self, inputs: dict[str, Any], produce: Callable[[], dict[str, Any]], *, index: int, total: int,
    ) -> dict[str, Any]:
        task_id = inputs["task"]["task_id"]
        digest = sha256_json(inputs)
        path = self.root / f"{digest.removeprefix('sha256:')}.json"
        if path.exists():
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
                recorded_digest = document.pop("record_digest")
                if (
                    recorded_digest != sha256_json(document)
                    or document["identity"] != self.identity
                    or document["inputs"] != digest
                    or document["version"] != "bfcl-task-progress-v1"
                    or not isinstance(document["result"], dict)
                ):
                    raise ValueError("task checkpoint binding mismatch")
            except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
                raise ValueError(
                    f"invalid task checkpoint {path}; inspect it and restart this generation run"
                ) from exc
            print(f"BFCL {self.stage} resumed {index}/{total} task={task_id}", file=sys.stderr, flush=True)
            return document["result"]
        print(f"BFCL {self.stage} starting {index}/{total} task={task_id}", file=sys.stderr, flush=True)
        result = produce()
        document = {"version": "bfcl-task-progress-v1", "identity": self.identity, "inputs": digest, "result": result}
        write_canonical_json({**document, "record_digest": sha256_json(document)}, path)
        print(f"BFCL {self.stage} completed {index}/{total} task={task_id}", file=sys.stderr, flush=True)
        return result
