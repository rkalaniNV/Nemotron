"""Run one task against the target model and collect the calls it made.

The model is given the system prompt, the opening user turn and the pack's tool
schemas, and then driven in a loop: every call it emits is executed against the real
backend and the result handed back, so a `dependent_call` task can actually read the id
it needs from the first result. When the model answers in prose instead of calling, the
next *scripted* user turn is delivered — the gold conversation's later user turns,
replayed verbatim rather than simulated, because a simulator would put a second model
in the measurement path and A2's whole point is that wording matters.

Tools execute through `ProcessWorker.run_episode`, the same isolation path the gold
replay and A4's mutation gate use. Because the backend is deterministic and reset at
the start of every episode, step *k* is obtained by replaying calls 0..k in a fresh
episode rather than by holding a worker open across model turns.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bfcl_ablation.target.client import Reply, TargetClient, ToolCall, to_responses_tools

# A task whose gold trace is one call gets a generous ceiling anyway: the interesting
# failure is a model that loops, and cutting it off early would score the loop as a
# short wrong answer instead of the runaway it is.
MAX_MODEL_TURNS = 8


@dataclass
class Rollout:
    task_id: str
    arm: str
    calls: list[dict[str, Any]] = field(default_factory=list)
    turns: int = 0
    final_text: str = ""
    stop_reason: str = ""
    tool_errors: list[str] = field(default_factory=list)


class Runner:
    def __init__(
        self,
        *,
        backend_path: Path,
        fixtures: dict[str, Any],
        assertions_path: Path,
        import_root: Path,
        runtime: Any,
        worker: str = "process",
    ) -> None:
        from nemotron.steps.byob.runtime.benchmark_families.bfcl.isolation import ProcessWorker

        self._worker = ProcessWorker(default_timeout_s=runtime.episode_timeout_s, worker=worker)
        self._backend_path = backend_path
        self._fixtures = fixtures
        self._assertions_path = assertions_path
        self._import_root = import_root
        self._runtime = runtime
        self.episodes_run = 0

    def _execute(self, *, task: dict[str, Any], calls: list[ToolCall]) -> list[Any]:
        """Replay `calls` from a clean backend and return one result per call."""
        steps: list[dict[str, Any]] = [{"op": "reset"}]
        steps.extend(
            {
                "op": "call_tool",
                "name": call.name,
                "arguments": call.arguments,
                "turn_index": index,
            }
            for index, call in enumerate(calls)
        )
        runtime = self._runtime
        try:
            outputs = self._worker.run_episode(
                backend_path=self._backend_path,
                endpoint_config=None,
                fixtures=copy.deepcopy(self._fixtures),
                clock_iso=runtime.clock,
                seed=int(task.get("seed") or 0),
                task_id=str(task["task_id"]),
                steps=steps,
                assertions_path=self._assertions_path,
                import_root=self._import_root,
                import_timeout_s=runtime.import_timeout_s,
                reset_timeout_s=runtime.reset_timeout_s,
                tool_timeout_s=runtime.tool_timeout_s,
                assertion_timeout_s=runtime.assertion_timeout_s,
                episode_timeout_s=runtime.episode_timeout_s,
            )
        except Exception as error:  # noqa: BLE001 — a failed call is a result, not a stop
            return [{"error": f"{type(error).__name__}: {error}"} for _ in calls]
        self.episodes_run += 1
        return list(outputs[1:])

    def run(
        self,
        *,
        client: TargetClient,
        task: dict[str, Any],
        arm: str,
        instructions: str,
        user_turns: list[str],
        tools: list[dict[str, Any]],
    ) -> Rollout:
        out = Rollout(task_id=str(task["task_id"]), arm=arm)
        schemas = to_responses_tools(tools)
        pending = list(user_turns)
        conversation: list[dict[str, Any]] = [
            {"role": "user", "content": pending.pop(0)}
        ]
        accepted: list[ToolCall] = []

        for _ in range(MAX_MODEL_TURNS):
            out.turns += 1
            reply: Reply = client.respond(
                instructions=instructions, conversation=conversation, tools=schemas
            )

            if not reply.calls:
                out.final_text = reply.text
                if pending:
                    # The model answered instead of calling. That is the correct move for
                    # `missing_slot` and `confirmation`, so hand it the next scripted turn
                    # rather than scoring the task as finished.
                    conversation.append({"role": "assistant", "content": reply.text or "..."})
                    conversation.append({"role": "user", "content": pending.pop(0)})
                    continue
                out.stop_reason = "answered"
                break

            results = self._execute(task=task, calls=accepted + reply.calls)
            fresh = results[len(accepted) :]
            for call, result in zip(reply.calls, fresh):
                accepted.append(call)
                out.calls.append({"function_name": call.name, "arguments": call.arguments})
                if isinstance(result, dict) and result.get("error"):
                    out.tool_errors.append(f"{call.name}: {result['error']}")
                conversation.append(
                    {
                        "type": "function_call",
                        "name": call.name,
                        "arguments": json.dumps(call.arguments, ensure_ascii=False),
                        "call_id": f"call_{len(accepted)}",
                    }
                )
                conversation.append(
                    {
                        "type": "function_call_output",
                        "call_id": f"call_{len(accepted)}",
                        "output": json.dumps(result, ensure_ascii=False, default=str),
                    }
                )
        else:
            out.stop_reason = "turn_limit"

        if not out.stop_reason:
            out.stop_reason = "answered"
        return out
