from __future__ import annotations

import asyncio
import hashlib
import json
import queue
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.call_matching import (
    CanonicalCallMatchGate,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.candidate_contract import (
    CandidateAttempt,
    CandidateCallOutcome,
    CandidateResponse,
    CandidateToolCall,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.contamination_contract import (
    CandidateEligibility,
    CommonEvaluationTaskSet,
    EligibleEvalPlan,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.conversation_contract import (
    ConversationScript,
    ScriptedCall,
    ScriptedTurn,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.executable_driver import (
    run_executable_episode,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.executable_errors import (
    ExecutableAuthorizationError,
    OracleSessionError,
    OracleStateError,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.executable_projection import (
    ExecutableTaskSpec,
    ExecutableToolPolicy,
    _assertion_bindings,
    build_executable_task_spec,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.identity import (
    candidate_identity_claim,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.oracle_session import (
    _ProcessEpisodeBridge,
    open_oracle_session,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.schemas import (
    CandidateApi,
    CandidateInference,
    CandidateModelIdentity,
    EvalCandidate,
    EvalFileRef,
    EvalLimits,
    EvalScoringConfig,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.source_contract import (
    VerifiedOracleSource,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.export_contract import (
    CanonicalExportRow,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.export_projection import (
    CanonicalExportProjection,
    ProjectionSource,
    conversation_plan,
    derive_provenance,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.held_out_contract import (
    fixture_ref,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.isolation import (
    _assertion_verdict,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.pack_loader import (
    project_model_facing_tools,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.row_schema import canonical_json

HASH = "sha256:" + "1" * 64
OTHER_HASH = "sha256:" + "2" * 64
SOURCE_IDENTITY = "sha256:" + "3" * 64
TASK_ID = "task-live-1"
PACK_ROOT = (
    Path(__file__).parents[3]
    / "src"
    / "nemotron"
    / "steps"
    / "byob"
    / "data"
    / "tiny_oracle_pack"
)


def _hash(value: Any) -> str:
    return f"sha256:{hashlib.sha256(canonical_json(value).encode('utf-8')).hexdigest()}"


def _candidate() -> EvalCandidate:
    return EvalCandidate(
        alias="candidate_a",
        model="candidate-route",
        provider="nvidia",
        provider_api_version="v1",
        api=CandidateApi(
            base_url="https://candidate.example.com/v1",
            api_key_env="CANDIDATE_API_KEY",
        ),
        model_identity=CandidateModelIdentity(
            source="huggingface",
            model="org/candidate",
            revision="a" * 40,
        ),
        inference=CandidateInference(
            temperature=0.0,
            top_p=1.0,
            max_tokens=512,
            seed=7,
            tool_choice="auto",
            provider_extensions={},
        ),
    )


def _limits() -> EvalLimits:
    return EvalLimits(
        max_turns=4,
        tool_timeout_s=3.0,
        candidate_timeout_s=3.0,
        episode_timeout_s=20.0,
        max_parallel_tasks=1,
        max_retries=0,
    )


def _scoring() -> EvalScoringConfig:
    return EvalScoringConfig(
        contract=EvalFileRef(path="/contract.md", content_hash=HASH),
        argument_matching="schema_then_canonical",
        insert_declared_defaults=True,
        respect_call_order=True,
        respect_call_group=True,
        allow_llm_repair=False,
        task_success="all_applicable_gates",
    )


def _plan(*, source_task_ids_hash: str = OTHER_HASH) -> EligibleEvalPlan:
    candidate = _candidate()
    scoring = _scoring()
    eligibility = CandidateEligibility(
        alias=candidate.alias,
        identity=candidate_identity_claim(candidate),
        canonical_model_identity=candidate.canonical_model_identity,
        eligible_task_ids=(TASK_ID,),
    )
    return EligibleEvalPlan(
        eval_config_hash=HASH,
        scoring_policy_hash=scoring.scoring_policy_hash,
        source_verification_identity=SOURCE_IDENTITY,
        source_run_id="run-1",
        source_task_ids_hash=source_task_ids_hash,
        enforce=True,
        on_violation="fail_run",
        comparison_set="common_intersection",
        candidates=(eligibility,),
        common=CommonEvaluationTaskSet(
            comparison_set="common_intersection",
            task_ids=(TASK_ID,),
            candidate_aliases=(candidate.alias,),
        ),
        publication_allowed=True,
    )


def _oracle() -> VerifiedOracleSource:
    return VerifiedOracleSource(
        kind="python",
        pack_id="tiny_library",
        pack_version="0.1.0",
        expected_pack_content_hash=HASH,
        actual_pack_content_hash=HASH,
        pack_root=PACK_ROOT,
        pack_manifest_path=PACK_ROOT / "manifest.yaml",
        pack_file_count=6,
        resource_role="backend",
        resource_path=PACK_ROOT / "backend.py",
        resource_content_hash=OTHER_HASH,
        interface_probed=True,
        backend_interface=("call_tool", "get_state", "list_tools", "reset"),
    )


def _tools() -> list[dict[str, Any]]:
    full = json.loads((PACK_ROOT / "tools.json").read_text(encoding="utf-8"))
    return project_model_facing_tools(full)


def _wire_call(call_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": "get_book_status",
                    "arguments": canonical_json(arguments),
                },
            }
        ],
    }


def _row() -> CanonicalExportRow:
    arguments = {"book_id": "BK-100"}
    return CanonicalExportRow(
        task_id=TASK_ID,
        template_id="lib_status_single",
        variant_index=0,
        messages=(
            {"role": "system", "content": "You are a library assistant."},
            {"role": "user", "content": "Is BK-100 available?"},
            _wire_call("gold-call", arguments),
            {
                "role": "tool",
                "tool_call_id": "gold-call",
                "content": canonical_json({"book_id": "BK-100", "status": "gold-only"}),
            },
            {"role": "assistant", "content": "BK-100 is available."},
        ),
        tools=_tools(),
        expected_tool_calls=(
            {
                "turn_index": 0,
                "call_group": 0,
                "position_in_group": 0,
                "function_name": "get_book_status",
                "arguments": arguments,
            },
        ),
        success_assertions=("assert_book_status_reported",),
        fixture_refs=(canonical_json(["books", "BK-100"]),),
        intent="check_book_status",
        category="circulation",
        difficulty="easy",
        required_tools=("get_book_status",),
        required_tools_fingerprint=canonical_json(["get_book_status"]),
        tools_present=("get_book_status", "checkout_book"),
        turn_policy="single_turn",
        is_multi_turn=False,
        num_tool_calls=1,
        call_order="strict",
        system_prompt_id="library",
        tier="gold",
        gold_eligible=True,
        validated_by=("schema", "replay", "assertions"),
        pack_id="tiny_library",
        pack_version="0.1.0",
        seed=7,
        src="tiny_library:lib_status_single",
        metadata={
            "language": "en",
            "expt_name": "runner",
            "base_task_id": TASK_ID,
            "surface_source": "oracle",
            "profile_hash": "profile",
        },
    )


def _projection(row: CanonicalExportRow) -> CanonicalExportProjection:
    return CanonicalExportProjection(
        source=ProjectionSource(
            file="benchmark.parquet",
            content_hash=OTHER_HASH,
            rows=1,
        ),
        provenance=derive_provenance((row,)),
        rows=(row,),
        plans=(conversation_plan(row),),
    )


def _source(tmp_path: Path, oracle: VerifiedOracleSource) -> Any:
    manifest = tmp_path / "run_manifest.json"
    manifest.write_text(
        json.dumps({"oracle_clock": "2026-03-02T09:00:00+07:00"}),
        encoding="utf-8",
    )
    return SimpleNamespace(
        executable=True,
        gold_eligible=True,
        oracle=oracle,
        verification_identity=SOURCE_IDENTITY,
        eval_config_hash=HASH,
        source_run_id="run-1",
        task_ids=(TASK_ID,),
        task_index=SimpleNamespace(task_ids_hash=OTHER_HASH),
        evaluation_benchmark=SimpleNamespace(content_hash=OTHER_HASH, rows=1),
        source_manifest_path=manifest,
    )


def _script(oracle: VerifiedOracleSource) -> ConversationScript:
    tools = _tools()
    call = ScriptedCall(
        call_index=0,
        position_in_group=0,
        function_name="get_book_status",
        arguments={"book_id": "BK-100"},
        recorded_result="",
    )
    return ConversationScript(
        task_id=TASK_ID,
        source_verification_identity=SOURCE_IDENTITY,
        seed_messages=(
            {"role": "system", "content": "You are a library assistant."},
            {"role": "user", "content": "Is BK-100 available?"},
        ),
        tools=tools,
        turns=(
            ScriptedTurn(
                turn_index=0,
                user_turn_index=0,
                call_group=0,
                calls=(call,),
            ),
            ScriptedTurn(
                turn_index=1,
                user_turn_index=0,
                expected_assistant_content="BK-100 is available.",
                is_terminal=True,
            ),
        ),
        user_turns=1,
        required_tools=("get_book_status",),
        call_order="strict",
    )


def _task(oracle: VerifiedOracleSource) -> ExecutableTaskSpec:
    plan = _plan()
    return ExecutableTaskSpec(
        task_id=TASK_ID,
        candidate_alias="candidate_a",
        canonical_model_identity=_candidate().canonical_model_identity,
        plan_identity=plan.plan_identity,
        eval_config_hash=plan.eval_config_hash,
        scoring_policy_hash=plan.scoring_policy_hash,
        source_verification_identity=SOURCE_IDENTITY,
        oracle_verification_identity=oracle.verification_identity,
        source_content_hash=OTHER_HASH,
        oracle_clock="2026-03-02T09:00:00+07:00",
        seed=7,
        fixture_refs=("books:BK-100",),
        script=_script(oracle),
        success_assertions=(),
        turn_policy="single_turn",
        assistant_milestones=(
            {"type": "tool_call", "tool": "get_book_status"},
            {"type": "final_answer"},
        ),
        tool_policies=(
            ExecutableToolPolicy(
                function_name="get_book_status",
                mutates=False,
            ),
            ExecutableToolPolicy(
                function_name="checkout_book",
                mutates=True,
                requires_confirmation=True,
            ),
        ),
        assertion_task={"task_id": TASK_ID},
    )


class _FakeOracle:
    def __init__(self, identity: str) -> None:
        self.oracle_verification_identity = identity
        self.calls: list[tuple[str, dict[str, Any], int]] = []
        self.closed = False

    async def reset(self) -> None:
        return None

    async def call_tool(
        self, function_name: str, arguments: dict[str, Any], *, turn_index: int
    ) -> Any:
        self.calls.append((function_name, arguments, turn_index))
        return {"book_id": "BK-100", "status": "available"}

    async def get_state(self) -> dict[str, Any]:
        return {"books": [{"book_id": "BK-100", "status": "available"}]}

    async def run_assertion(
        self, name: str, *, task: dict[str, Any]
    ) -> dict[str, Any]:
        return {"name": name, "status": "passed", "passed": True, "detail": None}

    async def close(self) -> None:
        self.closed = True


class _TimeoutOracle(_FakeOracle):
    async def call_tool(
        self, function_name: str, arguments: dict[str, Any], *, turn_index: int
    ) -> Any:
        self.calls.append((function_name, arguments, turn_index))
        raise TimeoutError("response lost")


class _FatalThenAssertionOracle(_TimeoutOracle):
    def __init__(self, identity: str) -> None:
        super().__init__(identity)
        self.assertion_calls = 0

    async def run_assertion(
        self, name: str, *, task: dict[str, Any]
    ) -> dict[str, Any]:
        self.assertion_calls += 1
        return {
            "name": name,
            "status": "infrastructure_error",
            "passed": False,
            "detail": "must not run after fatal tool evidence",
        }


class _SlowResetStateFailingOracle(_FakeOracle):
    async def reset(self) -> None:
        await asyncio.sleep(0.05)

    async def get_state(self) -> dict[str, Any]:
        raise OracleStateError(
            "eval.oracle.state",
            "could not read the final state",
            expected="a state object",
            recovery="restart the session",
        )


class _InvalidJsonObjectOracle(_FakeOracle):
    async def call_tool(
        self, function_name: str, arguments: dict[str, Any], *, turn_index: int
    ) -> Any:
        self.calls.append((function_name, arguments, turn_index))
        return {1: "object keys must be strings"}


class _UnreadableVerdictOracle(_FakeOracle):
    async def run_assertion(self, name: str, *, task: dict[str, Any]) -> dict[str, Any]:
        return {"name": name, "outcome": "probably fine"}

    async def close(self) -> None:
        self.closed = True
        raise OracleSessionError(
            "eval.oracle.close",
            "failed during cleanup",
            expected="a closed session",
            recovery="discard the session",
        )


class _PendingMutationOracle(_FakeOracle):
    async def call_tool(
        self, function_name: str, arguments: dict[str, Any], *, turn_index: int
    ) -> Any:
        self.calls.append((function_name, arguments, turn_index))
        return {"status": "awaiting_confirmation"}


class _FakeClient:
    def __init__(self, responses: list[CandidateResponse]) -> None:
        self.responses = responses
        self.requests: list[Any] = []

    async def complete(self, request: Any, *, deadline: float) -> CandidateCallOutcome:
        del deadline
        self.requests.append(request)
        response = self.responses.pop(0)
        return CandidateCallOutcome(
            request_hash=request.request_hash,
            status="completed",
            attempts=(
                CandidateAttempt(
                    attempt_index=0,
                    observed_at="2026-01-01T00:00:00+00:00",
                    status="completed",
                    retryable=False,
                    http_status=200,
                    latency_s=0.01,
                    raw_response="{}",
                    raw_response_hash=(
                        "sha256:" + hashlib.sha256(b"{}").hexdigest()
                    ),
                ),
            ),
            response=response,
        )


def _response_with_call(arguments: dict[str, Any]) -> CandidateResponse:
    raw = canonical_json(arguments)
    return CandidateResponse(
        assistant_content=None,
        tool_calls=(
            CandidateToolCall(
                index=0,
                id="candidate-call",
                type="function",
                function_name="get_book_status",
                raw_arguments=raw,
                parsed_arguments=arguments,
                arguments_status="valid_object",
            ),
        ),
        finish_reason="tool_calls",
        selected_attempt=0,
        raw_response_hash=HASH,
    )


def _text_response(text: str) -> CandidateResponse:
    return CandidateResponse(
        assistant_content=text,
        finish_reason="stop",
        selected_attempt=0,
        raw_response_hash=HASH,
    )


def test_executable_projection_binds_plan_source_oracle_and_pack_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    oracle = _oracle()
    source = _source(tmp_path, oracle)
    row = _row()
    monkeypatch.setattr(
        "nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.executable_projection."
        "assert_source_unchanged",
        lambda _source: None,
    )

    task = build_executable_task_spec(
        _projection(row),
        TASK_ID,
        candidate_alias="candidate_a",
        source=source,
        plan=_plan(),
    )

    assert task.oracle_verification_identity == oracle.verification_identity
    assert task.script.source_verification_identity == SOURCE_IDENTITY
    assert task.tool_policy("checkout_book").mutates
    assert task.assistant_milestones[-1]["type"] == "final_answer"
    assert task.assertion_task["slots"] == {"book_id": "BK-100"}
    assert task.assertion_task["assertion_task_schema_version"] == "1.0"
    assert task.assertion_task["intent"] == "check_book_status"
    assert "user_turn_templates" in task.assertion_task
    assert "recorded_result" not in task.assertion_task
    assert "gold-only" not in canonical_json(task.model_dump(mode="json"))


def _binding_row(
    *,
    opening: str,
    surface_source: str = "template",
    expected: tuple[tuple[str, dict[str, Any]], ...] = (),
    fixture_refs: tuple[str, ...] = (),
) -> Any:
    return SimpleNamespace(
        task_id=TASK_ID,
        messages=(SimpleNamespace(role="user", content=opening),),
        metadata={"surface_source": surface_source, "language": "en"},
        expected_tool_calls=tuple(
            SimpleNamespace(
                function_name=tool,
                arguments=arguments,
                call_group=0,
                position_in_group=position,
            )
            for position, (tool, arguments) in enumerate(expected)
        ),
        fixture_refs=fixture_refs,
    )


def test_a_slot_the_expected_trace_never_names_is_read_back_from_the_surface() -> None:
    # Two calls to one tool make milestone-to-call pairing ambiguous, and the
    # tool argument is named book_id rather than either slot, so the published
    # opening turn is the only evidence that says which book is which.
    template = {
        "slots": {
            "book_a": {"source": "fixture:books.book_id"},
            "book_b": {"source": "fixture:books.book_id"},
        },
        "user_turn_templates": {"en": "Please check status for {book_a} and {book_b}."},
    }
    bindings = _assertion_bindings(
        _binding_row(
            opening="Please check status for BK-100 and BK-200.",
            expected=(
                ("get_book_status", {"book_id": "BK-100"}),
                ("get_book_status", {"book_id": "BK-200"}),
            ),
            fixture_refs=(
                canonical_json(["books", "BK-100"]),
                canonical_json(["books", "BK-200"]),
            ),
        ),
        template,
        [
            {"type": "tool_call", "tool": "get_book_status", "args": {"book_id": "{book_a}"}},
            {"type": "tool_call", "tool": "get_book_status", "args": {"book_id": "{book_b}"}},
        ],
        fixtures={
            "books": [
                {"book_id": "BK-100", "status": "available"},
                {"book_id": "BK-200", "status": "on_loan"},
            ]
        },
    )

    assert bindings["slots"] == {"book_a": "BK-100", "book_b": "BK-200"}
    assert bindings["unresolved_slots"] == []


def test_a_slot_no_fixture_holds_is_still_recovered_from_the_surface() -> None:
    # A negative-path template binds a value that deliberately exists in no
    # fixture, so only the surface can state it.
    bindings = _assertion_bindings(
        _binding_row(opening="How is transaction TXN-MISSING going?"),
        {
            "slots": {"transaction_id": {"source": "absent:transactions"}},
            "user_turn_templates": {"en": "How is transaction {transaction_id} going?"},
        },
        [],
        pack_manifest={"absent_ids": {"transactions": ["TXN-MISSING"]}},
    )

    assert bindings["slots"] == {"transaction_id": "TXN-MISSING"}
    assert bindings["unresolved_slots"] == []


def test_a_paraphrased_surface_is_never_read_back_as_a_slot_value() -> None:
    bindings = _assertion_bindings(
        _binding_row(
            opening="Hey, could you look up how BK-999 is doing?",
            surface_source="model",
        ),
        {
            "slots": {"book_id": {"source": "absent:books"}},
            "user_turn_templates": {"en": "Can you check whether book {book_id} is available?"},
        },
        [],
    )

    assert bindings["slots"] == {}
    assert bindings["unresolved_slots"] == ["book_id"]


def test_an_unrecoverable_slot_is_named_rather_than_guessed_or_refused() -> None:
    # The pack offers two literals and the row publishes nothing that picks one.
    # Projection must not invent a value, and must not deny the task either: an
    # assertion that reads the slot fails as infrastructure, one that does not
    # still runs.
    bindings = _assertion_bindings(
        _binding_row(opening="Is BK-100 available?"),
        {"slots": {"unbound": {"source": "literal:[1, 2]"}}},
        [],
    )

    assert bindings["slots"] == {}
    assert bindings["unresolved_slots"] == ["unbound"]


def test_a_surface_slot_keeps_the_declared_type_its_rendering_erased() -> None:
    bindings = _assertion_bindings(
        _binding_row(opening="Send 200000 to 970436."),
        {
            "slots": {
                "amount": {"source": "literal:[200000]"},
                "bank": {"source": "literal:['970436']"},
            },
            "user_turn_templates": {"en": "Send {amount} to {bank}."},
        },
        [],
    )

    assert bindings["slots"] == {"amount": 200000, "bank": "970436"}


def test_surface_recovery_uses_every_verified_pack_source_kind() -> None:
    bindings = _assertion_bindings(
        _binding_row(
            opening="Book The Trial, mode compact, page 2, missing 404.",
            fixture_refs=(canonical_json(["books", "BK-100"]),),
        ),
        {
            "slots": {
                "title": {"source": "fixture:books.title"},
                "mode": {"source": "enum:search_books.mode"},
                "page": {"source": "range:{'min': 1, 'max': 3}"},
                "missing": {"source": "absent:books"},
            },
            "user_turn_templates": {
                "en": "Book {title}, mode {mode}, page {page}, missing {missing}."
            },
        },
        [],
        pack_manifest={"absent_ids": {"books": [404]}},
        fixtures={"books": [{"book_id": "BK-100", "title": "The Trial"}]},
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "search_books",
                    "parameters": {
                        "type": "object",
                        "properties": {"mode": {"type": "string", "enum": ["full", "compact"]}},
                    },
                },
            }
        ],
    )

    assert bindings["slots"] == {
        "title": "The Trial",
        "mode": "compact",
        "page": 2,
        "missing": 404,
    }
    assert bindings["unresolved_slots"] == []


def test_a_declared_null_slot_is_not_confused_with_an_unresolved_value() -> None:
    bindings = _assertion_bindings(
        _binding_row(opening="Optional value: None."),
        {
            "slots": {"optional": {"source": "literal:[None]"}},
            "user_turn_templates": {"en": "Optional value: {optional}."},
        },
        [],
    )

    assert "optional" in bindings["slots"]
    assert bindings["slots"]["optional"] is None
    assert bindings["unresolved_slots"] == []


def test_multiple_corrections_leave_the_last_delivered_value_in_force() -> None:
    bindings = _assertion_bindings(
        _binding_row(opening="Use original.", expected=()),
        {
            "slots": {"value": {"source": "literal:['original']"}},
            "user_turn_templates": {"en": "Use {value}."},
            "user_simulator_turns": [
                {
                    "after": "first",
                    "slot_updates": {"value": {"source": "literal:['middle']"}},
                },
                {
                    "after": "second",
                    "slot_updates": {"value": {"source": "literal:['final']"}},
                },
            ],
        },
        [{"id": "first"}, {"id": "second"}],
    )

    assert bindings["slots_initial"] == {"value": "original"}
    assert bindings["slot_updates"] == [
        {"entry_index": 0, "values": {"value": "middle"}, "aliases": {}},
        {"entry_index": 1, "values": {"value": "final"}, "aliases": {}},
    ]
    assert bindings["slots"] == {"value": "final"}
    assert bindings["unresolved_slots"] == []


def test_an_unknown_initial_value_is_distinct_from_a_known_corrected_value() -> None:
    bindings = _assertion_bindings(
        _binding_row(
            opening="Use the value I mentioned.",
            surface_source="model",
            expected=(("set_value", {"value": 20}),),
        ),
        {
            "slots": {"value": {"source": "literal:[10, 15]"}},
            "user_simulator_turns": [
                {
                    "after": "correction",
                    "slot_updates": {"value": {"source": "literal:[20]"}},
                }
            ],
        },
        [
            {"id": "correction"},
            {
                "type": "tool_call",
                "tool": "set_value",
                "args": {"value": "{value}"},
                "call_group": 0,
            },
        ],
    )

    assert bindings["slots"] == {"value": 20}
    assert bindings["slots_initial"] == {}
    assert bindings["unresolved_slots"] == []
    assert bindings["unresolved_slots_initial"] == ["value"]


def test_an_unknown_correction_keeps_its_timeline_position() -> None:
    bindings = _assertion_bindings(
        _binding_row(opening="Use original."),
        {
            "slots": {"value": {"source": "literal:['original']"}},
            "user_turn_templates": {"en": "Use {value}."},
            "user_simulator_turns": [
                {
                    "after": "correction",
                    "slot_updates": {
                        "value": {"source": "literal:['middle', 'final']"}
                    },
                }
            ],
        },
        [{"id": "correction"}],
    )

    assert bindings["slot_updates"] == [
        {"entry_index": 0, "values": {}, "aliases": {}}
    ]
    assert bindings["unresolved_slot_updates"] == [
        {"update_index": 0, "entry_index": 0, "slots": ["value"]}
    ]


def test_repeated_tool_milestones_pair_by_call_group_position() -> None:
    bindings = _assertion_bindings(
        _binding_row(
            opening="Check the hidden books.",
            surface_source="model",
            expected=(
                ("get_book_status", {"book_id": "BK-100"}),
                ("get_book_status", {"book_id": "BK-200"}),
            ),
        ),
        {
            "slots": {
                "book_a": {"source": "absent:books"},
                "book_b": {"source": "absent:books"},
            }
        },
        [
            {
                "type": "tool_call",
                "tool": "get_book_status",
                "args": {"book_id": "{book_a}"},
                "call_group": 0,
            },
            {
                "type": "tool_call",
                "tool": "get_book_status",
                "args": {"book_id": "{book_b}"},
                "call_group": 0,
            },
        ],
    )

    assert bindings["slots"] == {"book_a": "BK-100", "book_b": "BK-200"}


def test_assertion_reading_an_unresolved_slot_is_infrastructure() -> None:
    def needs_slot(*, state: dict, trace: list, task: dict, ctx: Any) -> None:
        if task.get("slots", {}).get("book_id") is None:
            raise AssertionError("candidate omitted the expected book")

    verdict = _assertion_verdict(
        needs_slot,
        name="needs_slot",
        state={},
        trace=[],
        task={"slots": {}, "unresolved_slots": ["book_id"]},
        ctx=None,
    )

    assert verdict["status"] == "infrastructure_error"
    assert "book_id" in verdict["detail"]


def test_unresolved_slot_unrelated_to_an_assertion_does_not_mask_failure() -> None:
    def no_tools(*, state: dict, trace: list, task: dict, ctx: Any) -> None:
        if trace:
            raise AssertionError("a tool was called")

    verdict = _assertion_verdict(
        no_tools,
        name="no_tools",
        state={},
        trace=[{"tool": "get_book_status"}],
        task={"slots": {}, "unresolved_slots": ["book_id"]},
        ctx=None,
    )

    assert verdict["status"] == "failed"


def test_an_empty_slots_object_still_reports_the_slot_an_assertion_needed() -> None:
    # Packs read slots through `task.get("slots") or {}`. An empty object is
    # falsy, so the fallback literal escapes read tracking unless the parent
    # records the whole child. Without that, an unbindable slot is charged to
    # the candidate as a failure instead of to the evaluator.
    def needs_slot(*, state: dict, trace: list, task: dict, ctx: Any) -> None:
        slots = task.get("slots") or {}
        if slots.get("transaction_id") is None:
            raise AssertionError("candidate never named the transaction")

    verdict = _assertion_verdict(
        needs_slot,
        name="needs_slot",
        state={},
        trace=[],
        task={"slots": {}, "unresolved_slots": ["transaction_id"]},
        ctx=None,
    )

    assert verdict["status"] == "infrastructure_error"
    assert "transaction_id" in verdict["detail"]


def test_reading_an_unknown_initial_slot_is_infrastructure() -> None:
    def needs_initial(*, state: dict, trace: list, task: dict, ctx: Any) -> None:
        initial = task.get("slots_initial") or {}
        if initial.get("amount") is None:
            raise AssertionError("the original amount is missing")

    verdict = _assertion_verdict(
        needs_initial,
        name="needs_initial",
        state={},
        trace=[],
        task={
            "slots": {"amount": 20},
            "slots_initial": {},
            "unresolved_slots": [],
            "unresolved_slots_initial": ["amount"],
        },
        ctx=None,
    )

    assert verdict["status"] == "infrastructure_error"
    assert "slots_initial.amount" in verdict["detail"]


def test_all_dictionary_reads_preserve_unresolved_slot_evidence() -> None:
    def by_length(*, state: dict, trace: list, task: dict, ctx: Any) -> None:
        if len(task["slots"]) != 2:
            raise AssertionError("a slot is missing")

    def by_inequality(*, state: dict, trace: list, task: dict, ctx: Any) -> None:
        if task["slots"] != {"known": 1, "missing": 2}:
            raise AssertionError("a slot is missing")

    def by_pop(*, state: dict, trace: list, task: dict, ctx: Any) -> None:
        if task["slots"].pop("missing", None) is None:
            raise AssertionError("a slot is missing")

    def by_popitem(*, state: dict, trace: list, task: dict, ctx: Any) -> None:
        task["slots"].popitem()
        raise AssertionError("the complete mapping was required")

    for assertion in (by_length, by_inequality, by_pop, by_popitem):
        verdict = _assertion_verdict(
            assertion,
            name=assertion.__name__,
            state={},
            trace=[],
            task={"slots": {"known": 1}, "unresolved_slots": ["missing"]},
            ctx=None,
        )
        assert verdict["status"] == "infrastructure_error"
        assert "slots.missing" in verdict["detail"]


def test_reading_an_unresolved_slot_update_is_infrastructure() -> None:
    def needs_update(*, state: dict, trace: list, task: dict, ctx: Any) -> None:
        values = task["slot_updates"][0].get("values") or {}
        if values.get("amount") is None:
            raise AssertionError("the corrected amount is missing")

    verdict = _assertion_verdict(
        needs_update,
        name="needs_update",
        state={},
        trace=[],
        task={
            "slot_updates": [{"entry_index": 0, "values": {}, "aliases": {}}],
            "unresolved_slot_updates": [
                {"update_index": 0, "entry_index": 0, "slots": ["amount"]}
            ],
        },
        ctx=None,
    )

    assert verdict["status"] == "infrastructure_error"
    assert "slot_updates[0].values.amount" in verdict["detail"]


def test_a_fixture_slot_binds_when_the_collection_keys_rows_by_number() -> None:
    # fixture_ref renders the cited primary id with str, so a numeric key is
    # published as text while the fixture row keeps the number.
    bindings = _assertion_bindings(
        _binding_row(
            opening="Check book The Trial.",
            fixture_refs=(fixture_ref("books", 100),),
        ),
        {
            "slots": {"title": {"source": "fixture:books.title"}},
            "user_turn_templates": {"en": "Check book {title}."},
        },
        [],
        fixtures={"books": [{"book_id": 100, "title": "The Trial"}]},
    )

    assert bindings["slots"] == {"title": "The Trial"}
    assert bindings["unresolved_slots"] == []


def test_fixture_shorthand_and_filters_match_generation_semantics() -> None:
    bindings = _assertion_bindings(
        _binding_row(
            opening="Check the selected books.",
            surface_source="model",
            fixture_refs=(
                fixture_ref("books", "BK-100"),
                fixture_ref("books", "BK-200"),
            ),
        ),
        {
            "slots": {
                "available_book": {
                    "source": "books.book_id",
                    "filter": "status == 'available'",
                },
                "loaned_book": {
                    "source": "books.book_id",
                    "filter": "status == 'on_loan'",
                },
            }
        },
        [],
        fixtures={
            "books": [
                {"book_id": "BK-100", "status": "available"},
                {"book_id": "BK-200", "status": "on_loan"},
            ]
        },
    )

    assert bindings["slots"] == {
        "available_book": "BK-100",
        "loaned_book": "BK-200",
    }
    assert bindings["unresolved_slots"] == []


async def _a_state_failure_keeps_the_reason_the_episode_already_ended_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    oracle_source = _oracle()
    source = _source(tmp_path, oracle_source)
    monkeypatch.setattr(
        "nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.executable_driver."
        "assert_source_unchanged",
        lambda _source: None,
    )

    return await run_executable_episode(
        candidate=_candidate(),
        limits=EvalLimits(
            max_turns=4,
            tool_timeout_s=0.01,
            candidate_timeout_s=0.01,
            episode_timeout_s=0.01,
            max_parallel_tasks=1,
            max_retries=0,
        ),
        client=_FakeClient([]),  # type: ignore[arg-type]
        task=_task(oracle_source),
        source=source,
        plan=_plan(),
        oracle=_SlowResetStateFailingOracle(oracle_source.verification_identity),
        gate=CanonicalCallMatchGate(_scoring()),
    )


def test_a_state_failure_keeps_the_reason_the_episode_already_ended_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The deadline elapses before the candidate is asked anything, so the
    # episode ends with no observed turn. Adopting the later state-snapshot
    # failure as the terminal status would describe an episode shape the
    # contract forbids, and the whole record would be lost to a validation
    # error instead of reporting the timeout that actually ended it.
    episode = asyncio.run(
        _a_state_failure_keeps_the_reason_the_episode_already_ended_for(
            tmp_path, monkeypatch
        )
    )

    assert episode.status == "episode_timeout"
    assert episode.observed == ()
    assert episode.final_state_hash is None


def test_closing_a_failed_bridge_still_signals_its_episode_iterator() -> None:
    class FinishedThread:
        def join(self, timeout: float) -> None:
            self.timeout = timeout

        def is_alive(self) -> bool:
            return False

    bridge = object.__new__(_ProcessEpisodeBridge)
    bridge._exchange_lock = threading.Lock()
    bridge._closed = True
    bridge._close_requested = False
    bridge._started = True
    bridge._commands = queue.Queue()
    bridge._thread = FinishedThread()
    bridge._limits = SimpleNamespace(tool_timeout_s=1)

    bridge._close_sync()

    assert bridge._commands.get_nowait() is None


async def _fatal_tool_status_is_not_overwritten_by_later_assertions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    oracle_source = _oracle()
    source = _source(tmp_path, oracle_source)
    task = _task(oracle_source).model_copy(
        update={"success_assertions": ("assert_book_status_reported",)}
    )
    oracle = _FatalThenAssertionOracle(oracle_source.verification_identity)
    monkeypatch.setattr(
        "nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.executable_driver."
        "assert_source_unchanged",
        lambda _source: None,
    )

    episode = await run_executable_episode(
        candidate=_candidate(),
        limits=_limits(),
        client=_FakeClient(  # type: ignore[arg-type]
            [_response_with_call({"book_id": "BK-100"})]
        ),
        task=task,
        source=source,
        plan=_plan(),
        oracle=oracle,
        gate=CanonicalCallMatchGate(_scoring()),
    )

    assert episode.status == "oracle_timeout"
    assert episode.assertions == ()
    assert oracle.assertion_calls == 0


def test_a_fatal_tool_status_is_not_overwritten_by_later_assertions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asyncio.run(
        _fatal_tool_status_is_not_overwritten_by_later_assertions(
            tmp_path, monkeypatch
        )
    )


async def _python_oracle_session_keeps_state_in_one_isolated_worker(
    tmp_path: Path,
) -> None:
    oracle = _oracle()
    source = _source(tmp_path, oracle)
    task = _task(oracle)
    session = open_oracle_session(
        source=source,
        task=task,
        limits=_limits(),
    )
    assert not session._bridge._started
    assert not session._bridge._thread.is_alive()
    try:
        await session.reset()
        result = await session.call_tool(
            "get_book_status",
            {"book_id": "BK-100"},
            turn_index=0,
        )
        state = await session.get_state()
    finally:
        await session.close()

    assert result["book_id"] == "BK-100"
    assert state["books"][0]["book_id"] == "BK-100"


def test_python_oracle_session_keeps_state_in_one_isolated_worker(
    tmp_path: Path,
) -> None:
    asyncio.run(_python_oracle_session_keeps_state_in_one_isolated_worker(tmp_path))


async def _projected_task_runs_its_pack_assertions_in_the_isolated_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    oracle_source = _oracle()
    source = _source(tmp_path, oracle_source)
    plan = _plan()
    for module in ("executable_projection", "executable_driver"):
        monkeypatch.setattr(
            "nemotron.steps.byob.runtime.benchmark_families.bfcl.eval."
            f"{module}.assert_source_unchanged",
            lambda _source: None,
        )
    task = build_executable_task_spec(
        _projection(_row()),
        TASK_ID,
        candidate_alias="candidate_a",
        source=source,
        plan=plan,
    )
    session = open_oracle_session(source=source, task=task, limits=_limits())
    client = _FakeClient(
        [
            _response_with_call({"book_id": "BK-100"}),
            _text_response("BK-100 is available."),
        ]
    )

    episode = await run_executable_episode(
        candidate=_candidate(),
        limits=_limits(),
        client=client,  # type: ignore[arg-type]
        task=task,
        source=source,
        plan=plan,
        oracle=session,
        gate=CanonicalCallMatchGate(_scoring()),
    )

    assert episode.status == "completed"
    assert [outcome.status for outcome in episode.assertions] == ["passed"]


def test_projected_task_runs_its_pack_assertions_in_the_isolated_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asyncio.run(
        _projected_task_runs_its_pack_assertions_in_the_isolated_session(
            tmp_path, monkeypatch
        )
    )


async def _live_driver_returns_live_result_to_candidate_not_recorded_gold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    oracle_source = _oracle()
    source = _source(tmp_path, oracle_source)
    task = _task(oracle_source)
    plan = _plan()
    oracle = _FakeOracle(oracle_source.verification_identity)
    client = _FakeClient(
        [
            _response_with_call({"book_id": "BK-100"}),
            _text_response("BK-100 is available."),
        ]
    )
    monkeypatch.setattr(
        "nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.executable_driver."
        "assert_source_unchanged",
        lambda _source: None,
    )

    episode = await run_executable_episode(
        candidate=_candidate(),
        limits=_limits(),
        client=client,  # type: ignore[arg-type]
        task=task,
        source=source,
        plan=plan,
        oracle=oracle,
        gate=CanonicalCallMatchGate(_scoring()),
    )

    second_messages = client.requests[1].body["messages"]
    tool_message = next(message for message in second_messages if message["role"] == "tool")
    assert json.loads(tool_message["content"]) == {
        "book_id": "BK-100",
        "status": "available",
    }
    assert "gold-only" not in canonical_json(second_messages)
    assert episode.status == "completed"
    assert episode.executions[0].released_to_model
    assert episode.final_state_hash is not None
    assert oracle.closed


def test_live_driver_returns_live_result_to_candidate_not_recorded_gold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asyncio.run(
        _live_driver_returns_live_result_to_candidate_not_recorded_gold(
            tmp_path, monkeypatch
        )
    )


async def _authorization_failure_closes_the_open_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    oracle_source = _oracle()
    source = _source(tmp_path, oracle_source)
    source.verification_identity = "sha256:" + "b" * 64
    oracle = _FakeOracle(oracle_source.verification_identity)
    monkeypatch.setattr(
        "nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.executable_driver."
        "assert_source_unchanged",
        lambda _source: None,
    )

    with pytest.raises(ExecutableAuthorizationError, match="different verified source"):
        await run_executable_episode(
            candidate=_candidate(),
            limits=_limits(),
            client=_FakeClient([]),  # type: ignore[arg-type]
            task=_task(oracle_source),
            source=source,
            plan=_plan(),
            oracle=oracle,
            gate=CanonicalCallMatchGate(_scoring()),
        )

    assert oracle.closed


def test_authorization_failure_closes_the_open_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asyncio.run(_authorization_failure_closes_the_open_session(tmp_path, monkeypatch))


async def _schema_invalid_candidate_call_is_recorded_but_not_executed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    oracle_source = _oracle()
    source = _source(tmp_path, oracle_source)
    task = _task(oracle_source)
    oracle = _FakeOracle(oracle_source.verification_identity)
    client = _FakeClient([_response_with_call({})])
    monkeypatch.setattr(
        "nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.executable_driver."
        "assert_source_unchanged",
        lambda _source: None,
    )

    episode = await run_executable_episode(
        candidate=_candidate(),
        limits=_limits(),
        client=client,  # type: ignore[arg-type]
        task=task,
        source=source,
        plan=_plan(),
        oracle=oracle,
        gate=CanonicalCallMatchGate(_scoring()),
    )

    assert episode.status == "candidate_mismatch"
    assert episode.executions[0].status == "not_executed"
    assert episode.executions[0].schema_valid is False
    assert not oracle.calls


def test_schema_invalid_candidate_call_is_recorded_but_not_executed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asyncio.run(
        _schema_invalid_candidate_call_is_recorded_but_not_executed(
            tmp_path, monkeypatch
        )
    )


async def _invalid_json_object_result_is_recorded_as_an_oracle_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    oracle_source = _oracle()
    source = _source(tmp_path, oracle_source)
    monkeypatch.setattr(
        "nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.executable_driver."
        "assert_source_unchanged",
        lambda _source: None,
    )

    episode = await run_executable_episode(
        candidate=_candidate(),
        limits=_limits(),
        client=_FakeClient(  # type: ignore[arg-type]
            [_response_with_call({"book_id": "BK-100"})]
        ),
        task=_task(oracle_source),
        source=source,
        plan=_plan(),
        oracle=_InvalidJsonObjectOracle(oracle_source.verification_identity),
        gate=CanonicalCallMatchGate(_scoring()),
    )

    assert episode.status == "oracle_call_failed"
    assert episode.executions[0].status == "tool_error"
    assert episode.executions[0].result is None


def test_invalid_json_object_result_is_recorded_as_an_oracle_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asyncio.run(
        _invalid_json_object_result_is_recorded_as_an_oracle_failure(
            tmp_path, monkeypatch
        )
    )


async def _cleanup_failure_does_not_replace_a_candidate_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    oracle_source = _oracle()
    source = _source(tmp_path, oracle_source)
    oracle = _UnreadableVerdictOracle(oracle_source.verification_identity)
    monkeypatch.setattr(
        "nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.executable_driver."
        "assert_source_unchanged",
        lambda _source: None,
    )

    episode = await run_executable_episode(
        candidate=_candidate(),
        limits=_limits(),
        client=_FakeClient([_response_with_call({})]),  # type: ignore[arg-type]
        task=_task(oracle_source),
        source=source,
        plan=_plan(),
        oracle=oracle,
        gate=CanonicalCallMatchGate(_scoring()),
    )

    assert episode.status == "candidate_mismatch"
    assert episode.reason_code == "episode.candidate_mismatch"
    assert oracle.closed


def test_cleanup_failure_does_not_replace_a_candidate_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asyncio.run(
        _cleanup_failure_does_not_replace_a_candidate_outcome(
            tmp_path, monkeypatch
        )
    )


async def _unreadable_assertion_verdict_keeps_the_episode_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    oracle_source = _oracle()
    source = _source(tmp_path, oracle_source)
    task = _task(oracle_source).model_copy(
        update={"success_assertions": ("assert_book_status_reported",)}
    )
    oracle = _UnreadableVerdictOracle(oracle_source.verification_identity)
    monkeypatch.setattr(
        "nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.executable_driver."
        "assert_source_unchanged",
        lambda _source: None,
    )

    episode = await run_executable_episode(
        candidate=_candidate(),
        limits=_limits(),
        client=_FakeClient(  # type: ignore[arg-type]
            [
                _response_with_call({"book_id": "BK-100"}),
                _text_response("BK-100 is available."),
            ]
        ),
        task=task,
        source=source,
        plan=_plan(),
        oracle=oracle,
        gate=CanonicalCallMatchGate(_scoring()),
    )

    assert episode.status == "assertion_infrastructure_failed"
    assert [outcome.status for outcome in episode.assertions] == ["infrastructure_error"]
    assert oracle.closed


def test_an_unreadable_assertion_verdict_keeps_the_episode_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asyncio.run(
        _unreadable_assertion_verdict_keeps_the_episode_evidence(tmp_path, monkeypatch)
    )


async def _pending_mutation_is_not_recorded_as_committed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    oracle_source = _oracle()
    source = _source(tmp_path, oracle_source)
    task = _task(oracle_source).model_copy(
        update={
            "tool_policies": (
                ExecutableToolPolicy(
                    function_name="get_book_status",
                    mutates=True,
                ),
                ExecutableToolPolicy(
                    function_name="checkout_book",
                    mutates=True,
                    requires_confirmation=True,
                ),
            )
        }
    )
    oracle = _PendingMutationOracle(oracle_source.verification_identity)
    client = _FakeClient(
        [
            _response_with_call({"book_id": "BK-100"}),
            _text_response("BK-100 is available."),
        ]
    )
    monkeypatch.setattr(
        "nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.executable_driver."
        "assert_source_unchanged",
        lambda _source: None,
    )

    episode = await run_executable_episode(
        candidate=_candidate(),
        limits=_limits(),
        client=client,  # type: ignore[arg-type]
        task=task,
        source=source,
        plan=_plan(),
        oracle=oracle,
        gate=CanonicalCallMatchGate(_scoring()),
    )

    assert episode.status == "completed"
    assert episode.executions[0].state_commit == "not_committed"
    assert episode.executions[0].state_before_hash is not None
    assert (
        episode.executions[0].state_before_hash
        == episode.executions[0].state_after_hash
    )


def test_a_pending_mutation_is_not_recorded_as_committed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asyncio.run(_pending_mutation_is_not_recorded_as_committed(tmp_path, monkeypatch))


async def _terminal_tool_call_can_complete_without_releasing_its_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    oracle_source = _oracle()
    source = _source(tmp_path, oracle_source)
    task = _task(oracle_source)
    terminal_call = task.script.turns[0].model_copy(update={"is_terminal": True})
    task = task.model_copy(
        update={"script": task.script.model_copy(update={"turns": (terminal_call,)})}
    )
    oracle = _FakeOracle(oracle_source.verification_identity)
    monkeypatch.setattr(
        "nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.executable_driver."
        "assert_source_unchanged",
        lambda _source: None,
    )

    episode = await run_executable_episode(
        candidate=_candidate(),
        limits=_limits(),
        client=_FakeClient(  # type: ignore[arg-type]
            [_response_with_call({"book_id": "BK-100"})]
        ),
        task=task,
        source=source,
        plan=_plan(),
        oracle=oracle,
        gate=CanonicalCallMatchGate(_scoring()),
    )

    assert episode.status == "completed"
    assert episode.observed[0].advanced
    assert not episode.executions[0].released_to_model


def test_a_terminal_tool_call_can_complete_without_releasing_its_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asyncio.run(
        _terminal_tool_call_can_complete_without_releasing_its_result(
            tmp_path, monkeypatch
        )
    )


async def _mutating_timeout_is_not_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    oracle_source = _oracle()
    source = _source(tmp_path, oracle_source)
    task = _task(oracle_source).model_copy(
        update={
            "tool_policies": (
                ExecutableToolPolicy(
                    function_name="get_book_status",
                    mutates=True,
                ),
                ExecutableToolPolicy(
                    function_name="checkout_book",
                    mutates=True,
                    requires_confirmation=True,
                ),
            )
        }
    )
    oracle = _TimeoutOracle(oracle_source.verification_identity)
    client = _FakeClient([_response_with_call({"book_id": "BK-100"})])
    monkeypatch.setattr(
        "nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.executable_driver."
        "assert_source_unchanged",
        lambda _source: None,
    )

    episode = await run_executable_episode(
        candidate=_candidate(),
        limits=_limits(),
        client=client,  # type: ignore[arg-type]
        task=task,
        source=source,
        plan=_plan(),
        oracle=oracle,
        gate=CanonicalCallMatchGate(_scoring()),
    )

    assert len(oracle.calls) == 1
    assert episode.status == "unknown_commit_state"
    assert episode.executions[0].status == "unknown_commit_state"
    assert episode.executions[0].state_commit == "unknown"


def test_a_mutating_timeout_is_not_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asyncio.run(_mutating_timeout_is_not_retried(tmp_path, monkeypatch))
