"""Path 1 adapter metrics and path 2 mocked Guardrails scoring."""

import json
from pathlib import Path

import httpx
import pytest
import yaml

from nemotron.steps.byob.runtime.benchmark_families.safety import extract
from nemotron.steps.byob.runtime.benchmark_families.safety.eval import (
    PATH1_FILE,
    PATH2_FILE,
    SMOKE_FILE,
    EvalInputError,
    TestAdapter,
    _smoke_adapter,
    events_from_case,
    percentile,
    score_path1,
    score_path2,
    score_reply,
    write_evaluation,
)
from nemotron.steps.byob.runtime.benchmark_families.safety.eval_adapters import (
    NemoGuardrailsAdapter,
    OpenAICompatibleAdapter,
)
from nemotron.steps.byob.runtime.benchmark_families.safety.ir import ACTIONS
from nemotron.steps.byob.runtime.benchmark_families.safety.pipeline import sign_safety_packet
from nemotron.steps.byob.runtime.benchmark_families.safety.status import RunStatus
from tests.steps.byob.test_safety_family import _cli, _config, _EchoExtractor, _expect, _prepared, _run


@pytest.fixture(autouse=True)
def _fake_extractor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(extract, "make_extractor", lambda config, run_dir: _EchoExtractor())


def _case(case_id: str, action: str, text: str = "hello") -> dict:
    return {
        "case_id": case_id,
        "policy_id": "P-1",
        "kind": "clear_violation" if action != "allow" else "missing_leg",
        "expected_action": action,
        "turns": [{"turn": 1, "role": "user", "text": text, "scored": True, "expected_action": action}],
    }


def _reply(event_id: str, **fields: object) -> dict:
    payload = {"event_id": event_id, "latency_ms": 10, **fields}
    return payload


def test_path1_metrics_match_hand_computed_fixture() -> None:
    events = [
        events_from_case(_case("u1", "block"))[0],
        events_from_case(_case("u2", "block"))[0],
        events_from_case(_case("b1", "allow"))[0],
        events_from_case(_case("b2", "allow"))[0],
        events_from_case(_case("m1", "block"))[0],
        events_from_case(_case("t1", "block"))[0],
        events_from_case(_case("e1", "block"))[0],
    ]
    adapter = TestAdapter(
        {
            "u1#1": {"decision": "block", "latency_ms": 10},
            "u2#1": {"decision": "allow", "latency_ms": 20},
            "b1#1": {"decision": "allow", "latency_ms": 30},
            "b2#1": {"decision": "block", "latency_ms": 40},
            "m1#1": {"parse_status": "malformed", "latency_ms": 50},
            "t1#1": {"parse_status": "timeout", "latency_ms": 60},
            "e1#1": {"parse_status": "model_error", "latency_ms": 70},
        }
    )

    metrics = score_path1(events, adapter, split="fixture")

    assert metrics["path"] == "path1"
    assert metrics["status"] == "completed"
    assert metrics["events"] == 7
    assert metrics["scored"] == 4
    assert metrics["true_positives"] == 1
    assert metrics["false_negatives"] == 1
    assert metrics["true_negatives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["unsafe_recall"] == 0.5
    assert metrics["false_negative_rate"] == 0.5
    assert metrics["unsafe_precision"] == 0.5
    assert metrics["benign_specificity"] == 0.5
    assert metrics["false_positive_rate"] == 0.5
    assert metrics["over_refusal"] == 0.5
    assert metrics["binary_macro_f1"] == 0.5
    assert metrics["exact_action_accuracy"] == 0.5
    assert metrics["parse_success_rate"] == 4 / 7
    assert metrics["parse_failures"] == 1
    assert metrics["timeouts"] == 1
    assert metrics["rate_limits"] == 0
    assert metrics["authentication_errors"] == 0
    assert metrics["model_errors"] == 1
    assert metrics["latency_ms"]["mean"] == 40
    assert metrics["latency_ms"]["p50"] == 40
    assert metrics["latency_ms"]["p95"] == 60
    assert metrics["latency_ms"]["p99"] == 60
    assert metrics["by_policy"]["P-1"]["events"] == 7
    assert len(metrics["comparisons"]) == 7
    assert metrics["wall_time_seconds"] >= 0
    assert metrics["throughput_events_per_second"] > 0


def test_failures_are_not_counted_as_allow_or_block() -> None:
    event = events_from_case(_case("u1", "block"))[0]
    malformed = score_reply(event, {"parse_status": "malformed", "decision": "allow", "latency_ms": 1})
    timeout = score_reply(event, {"parse_status": "timeout", "latency_ms": 2})
    error = score_reply(event, {"parse_status": "model_error", "latency_ms": 3})

    for row in (malformed, timeout, error):
        assert row["scored"] is False
        assert row["true_positive"] is False
        assert row["false_negative"] is False
        assert row["match"] is None


def test_path2_compares_events_and_records_rail_timing() -> None:
    events = [events_from_case(_case("u1", "block"))[0], events_from_case(_case("b1", "allow"))[0]]
    log = [
        _reply(
            "u1#1",
            decision="block",
            rails=[
                {"name": "input", "activated": True, "duration_ms": 5},
                {"name": "output", "activated": True, "duration_ms": 2},
            ],
        ),
        _reply(
            "b1#1",
            decision="block",
            rails=[{"name": "input", "activated": True, "duration_ms": 8}],
        ),
    ]

    metrics = score_path2(events, log, split="fixture")

    assert metrics["path"] == "path2"
    assert metrics["agreement"] == 0.5
    assert metrics["over_refusal"] == 1.0
    assert metrics["unsafe_recall"] == 1.0
    assert metrics["rails"]["input"]["activations"] == 2
    assert metrics["rails"]["input"]["duration_ms"]["total"] == 13
    assert metrics["rails"]["input"]["duration_ms"]["mean"] == 6.5
    assert metrics["rails"]["input"]["duration_ms"]["p50"] == 5
    assert metrics["rails"]["output"]["activations"] == 1
    assert {row["event_id"] for row in metrics["comparisons"]} == {"u1#1", "b1#1"}


def test_path2_refuses_a_log_that_does_not_cover_the_events() -> None:
    events = [events_from_case(_case("u1", "block"))[0]]
    try:
        score_path2(events, [_reply("other#1", decision="block")], split="fixture")
    except EvalInputError as exc:
        assert "does not match scored events" in str(exc)
        assert "u1#1" in str(exc)
    else:
        raise AssertionError("a mismatched Guardrails log was accepted")


def test_metric_files_are_separate_and_do_not_overwrite_each_other(tmp_path: Path) -> None:
    write_evaluation(tmp_path, {"path": "path1", "marker": "one"}, {"path": "path2", "marker": "two"})
    first = (tmp_path / PATH1_FILE).read_text(encoding="utf-8")
    write_evaluation(tmp_path, {"path": "path1", "marker": "one"}, {"path": "path2", "marker": "changed"})

    assert json.loads(first) == {"path": "path1", "marker": "one"}
    assert json.loads((tmp_path / PATH1_FILE).read_text(encoding="utf-8"))["marker"] == "one"
    assert json.loads((tmp_path / PATH2_FILE).read_text(encoding="utf-8"))["marker"] == "changed"
    assert PATH1_FILE != PATH2_FILE


def test_percentile_uses_the_documented_index() -> None:
    assert percentile([10, 20, 30, 40, 50, 60, 70], 50) == 40
    assert percentile([10, 20, 30, 40, 50, 60, 70], 95) == 60
    assert percentile([], 50) is None


def test_eval_stage_scores_a_published_benchmark(tmp_path: Path) -> None:
    config, run_dir, _, _ = _prepared(tmp_path)
    _expect(config, "generate", RunStatus.REVIEW_REQUIRED)
    sign_safety_packet(config, approved_by="Lan")
    _expect(config, "generate", RunStatus.GOLD_COMPLETED)
    cases = _published_cases(run_dir)
    responses, log_path = _scripted_eval_inputs(tmp_path, cases)
    eval_config = _eval_config(tmp_path, responses, log_path)

    output = _run(eval_config, "eval")

    assert output == run_dir
    path1 = json.loads((run_dir / PATH1_FILE).read_text(encoding="utf-8"))
    path2 = json.loads((run_dir / PATH2_FILE).read_text(encoding="utf-8"))
    assert path1["status"] == "completed"
    assert path2["status"] == "completed"
    assert path1["parse_failures"] == 0
    assert path1["timeouts"] == 0
    assert path1["model_errors"] == 0
    if path1["unsafe"]:
        assert path1["unsafe_recall"] == 1.0
    if path1["benign"]:
        assert path1["benign_specificity"] == 1.0
        assert path1["over_refusal"] == 0.0
    if path2["scored"]:
        assert path2["agreement"] == 1.0
    assert path2["rails"]["input"]["activations"] == path2["events"]
    assert path1["path"] == "path1"
    assert path2["path"] == "path2"
    record = json.loads((run_dir / "run_record.json").read_text(encoding="utf-8"))
    assert record["publication_commit"] is True
    assert record["evaluation"] == {"path1": "completed", "path2": "completed"}


def test_eval_without_a_published_benchmark_is_refused(tmp_path: Path) -> None:
    config, _, _, _ = _prepared(tmp_path)

    caught = _expect(config, "eval", RunStatus.INVALID_RUN_ARTIFACT)

    assert "published benchmark" in str(caught)


def test_eval_cli_writes_both_metric_files(tmp_path: Path) -> None:
    config, run_dir, _, _ = _prepared(tmp_path)
    _expect(config, "generate", RunStatus.REVIEW_REQUIRED)
    sign_safety_packet(config, approved_by="Lan")
    _expect(config, "generate", RunStatus.GOLD_COMPLETED)
    cases = _published_cases(run_dir)
    responses, log_path = _scripted_eval_inputs(tmp_path, cases)
    eval_config = _eval_config(tmp_path, responses, log_path)

    result = _cli(eval_config, "eval")

    assert result.returncode == 0, result.stderr
    assert "Traceback" not in result.stderr
    assert (run_dir / PATH1_FILE).is_file()
    assert (run_dir / PATH2_FILE).is_file()


def test_eval_not_run_when_paths_are_omitted(tmp_path: Path) -> None:
    config, run_dir, _, _ = _prepared(tmp_path)
    _expect(config, "generate", RunStatus.REVIEW_REQUIRED)
    sign_safety_packet(config, approved_by="Lan")
    _expect(config, "generate", RunStatus.GOLD_COMPLETED)

    _run(config, "eval")

    path1 = json.loads((run_dir / PATH1_FILE).read_text(encoding="utf-8"))
    path2 = json.loads((run_dir / PATH2_FILE).read_text(encoding="utf-8"))
    assert path1["status"] == "not_run"
    assert path2["status"] == "not_run"
    assert "path1" in path1["reason"]
    assert "path2" in path2["reason"]


def test_prepare_clears_previous_evaluation_files(tmp_path: Path) -> None:
    config, run_dir, _, _ = _prepared(tmp_path)
    write_evaluation(run_dir, {"path": "path1"}, {"path": "path2"})
    (run_dir / SMOKE_FILE).write_text("{}\n", encoding="utf-8")

    _run(config, "prepare")

    assert not (run_dir / PATH1_FILE).exists()
    assert not (run_dir / PATH2_FILE).exists()
    assert not (run_dir / SMOKE_FILE).exists()


def test_adapter_smoke_uses_only_synthetic_input_and_writes_no_prompt(tmp_path: Path) -> None:
    seen: list[dict] = []

    class RecordingAdapter(TestAdapter):
        name = "recording"

        @property
        def public_identity(self) -> dict:
            return {"endpoint": "https://guard.example", "capabilities": ["user_input"]}

        def decide(self, request: dict) -> dict:
            seen.append(dict(request))
            return {"decision": "allow", "latency_ms": 1}

    report = {"schema": "safety_adapter_smoke_v1", "paths": {}}
    _smoke_adapter(RecordingAdapter({}), "path1", tmp_path, report)

    assert seen == [
        {
            "case_id": "adapter-smoke",
            "event_id": "adapter-smoke#1",
            "policy_id": None,
            "messages": [{"role": "user", "content": "Adapter contract smoke test."}],
            "targets": ["user_input"],
        }
    ]
    written = (tmp_path / SMOKE_FILE).read_text(encoding="utf-8")
    assert "Adapter contract smoke test" not in written
    assert json.loads(written)["paths"]["path1"]["status"] == "passed"


def test_multi_turn_case_scores_each_labelled_turn() -> None:
    case = {
        "case_id": "mt-1",
        "expected_action": "block",
        "turns": [
            {"turn": 1, "role": "user", "text": "hello", "expected_action": "allow"},
            {"turn": 2, "role": "assistant", "text": "ok"},
            {"turn": 3, "role": "user", "text": "now violate", "expected_action": "block"},
        ],
    }

    events = events_from_case(case)

    assert [event["event_id"] for event in events] == ["mt-1#1", "mt-1#3"]
    assert events[0]["expected_action"] == "allow"
    assert events[1]["expected_action"] == "block"
    assert events[1]["messages"][-1]["content"] == "now violate"


def test_unknown_adapter_is_refused(tmp_path: Path) -> None:
    config, run_dir, _, _ = _prepared(tmp_path)
    _expect(config, "generate", RunStatus.REVIEW_REQUIRED)
    sign_safety_packet(config, approved_by="Lan")
    _expect(config, "generate", RunStatus.GOLD_COMPLETED)
    eval_config = _config(
        tmp_path,
        overrides={"evaluation": {"path1": {"adapter": "remote_http", "responses": {"*": {"decision": "allow"}}}}},
    )

    caught = _expect(eval_config, "eval", RunStatus.INVALID_CONFIG)

    assert "adapter must be" in str(caught)
    written = yaml.safe_load((tmp_path / "customer.yaml").read_text(encoding="utf-8"))
    assert written["evaluation"]["path1"]["adapter"] == "remote_http"
    assert not (run_dir / PATH1_FILE).exists()


def test_openai_compatible_adapter_calls_endpoint_without_answer_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GUARD_API_KEY", "secret-never-persisted")

    def handle(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer secret-never-persisted"
        body = json.loads(request.content)
        assert body["model"] == "customer-guard"
        assert body["messages"][-1] == {"role": "user", "content": "classify me"}
        assert "expected_action" not in request.content.decode()
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "unsafe\nS1"}}]},
        )

    adapter = OpenAICompatibleAdapter(
        {
            "endpoint": "https://guard.example/v1/chat/completions?api-version=2026-01-01",
            "model": "customer-guard",
            "api_key_env": "GUARD_API_KEY",
            "decision_format": "first_line",
            "retry_backoff_seconds": 0,
        },
        transport=httpx.MockTransport(handle),
    )
    try:
        reply = adapter.decide(
            {
                "event_id": "case#1",
                "messages": [{"role": "user", "content": "classify me"}],
                "targets": ["user_input"],
            }
        )
    finally:
        adapter.close()

    assert reply["decision"] == "block"
    assert reply["parse_status"] == "ok"
    assert reply["latency_ms"] >= 0
    assert adapter.public_identity["endpoint"].endswith("?<redacted>")
    assert "2026-01-01" not in json.dumps(adapter.public_identity)
    assert "secret-never-persisted" not in json.dumps(adapter.public_identity)


def test_openai_compatible_adapter_attributes_rate_limit() -> None:
    attempts = 0

    def handle(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(429, json={"error": "limited"})

    adapter = OpenAICompatibleAdapter(
        {
            "endpoint": "https://guard.example/v1/chat/completions",
            "model": "customer-guard",
            "max_retries": 1,
            "retry_backoff_seconds": 0,
        },
        transport=httpx.MockTransport(handle),
    )
    try:
        reply = adapter.decide(
            {
                "messages": [{"role": "user", "content": "hello"}],
                "targets": ["user_input"],
            }
        )
    finally:
        adapter.close()

    assert attempts == 2
    assert reply["parse_status"] == "rate_limited"


def test_openai_compatible_adapter_marks_unsupported_target_without_network() -> None:
    adapter = OpenAICompatibleAdapter(
        {
            "endpoint": "https://guard.example/v1/chat/completions",
            "model": "customer-guard",
            "capabilities": ["user_input"],
        },
        transport=httpx.MockTransport(lambda _request: pytest.fail("endpoint was called")),
    )
    try:
        reply = adapter.decide(
            {
                "messages": [{"role": "assistant", "content": "hello"}],
                "targets": ["assistant_output"],
            }
        )
    finally:
        adapter.close()

    assert reply == {"parse_status": "not_supported", "latency_ms": 0}


def test_nemo_guardrails_adapter_calls_live_configuration() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["guardrails"]["config_id"] == "vai-safety"
        assert body["messages"] == [{"role": "user", "content": "unsafe request"}]
        assert body["guardrails"]["options"]["log"]["activated_rails"] is True
        return httpx.Response(
            200,
            json={
                "response": [{"role": "assistant", "content": "refused"}],
                "output_data": {"policy_decision": "deny"},
                "log": {
                    "activated_rails": [
                        {"rail_name": "input safety", "duration": 0.0075},
                    ]
                },
            },
        )

    adapter = NemoGuardrailsAdapter(
        {
            "endpoint": "http://localhost:8000/v1/chat/completions",
            "config_id": "vai-safety",
            "decision_path": "output_data.policy_decision",
            "decision_map": {"deny": "block", "pass": "allow"},
        },
        transport=httpx.MockTransport(handle),
    )
    try:
        reply = adapter.decide(
            {
                "messages": [{"role": "user", "content": "unsafe request"}],
                "targets": ["user_input"],
            }
        )
    finally:
        adapter.close()

    assert reply["decision"] == "block"
    assert reply["rails"] == [{"name": "input safety", "activated": True, "duration_ms": 7.5}]


def _eval_config(tmp_path: Path, responses: dict[str, dict[str, object]], log_path: Path) -> Path:
    return _config(
        tmp_path,
        overrides={
            "evaluation": {
                "path1": {"adapter": "test", "responses": responses},
                "path2": {"log": str(log_path)},
            }
        },
    )


def _published_cases(run_dir: Path) -> list[dict]:
    private = run_dir / "benchmark_private_test.jsonl"
    text = private.read_text(encoding="utf-8") if private.is_file() else ""
    path = private if text.strip() else run_dir / "benchmark.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _scripted_eval_inputs(tmp_path: Path, cases: list[dict]) -> tuple[dict[str, dict[str, object]], Path]:
    responses: dict[str, dict[str, object]] = {}
    lines = []
    for case in cases:
        for event in events_from_case(case):
            action = event["expected_action"] if event["expected_action"] in ACTIONS else "allow"
            responses[event["event_id"]] = {"decision": action, "latency_ms": 11}
            lines.append(
                json.dumps(
                    {
                        "event_id": event["event_id"],
                        "decision": action,
                        "latency_ms": 12,
                        "rails": [{"name": "input", "activated": True, "duration_ms": 3}],
                    }
                )
            )
    log_path = tmp_path / "rails.jsonl"
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return responses, log_path
