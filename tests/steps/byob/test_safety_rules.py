"""Safety rule stages: extraction from evidence windows, normalization, severity, and conflicts."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from nemotron.steps.byob.runtime.benchmark_families.safety import extract
from nemotron.steps.byob.runtime.benchmark_families.safety.confidence import canonical_text
from nemotron.steps.byob.runtime.benchmark_families.safety.conflicts import detect_conflicts
from nemotron.steps.byob.runtime.benchmark_families.safety.ingest import rule_input_digests
from nemotron.steps.byob.runtime.benchmark_families.safety.ir import validate_rule
from nemotron.steps.byob.runtime.benchmark_families.safety.pipeline import generate_safety, prepare_safety
from nemotron.steps.byob.runtime.benchmark_families.safety.rules import condition_name, stable_policy_id
from nemotron.steps.byob.runtime.benchmark_families.safety.status import RunStatus, SafetyStatus
from nemotron.steps.byob.runtime.benchmark_families.safety.windows import build_windows
from tests.steps.byob.safety_fixtures import (
    ANTI_STATE,
    ANTI_STATE_ALLOW,
    DAN,
    DAN_ALLOW,
    FIVE_TOPIC,
    FIVE_TOPIC_REPLIES,
    PROBING,
    PROBING_ALLOW,
    ScriptedExtractor,
)
from tests.steps.byob.safety_fixtures import candidate as _candidate
from tests.steps.byob.safety_fixtures import leg as _leg


@pytest.fixture
def use(monkeypatch: pytest.MonkeyPatch):
    def install(extractor: Any) -> Any:
        monkeypatch.setattr(extract, "make_extractor", lambda config, run_dir: extractor)
        return extractor

    return install


# Five-topic policy


def test_five_topic_policy_yields_block_rules_with_paired_allow_rules(tmp_path: Path, use: Any) -> None:
    use(ScriptedExtractor(FIVE_TOPIC_REPLIES))
    run_dir = _generate(_config(tmp_path, {"five-topic.md": FIVE_TOPIC}))

    rows = _rows(run_dir)
    assert len(rows) == 10
    assert {row["status"] for row in rows.values()} == {"eligible_for_generation"}
    by_quote = {row["rule"]["provenance"]["source_quote"]: row["rule"] for row in rows.values()}
    anti_state, probing, dan = by_quote[ANTI_STATE], by_quote[PROBING], by_quote[DAN]
    assert anti_state["rule"]["block_if"] == {
        "all_of": ["instructional_or_organizing_verb", "anti_government_target"],
        "operator_source": "explicit",
    }
    assert probing["rule"]["block_if"] == {
        "all_of": ["probe_intent", "mentions_this_system"],
        "operator_source": "explicit",
    }
    assert dan["rule"]["block_if"] == {
        "any_of": ["bo_qua_moi_quy_dinh", "ignore_previous_instructions", "developer_mode", "dan"],
        "operator_source": "default_any_of",
    }
    for block, allow in ((ANTI_STATE, ANTI_STATE_ALLOW), (DAN, DAN_ALLOW), (PROBING, PROBING_ALLOW)):
        partner = by_quote[block]["policy_id"]
        assert by_quote[allow]["policy_id"] == f"{partner}-ALLOW"
        assert by_quote[allow]["paired_with"] == partner
        assert by_quote[allow]["rule"]["allow_if"]["operator_source"] == "single_condition"
    assert anti_state["severity"] == "medium"
    assert anti_state["severity_basis"] == "rubric:conditional_restriction"
    assert dan["severity_basis"] == "rubric:explicit_prohibition"
    assert by_quote[DAN_ALLOW]["severity"] == "low"
    eligible = yaml.safe_load((run_dir / "stage_cache" / "policy_rules_eligible.yaml").read_text(encoding="utf-8"))
    assert len(eligible["rules"]) == 10
    graph = json.loads((run_dir / "stage_cache" / "policy_graph.json").read_text(encoding="utf-8"))
    # Only a block rule and the allow rule of its own section are combined.
    assert all(
        len({policy_id.removesuffix("-ALLOW") for policy_id in node["matched_policy_ids"]}) == 1
        for node in graph["nodes"]
    )


def test_every_rule_quote_is_inside_its_cited_span(tmp_path: Path, use: Any) -> None:
    use(ScriptedExtractor(FIVE_TOPIC_REPLIES))
    run_dir = _generate(_config(tmp_path, {"five-topic.md": FIVE_TOPIC}))

    spans = _spans(run_dir)
    for row in _rows(run_dir).values():
        provenance = row["rule"]["provenance"]
        span = spans[provenance["source_span_id"]]
        assert canonical_text(provenance["source_quote"]) in canonical_text(span["text"])
        assert provenance["section"] == span["section"]


def test_rerun_keeps_rule_ids_and_reuses_cached_extraction(tmp_path: Path, use: Any) -> None:
    extractor = use(ScriptedExtractor(FIVE_TOPIC_REPLIES))
    config = _config(tmp_path, {"five-topic.md": FIVE_TOPIC})
    run_dir = _generate(config)
    first = set(_rows(run_dir))
    windows = extractor.windows

    _generate(config)

    assert set(_rows(run_dir)) == first
    assert extractor.windows == windows


def test_rule_ids_depend_on_document_section_quote_and_action() -> None:
    first = stable_policy_id("five-topic", "4", "Block card data.", "block")

    assert first == stable_policy_id("five-topic", "4", "  Block   card data. ", "block")
    assert first != stable_policy_id("five-topic", "5", "Block card data.", "block")
    assert first != stable_policy_id("five-topic", "4", "Block card data.", "escalate")
    assert first.startswith("FIVE-TOPIC-4-")
    assert condition_name("Bỏ qua mọi quy định!") == "bo_qua_moi_quy_dinh"
    assert condition_name("禁止").startswith("c_")
    assert condition_name("禁止") == condition_name(" 禁止 ")
    assert condition_name("???").startswith("c_")
    assert condition_name("   ") is None


def test_quote_without_an_operator_frame_stores_default_any_of(tmp_path: Path, use: Any) -> None:
    quote = ANTI_STATE.replace("đồng thời ", "")
    replies = {quote: [{**FIVE_TOPIC_REPLIES[ANTI_STATE][0], "source_quote": quote, "operator_hint": "all_of"}]}
    use(ScriptedExtractor(replies))
    run_dir = _generate(_config(tmp_path, {"policy.md": f"## anti_state_content\n\n{quote}\n"}))

    [row] = _rows(run_dir).values()
    assert row["rule"]["rule"]["block_if"]["operator_source"] == "default_any_of"
    assert "operator_hint_ignored" in {flag["code"] for flag in row["flags"]}


def test_written_files_hold_no_model_self_score(tmp_path: Path, use: Any) -> None:
    use(ScriptedExtractor(FIVE_TOPIC_REPLIES))
    run_dir = _generate(_config(tmp_path, {"five-topic.md": FIVE_TOPIC}))

    written = "\n".join(path.read_text(encoding="utf-8") for path in (run_dir / "stage_cache").rglob("*.json*"))
    assert "self_score" not in written
    assert "0.97" not in written
    assert "0.99" not in written


# Routing and severity


def test_uncertain_rule_waits_for_a_decision_and_stays_out_of_the_eligible_file(tmp_path: Path, use: Any) -> None:
    quote = "Có thể chặn tin nhắn quảng cáo lặp lại."
    candidate = _candidate(quote, block_if=[_leg("repeated_ads", "spam"), _leg("bulk_sender", "nhiều người")])
    candidate["testability"] = None
    use(ScriptedExtractor({quote: [candidate]}))
    run_dir = _generate(_config(tmp_path, {"policy.md": f"## Ads\n\n{quote}\n"}))

    [row] = _rows(run_dir).values()
    assert row["status"] == "needs_decision"
    assert 0.70 <= row["confidence"] < 0.90
    assert set(row["rule"]["unsupported_fields"]) == {"rule.block_if.repeated_ads", "rule.block_if.bulk_sender"}
    flagged = {item["policy_id"]: item for item in _jsonl(run_dir / "stage_cache" / "policy_flags.jsonl")}
    assert flagged[row["policy_id"]]["status"] == "needs_decision"
    eligible = yaml.safe_load((run_dir / "stage_cache" / "policy_rules_eligible.yaml").read_text(encoding="utf-8"))
    assert eligible["rules"] == []


FRAUD = "The assistant must escalate a transfer when fraud is suspected."
TRANSFER = "The assistant must block a transfer when explicit confirmation is missing."
TRANSFER_REPLIES = {
    FRAUD: [
        {
            **_candidate(FRAUD, "escalate", escalate_if=[_leg("suspected_fraud", "fraud is suspected")]),
            "object": "transfer",
            "risk_category": "fraud",
        }
    ],
    TRANSFER: [
        {
            **_candidate(
                TRANSFER, block_if=[_leg("explicit_confirmation_missing", "explicit confirmation is missing")]
            ),
            "object": "transfer",
            "risk_category": "irreversible_tool_action",
        }
    ],
}


def test_binding_rule_on_an_irreversible_tool_action_is_critical(tmp_path: Path, use: Any) -> None:
    use(ScriptedExtractor(TRANSFER_REPLIES))
    run_dir = _generate(_config(tmp_path, {"transfer.md": f"## 3. Transfers\n\n{FRAUD}\n{TRANSFER}\n"}))

    rules = {row["rule"]["provenance"]["source_quote"]: row["rule"] for row in _rows(run_dir).values()}
    assert rules[TRANSFER]["severity"] == "critical"
    assert rules[TRANSFER]["severity_basis"] == "rubric:binding_language+irreversible_tool_action"
    assert rules[FRAUD]["severity_basis"].startswith("rubric:")


def test_fixed_action_order_resolves_escalate_over_block(tmp_path: Path, use: Any) -> None:
    use(ScriptedExtractor(TRANSFER_REPLIES))
    run_dir = _generate(_config(tmp_path, {"transfer.md": f"## 3. Transfers\n\n{FRAUD}\n{TRANSFER}\n"}))

    ids = {row["rule"]["provenance"]["source_quote"]: policy_id for policy_id, row in _rows(run_dir).items()}
    graph = json.loads((run_dir / "stage_cache" / "policy_graph.json").read_text(encoding="utf-8"))
    [node] = [node for node in graph["nodes"] if len(node["case_context"]) == 2]
    assert node == {
        "case_context": ["explicit_confirmation_missing", "suspected_fraud"],
        "matched_policy_ids": sorted([ids[FRAUD], ids[TRANSFER]]),
        "actions": ["escalate", "block"],
        "expected_action": "escalate",
        "primary_policy_id": ids[FRAUD],
        "resolution": "action_order",
    }
    assert {row["status"] for row in _rows(run_dir).values()} == {"eligible_for_generation"}


# Conflicts and uncertainties


def test_unresolved_conflict_is_reported_and_quarantines_only_the_critical_rule(tmp_path: Path, use: Any) -> None:
    block = "The assistant must never reveal card data."
    allow = "Card data requests are allowed for the account owner."
    other = "Block weapon assembly instructions."
    replies = {
        block: [{**_candidate(block, block_if=[_leg("card_data_request", "card data")]), "risk_category": "fraud"}],
        allow: [_candidate(allow, "allow", allow_if=[_leg("card_data_request", "Card data requests")])],
        other: [_candidate(other, block_if=[_leg("weapon_assembly", "weapon assembly instructions")])],
    }
    use(ScriptedExtractor(replies))
    run_dir = _generate(
        _config(
            tmp_path, {"policy.md": f"## 1. Cards\n\n{block}\n\n## 2. Owners\n\n{allow}\n\n## 3. Weapons\n\n{other}\n"}
        )
    )

    status = {row["rule"]["provenance"]["source_quote"]: row["status"] for row in _rows(run_dir).values()}
    assert status == {block: "quarantined", allow: "needs_decision", other: "eligible_for_generation"}
    report = json.loads((run_dir / "stage_cache" / "conflict_report.json").read_text(encoding="utf-8"))
    [conflict] = report["conflicts"]
    assert conflict["type"] == "contradictory_rules"
    assert conflict["resolution"] == "unresolved"
    assert _record(run_dir)["rules"]["unresolved_conflicts"] == 1


def test_conflicting_definition_sends_the_rule_to_review(tmp_path: Path, use: Any) -> None:
    files = {
        "a.md": '## 1. Definitions\n\n"Card data" means a card number.\n\n'
        "## 2. Rules\n\nBlock requests for card data.\n",
        "b.md": '## 1. Definitions\n\n"Card data" means a card number or a bank account.\n',
    }
    quote = "Block requests for card data."
    use(
        ScriptedExtractor({quote: [_candidate(quote, block_if=[_leg("card_data_request", "requests for card data")])]})
    )
    run_dir = _generate(_config(tmp_path, files))

    [row] = _rows(run_dir).values()
    assert row["status"] == "needs_decision"
    assert any("definition_conflict" in reason for reason in row["reasons"])


def test_unsupported_candidates_are_listed_and_never_become_rules(tmp_path: Path, use: Any) -> None:
    first, second = "Block requests for card data.", "Escalate self-harm."
    replies = {
        first: [
            _candidate(second, "escalate", escalate_if=[_leg("self_harm", "self-harm")]),
            {"source_quote": first},
            _candidate(first, block_if=[]),
        ],
        second: [_candidate(second, "escalate", escalate_if=[_leg("self_harm", "self-harm")])],
    }
    use(ScriptedExtractor(replies))
    run_dir = _generate(_config(tmp_path, {"policy.md": f"## 4. Restricted\n\n{first} {second}\n"}))

    reasons = [row["reason"] for row in _jsonl(run_dir / "stage_cache" / "extraction_uncertainties.jsonl")]
    assert reasons == ["quote_in_neighbor", "malformed_candidate", "no_conditions"]
    assert _record(run_dir)["rules"]["extraction_uncertainties"] == 3


def test_invented_quote_is_quarantined_and_a_run_without_usable_rules_stops(tmp_path: Path, use: Any) -> None:
    quote = "Block requests for card data."
    invented = _candidate("Block every payment question.", block_if=[_leg("payment_question", "payment question")])
    extractor = use(ScriptedExtractor({quote: [invented]}))
    config = _config(tmp_path, {"policy.md": f"## 4. Cards\n\n{quote}\n"})
    prepare_safety(config)

    caught = _expect(config, RunStatus.NO_USABLE_RULES)

    assert caught.cli_exit_code == 1
    run_dir = tmp_path / "out" / "customer-safety-v1"
    [row] = _rows(run_dir).values()
    assert row["status"] == "quarantined"
    assert extractor.calls == 1


# Model and configuration


def test_model_failure_stops_with_model_unavailable_and_rerun_continues(tmp_path: Path, use: Any) -> None:
    config = _config(tmp_path, {"five-topic.md": FIVE_TOPIC})
    prepare_safety(config)
    use(ScriptedExtractor(FIVE_TOPIC_REPLIES, fail=True))

    caught = _expect(config, RunStatus.MODEL_UNAVAILABLE)

    assert "ConnectionError" in str(caught)
    use(ScriptedExtractor(FIVE_TOPIC_REPLIES))
    _expect(config, RunStatus.PIPELINE_INCOMPLETE)


@pytest.mark.parametrize(
    ("model_config", "message"),
    [(None, "generation_model_config is required"), ({"alias": "extractor"}, "missing provider, model")],
)
def test_policy_text_without_a_complete_model_config_is_refused(
    tmp_path: Path, model_config: dict | None, message: str
) -> None:
    config = _config(tmp_path, {"policy.md": "Block requests for card data.\n"}, model_config=model_config)
    prepare_safety(config)

    caught = _expect(config, RunStatus.INVALID_CONFIG)

    assert message in str(caught)


def test_pre_authored_rules_need_no_model(tmp_path: Path) -> None:
    rule = {
        "policy_id": "CARD-001",
        "rule": {"actor": "assistant", "action": "provide", "object": "card_number", "block_if": ["asks"]},
        "severity": "high",
        "provenance": {"source_quote": "Block requests for card numbers."},
    }
    config = _config(tmp_path, {"rules.yaml": yaml.safe_dump(rule)})
    run_dir = _generate(config)

    [row] = _rows(run_dir).values()
    assert row["origin"] == "pre_authored"
    assert row["status"] == "eligible_for_generation"
    assert row["rule"]["severity_basis"] == "source:pre_authored"
    assert "testability_from_author" in {flag["code"] for flag in row["flags"]}
    assert _jsonl(run_dir / "stage_cache" / "candidate_policy_rules.jsonl") == []


def test_a_matching_operator_hint_is_not_flagged(tmp_path: Path, use: Any) -> None:
    quote = "Block requests for card data."
    candidate = {**_candidate(quote, block_if=[_leg("card_data", "card data")]), "operator_hint": "any_of"}
    use(ScriptedExtractor({quote: [candidate]}))
    run_dir = _generate(_config(tmp_path, {"policy.md": f"## Cards\n\n{quote}\n"}))

    [row] = _rows(run_dir).values()
    assert row["rule"]["rule"]["block_if"]["operator_source"] == "default_any_of"
    assert "operator_hint_ignored" not in {flag["code"] for flag in row["flags"]}


def test_one_bad_window_does_not_drop_the_rules_around_it(tmp_path: Path, use: Any) -> None:
    good = "Block requests for card data."
    bad = "Escalate self-harm."

    class _Mixed:
        identity = {"provider": "fake", "model": "mixed"}

        def extract(self, requests: list[Any]) -> dict[str, Any]:
            replies = {}
            for request in requests:
                clause = json.loads(request.window)["clause"]["text"]
                if clause == good:
                    replies[request.request_id] = {
                        "candidates": [_candidate(good, block_if=[_leg("card_data", "card data")])]
                    }
                else:
                    replies[request.request_id] = "not a reply"
            return replies

    use(_Mixed())
    run_dir = _generate(_config(tmp_path, {"policy.md": f"## 4. Restricted\n\n{good}\n\n{bad}\n"}))

    assert len(_rows(run_dir)) == 1
    reasons = [row["reason"] for row in _jsonl(run_dir / "stage_cache" / "extraction_uncertainties.jsonl")]
    assert reasons == ["malformed_response"]


def test_a_failed_second_pass_sends_rules_to_review(tmp_path: Path, use: Any) -> None:
    quote = "Block requests for card data."

    class _Flaky:
        identity = {"provider": "fake", "model": "flaky"}

        def __init__(self) -> None:
            self.calls = 0

        def extract(self, requests: list[Any]) -> dict[str, Any]:
            self.calls += 1
            if self.calls > 1:
                raise ConnectionError("timed out")
            return {
                request.request_id: {"candidates": [_candidate(quote, block_if=[_leg("card_data", "card data")])]}
                for request in requests
            }

    extractor = use(_Flaky())
    run_dir = _generate(_config(tmp_path, {"policy.md": f"## Cards\n\n{quote}\n"}, profile="regulated"))

    [row] = _rows(run_dir).values()
    assert extractor.calls == 2
    assert row["status"] == "needs_decision"
    assert row["confidence"] == 0.8
    assert any(
        item["reason"] == "second_pass_failed"
        for item in _jsonl(run_dir / "stage_cache" / "extraction_uncertainties.jsonl")
    )
    assert _record(run_dir)["last_outcome"]["status"] == "pipeline_incomplete"


def test_same_quote_with_different_conditions_is_not_silently_kept(tmp_path: Path, use: Any) -> None:
    quote = "Block requests for card data."
    first = _candidate(quote, block_if=[_leg("card_data", "card data")])
    second = _candidate(quote, block_if=[_leg("payment_question", "card data")])
    use(ScriptedExtractor({quote: [first, second]}))
    run_dir = _generate(_config(tmp_path, {"policy.md": f"## Cards\n\n{quote}\n"}))

    [row] = _rows(run_dir).values()
    assert row["status"] == "needs_decision"
    assert any("different rule" in reason for reason in row["reasons"])


def test_a_condition_without_ascii_is_kept_for_review(tmp_path: Path, use: Any) -> None:
    quote = "Block requests for card data."
    use(ScriptedExtractor({quote: [_candidate(quote, block_if=[_leg("禁止", "card data")])]}))
    run_dir = _generate(_config(tmp_path, {"policy.md": f"## Cards\n\n{quote}\n"}))

    [row] = _rows(run_dir).values()
    assert row["status"] == "needs_decision"
    assert row["rule"]["rule"]["block_if"]["any_of"] == [condition_name("禁止")]
    assert "condition_name_hashed" in {flag["code"] for flag in row["flags"]}


def test_rules_about_the_same_tool_conflict_across_sections(tmp_path: Path, use: Any) -> None:
    use(ScriptedExtractor(TRANSFER_REPLIES))
    text = f"## 2. Fraud\n\n{FRAUD}\n\n## 3. Confirmation\n\n{TRANSFER}\n"
    run_dir = _generate(_config(tmp_path, {"transfer.md": text}))

    graph = json.loads((run_dir / "stage_cache" / "policy_graph.json").read_text(encoding="utf-8"))
    [node] = [
        node
        for node in graph["nodes"]
        if set(node["case_context"]) == {"suspected_fraud", "explicit_confirmation_missing"}
    ]
    assert node["expected_action"] == "escalate"
    assert node["resolution"] == "action_order"
    assert len(node["matched_policy_ids"]) == 2


def test_a_rule_that_allows_and_blocks_the_same_situation_is_unresolved() -> None:
    rule = validate_rule(
        {
            "policy_id": "CARD-1",
            "rule": {
                "actor": "assistant",
                "action": "provide",
                "object": "card_data",
                "allow_if": {"any_of": ["card_data_request"]},
                "block_if": {"any_of": ["card_data_request"]},
            },
            "severity": "high",
            "provenance": {"document_id": "policy", "section": "1", "source_quote": "See card data."},
        }
    )

    result = detect_conflicts([rule])

    assert result.conflicts[0]["type"] == "contradictory_rules"
    assert result.conflicts[0]["policy_ids"] == ["CARD-1"]
    assert result.graph[0]["resolution"] == "unresolved"


def test_disjoint_targets_are_not_treated_as_one_situation() -> None:
    def rule(policy_id: str, field: str, leg: str, targets: list[str]) -> Any:
        return validate_rule(
            {
                "policy_id": policy_id,
                "rule": {
                    "actor": "assistant",
                    "action": "respond",
                    "object": "transfer",
                    field: {"any_of": [leg]},
                },
                "severity": "high",
                "scope": {"targets": targets},
                "decision": {"violation_action": "block", "critical_action": "escalate"},
                "provenance": {"document_id": "policy", "section": "3", "source_quote": f"Handle {leg}."},
            }
        )

    separated = detect_conflicts(
        [
            rule("FRAUD-1", "escalate_if", "suspected_fraud", ["user_input"]),
            rule("TRANSFER-1", "block_if", "explicit_confirmation_missing", ["tool_call"]),
        ]
    )
    shared = detect_conflicts(
        [
            rule("FRAUD-1", "escalate_if", "suspected_fraud", ["tool_call"]),
            rule("TRANSFER-1", "block_if", "explicit_confirmation_missing", ["tool_call"]),
        ]
    )

    assert separated.graph == []
    assert shared.graph[0]["expected_action"] == "escalate"


def test_a_damaged_span_index_stops_before_the_model_is_called(tmp_path: Path, use: Any) -> None:
    extractor = use(ScriptedExtractor(FIVE_TOPIC_REPLIES))
    config = _config(tmp_path, {"policy.md": "Block requests for card data.\n"})
    run_dir = prepare_safety(config)
    index_path = run_dir / "stage_cache" / "source_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["spans"].append(dict(index["spans"][0]))
    index_path.write_text(json.dumps(index), encoding="utf-8")
    _accept_ingest_outputs(run_dir)

    caught = _expect(config, RunStatus.INVALID_RUN_ARTIFACT)

    assert "duplicate span id" in str(caught)
    assert extractor.calls == 0
    assert _record(run_dir)["last_outcome"]["status"] == "invalid_run_artifact"


@pytest.mark.parametrize("name", ["source_index.json", "glossary.json", "preauthored_rules.jsonl"])
def test_an_ingest_output_edited_after_prepare_is_refused_before_the_model_is_called(
    tmp_path: Path, use: Any, name: str
) -> None:
    extractor = use(ScriptedExtractor(FIVE_TOPIC_REPLIES))
    config = _config(tmp_path, {"five-topic.md": FIVE_TOPIC})
    run_dir = prepare_safety(config)
    path = run_dir / "stage_cache" / name
    path.write_bytes(path.read_bytes() + b"\n")

    caught = _expect(config, RunStatus.INVALID_RUN_ARTIFACT)

    assert f"stage_cache/{name} changed after prepare" in str(caught)
    assert extractor.calls == 0
    prepare_safety(config)
    _expect(config, RunStatus.PIPELINE_INCOMPLETE)


def test_a_run_record_without_ingest_hashes_asks_for_prepare(tmp_path: Path, use: Any) -> None:
    extractor = use(ScriptedExtractor(FIVE_TOPIC_REPLIES))
    config = _config(tmp_path, {"five-topic.md": FIVE_TOPIC})
    run_dir = prepare_safety(config)
    record = _record(run_dir)
    del record["ingest_artifacts"]
    (run_dir / "run_record.json").write_text(json.dumps(record), encoding="utf-8")

    _expect(config, RunStatus.PREPARE_REQUIRED)

    assert extractor.calls == 0


def test_a_failed_generate_marks_the_rule_stages_failed(tmp_path: Path, use: Any) -> None:
    use(ScriptedExtractor(FIVE_TOPIC_REPLIES))
    config = _config(tmp_path, {"five-topic.md": FIVE_TOPIC})
    run_dir = _generate(config)
    failing = ScriptedExtractor(FIVE_TOPIC_REPLIES, fail=True)
    failing.identity = {"provider": "fake", "model": "other"}
    use(failing)

    _expect(config, RunStatus.MODEL_UNAVAILABLE)

    stages = _cache_index(run_dir)["stages"]
    for stage in ("extract", "normalize", "conflicts"):
        assert stages[stage]["status"] == "failed"
        assert stages[stage]["outcome"] == "model_unavailable"
    assert not (run_dir / "stage_cache" / "policy_rules_eligible.yaml").exists()
    use(ScriptedExtractor(FIVE_TOPIC_REPLIES))
    _expect(config, RunStatus.PIPELINE_INCOMPLETE)
    assert {_cache_index(run_dir)["stages"][stage]["status"] for stage in ("extract", "normalize", "conflicts")} == {
        "completed"
    }


def test_a_reply_with_a_malformed_candidate_is_used_but_asked_again(tmp_path: Path, use: Any) -> None:
    quote = "Block requests for card data."
    good = _candidate(quote, block_if=[_leg("card_data_request", "card data")])
    broken = {"source_quote": quote}
    first = use(ScriptedExtractor({quote: [good, broken]}))
    config = _config(tmp_path, {"policy.md": f"## Cards\n\n{quote}\n"})
    run_dir = _generate(config)

    assert len(_rows(run_dir)) == 1
    assert "malformed_candidate" in {
        row["reason"] for row in _jsonl(run_dir / "stage_cache" / "extraction_uncertainties.jsonl")
    }
    log = json.loads((run_dir / "stage_cache" / "policy_routing_log.json").read_text(encoding="utf-8"))
    assert log["extraction"]["passes"][0]["not_cached"] == 1
    assert first.calls == 1
    fixed = use(ScriptedExtractor({quote: [good]}))

    _generate(config)

    assert fixed.calls == 1
    fixed_again = use(ScriptedExtractor({quote: [good]}))
    _generate(config)
    assert fixed_again.calls == 0


class _FakeDesigner:
    """Stands in for the Data Designer engine. The config builder and seed file are the real ones."""

    created: list["_FakeDesigner"] = []

    def __init__(self, artifact_path: str, model_providers: list[Any] | None = None) -> None:
        self.artifact_path = Path(artifact_path)
        self.model_providers = model_providers or []
        self.seed_rows: list[dict[str, Any]] = []
        _FakeDesigner.created.append(self)

    def validate(self, builder: Any) -> None:
        self.columns = {column.name: column for column in builder.get_column_configs()}

    def create(self, config_builder: Any, num_records: int) -> Any:
        import pandas as pd

        seed = Path(config_builder.get_seed_config().source.path)
        self.seed_rows = pd.read_csv(seed).to_dict("records")
        assert num_records == len(self.seed_rows)
        rows = [
            {"request_id": row["request_id"], "response": json.dumps({"candidates": []})} for row in self.seed_rows
        ]
        rows.append({"request_id": "repeated", "response": "{}"})
        rows.append({"request_id": "repeated", "response": "{}"})
        self.artifact_path.mkdir(parents=True, exist_ok=True)
        path = self.artifact_path / "final.parquet"
        pd.DataFrame(rows).to_parquet(path)
        return SimpleNamespace(artifact_storage=SimpleNamespace(final_dataset_path=path))


def test_data_designer_extractor_sends_redacted_windows_and_names_the_key_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import data_designer.interface

    monkeypatch.setattr(data_designer.interface, "DataDesigner", _FakeDesigner)
    monkeypatch.setenv("SAFETY_TEST_KEY", "sk-test-value-never-written")
    model = {
        "alias": "extractor",
        "provider": "nvidia",
        "model": "nvidia/openai/gpt-oss-120b",
        "base_url": "https://provider.example.com/v1",
        "api_key_env": "SAFETY_TEST_KEY",
        "inference_parameters": {"temperature": 0, "max_tokens": 4000},
    }
    extractor = extract.make_extractor(SimpleNamespace(extras={"generation_model_config": model}), tmp_path)
    requests = [
        extract.ExtractionRequest("doc#1#span-01", json.dumps({"clause": {"text": "Block REDACTED_EMAIL_1."}})),
        extract.ExtractionRequest("repeated", json.dumps({"clause": {"text": "Block it."}})),
    ]

    replies = extractor.extract(requests)

    designer = _FakeDesigner.created[-1]
    assert [row["request_id"] for row in designer.seed_rows] == ["doc#1#span-01", "repeated"]
    assert designer.seed_rows[0]["window"] == requests[0].window
    [provider] = designer.model_providers
    assert (provider.name, provider.endpoint, provider.api_key) == (
        "nvidia",
        "https://provider.example.com/v1",
        "SAFETY_TEST_KEY",
    )
    assert designer.columns["response"].model_alias == "extractor"
    assert replies == {"doc#1#span-01": json.dumps({"candidates": []})}
    assert "base_url" in extractor.identity
    assert "api_key_env" not in extractor.identity
    assert list((tmp_path / "artifacts" / "model_seeds").iterdir()) == []
    written = b"".join(path.read_bytes() for path in tmp_path.rglob("*") if path.is_file())
    assert b"sk-test-value-never-written" not in written


def test_duplicate_span_ids_are_rejected() -> None:
    span = {
        "source_span_id": "doc#1#span-01",
        "document_id": "doc",
        "section": "1",
        "heading_path": ["Policy"],
        "text": "Block it.",
    }
    with pytest.raises(ValueError, match="duplicate span id"):
        build_windows([{"document_id": "doc", "sha256": "a" * 64}], [span, dict(span)], [])


def test_regulated_profile_runs_a_second_independent_pass(tmp_path: Path, use: Any) -> None:
    extractor = use(ScriptedExtractor(FIVE_TOPIC_REPLIES))
    run_dir = _generate(_config(tmp_path, {"five-topic.md": FIVE_TOPIC}, profile="regulated"))

    assert extractor.calls == 2
    candidates = _jsonl(run_dir / "stage_cache" / "candidate_policy_rules.jsonl")
    assert {row["passes"] for row in candidates} == {2}
    assert {row["signals"]["pass_agreement"] for row in candidates} == {1.0}
    log = json.loads((run_dir / "stage_cache" / "policy_routing_log.json").read_text(encoding="utf-8"))
    assert [item["pass"] for item in log["extraction"]["passes"]] == [1, 2]
    assert log["extraction"]["model"] == {"provider": "fake", "model": "scripted"}


# Helpers


def _config(
    tmp_path: Path,
    files: dict[str, str],
    *,
    profile: str | None = None,
    model_config: dict | None = None,
) -> Path:
    policies = tmp_path / "policies"
    policies.mkdir(exist_ok=True)
    for relative, text in files.items():
        (policies / relative).write_text(text, encoding="utf-8")
    data: dict[str, Any] = {
        "family": "safety",
        "expt_name": "customer-safety-v1",
        "config_status": "resolved",
        "mode": "assisted",
        "input_dir": str(policies),
        "output_dir": str(tmp_path / "out"),
    }
    if profile:
        data["assurance"] = {"profile": profile}
    if model_config:
        data["generation_model_config"] = model_config
    config = tmp_path / "customer.yaml"
    config.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return config


def _generate(config: Path) -> Path:
    prepare_safety(config)
    _expect(config, RunStatus.PIPELINE_INCOMPLETE)
    return config.parent / "out" / "customer-safety-v1"


def _expect(config: Path, status: RunStatus) -> SafetyStatus:
    with pytest.raises(SafetyStatus) as caught:
        generate_safety(config)
    assert caught.value.status is status, str(caught.value)
    return caught.value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _rows(run_dir: Path) -> dict[str, dict[str, Any]]:
    return {row["policy_id"]: row for row in _jsonl(run_dir / "stage_cache" / "policy_rules.jsonl")}


def _spans(run_dir: Path) -> dict[str, dict[str, Any]]:
    index = json.loads((run_dir / "stage_cache" / "source_index.json").read_text(encoding="utf-8"))
    return {span["source_span_id"]: span for span in index["spans"]}


def _record(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "run_record.json").read_text(encoding="utf-8"))


def _cache_index(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "stage_cache" / "cache_index.json").read_text(encoding="utf-8"))


def _accept_ingest_outputs(run_dir: Path) -> None:
    """Record the current ingest outputs as committed, to reach the checks behind the hash check."""
    record = _record(run_dir)
    record["ingest_artifacts"] = rule_input_digests(run_dir / "stage_cache")
    (run_dir / "run_record.json").write_text(json.dumps(record), encoding="utf-8")
