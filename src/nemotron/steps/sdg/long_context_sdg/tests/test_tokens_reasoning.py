from long_context_sdg.reasoning import validate_reasoning
from long_context_sdg.schemas import ReasoningContent
from long_context_sdg.tokens import ContextMeter, count_tokens


def test_token_count_is_monotonic():
    assert count_tokens("short") < count_tokens("short " * 100)


def test_context_meter_spacing_and_reset():
    meter = ContextMeter(threshold=10, min_turns_between=3)
    meter.add_all([{"role": "user", "content": "text " * 100}])
    assert not meter.should_compress(2)
    assert meter.should_compress(3)
    meter.reset(3, "summary", [])
    assert not meter.should_compress(4)
    assert meter.history[0]["turn"] == 3


def test_reasoning_rejects_fabricated_citation_and_budget():
    fabricated = validate_reasoning(
        ReasoningContent(think="brief", cited_chunk_ids=["missing"]),
        ["known"],
        max_tokens=100,
    )
    assert not fabricated.ok and "not retrieved" in fabricated.errors[0]
    too_long = validate_reasoning(
        ReasoningContent(think="word " * 1000),
        [],
        max_tokens=10,
    )
    assert not too_long.ok
