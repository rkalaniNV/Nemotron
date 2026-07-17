"""Token accounting + the 32k-style context-compression trigger."""

from __future__ import annotations

from mtsdg.schemas import Message
from mtsdg.tokens import ContextMeter, count_tokens, context_tokens, message_tokens


def test_count_tokens_monotone():
    assert count_tokens("") == 0
    assert count_tokens("a b c") >= 1
    assert count_tokens("a" * 1000) > count_tokens("a" * 10)


def test_message_tokens_counts_content_and_tool_calls():
    m = Message(role="assistant", content="hello world",
                tool_calls=[{"id": "1", "type": "function",
                             "function": {"name": "retrieve", "arguments": '{"query":"x"}'}}])
    assert message_tokens(m) > count_tokens("hello world")


def test_meter_triggers_when_over_threshold():
    meter = ContextMeter(threshold=50, min_turns_between=1)
    big = Message(role="tool", name="retrieve", content="word " * 200)
    meter.add(big)
    assert meter.active_tokens >= 50
    assert meter.should_compress(current_turn=5) is True


def test_meter_respects_min_turns_between():
    meter = ContextMeter(threshold=10, min_turns_between=3)
    meter.add(Message(role="user", content="word " * 50))
    meter._last_compression_turn = 4
    assert meter.should_compress(current_turn=5) is False   # only 1 turn since last
    assert meter.should_compress(current_turn=7) is True     # 3 turns since last


def test_meter_reset_collapses_to_summary():
    meter = ContextMeter(threshold=100, min_turns_between=1)
    meter.add(Message(role="tool", name="retrieve", content="word " * 500))
    before = meter.active_tokens
    meter.reset_after_compression(current_turn=6, summary_text="short summary")
    assert meter.active_tokens < before
    assert len(meter.history) == 1
    assert meter.history[0]["turn"] == 6


def test_context_tokens_sums_window():
    msgs = [Message(role="user", content="hi"), Message(role="assistant", content="hello there friend")]
    assert context_tokens(msgs) == sum(message_tokens(m) for m in msgs)
