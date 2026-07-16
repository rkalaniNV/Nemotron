import json

from retrieval_sdg.conversation.verifiers import ToolCallVerifier

TOOLS = [{"function": {
    "name": "search",
    "description": "search",
    "parameters": {"type": "object",
                   "properties": {"query": {"type": "string"}, "top_k": {"type": "integer"}},
                   "required": ["query"]}}}]


def _call(name, args):
    return {"id": "c0", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


def test_valid_call_passes():
    ok, err = ToolCallVerifier().verify_single(_call("search", {"query": "x", "top_k": 4}), TOOLS)
    assert ok and err is None


def test_missing_required_fails():
    ok, err = ToolCallVerifier().verify_single(_call("search", {"top_k": 4}), TOOLS)
    assert not ok and "query" in err


def test_wrong_type_fails():
    ok, err = ToolCallVerifier().verify_single(_call("search", {"query": "x", "top_k": "four"}), TOOLS)
    assert not ok and "top_k" in err


def test_unknown_tool_fails():
    ok, err = ToolCallVerifier().verify_single(_call("nope", {"query": "x"}), TOOLS)
    assert not ok and "not found" in err


def test_unknown_argument_fails():
    ok, err = ToolCallVerifier().verify_single(_call("search", {"query": "x", "bogus": 1}), TOOLS)
    assert not ok and "bogus" in err


def test_verify_batch_collects_errors():
    all_ok, err_msgs, *_ = ToolCallVerifier().verify(
        [_call("search", {"query": "x"}), _call("search", {})], TOOLS)
    assert not all_ok and len(err_msgs) == 1  # one invalid call reported
