import json

import httpx
import pytest
from long_context_sdg.config import ToolConfig
from long_context_sdg.executors.base import (
    ConversationState,
    ExecutionServices,
    ToolExecutionError,
)
from long_context_sdg.retrieval import RetrieverClient
from long_context_sdg.schemas import ToolCall, ToolResult
from long_context_sdg.tool_registry import ToolRegistry, normalize_tool_call

from tests.fixtures import FakeRetriever, make_config


def test_retriever_maps_fields_and_hashes_missing_id(tmp_path):
    cfg = make_config(tmp_path)

    def handler(request):
        assert request.method == "POST"
        assert request.read()
        return httpx.Response(200, json={"chunks": [{"text": "passage", "source": "doc"}]})

    client = RetrieverClient(cfg.retriever, transport=httpx.MockTransport(handler))
    try:
        result = client.query("query")
        assert result[0].chunk_id.startswith("h-")
        assert result[0].content == "passage" and result[0].source == "doc"
    finally:
        client.close()


def test_custom_executor_import_path(tmp_path):
    tool = ToolConfig.model_validate(
        {
            "schema": {
                "type": "function",
                "function": {
                    "name": "custom",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            "executor": "tests.fixtures:CustomExecutor",
            "executor_kwargs": {"prefix": "loaded"},
        }
    )
    registry = ToolRegistry([tool], ExecutionServices())
    result = registry.execute(
        ToolCall(id="c1", name="custom", arguments={}),
        ConversationState(conversation_id="x"),
        "instructions",
    )
    assert result.payload == {"value": "loaded"}


def test_registry_rejects_arguments_before_execution(tmp_path):
    cfg = make_config(tmp_path)
    registry = ToolRegistry(
        cfg.tools,
        ExecutionServices(retriever=FakeRetriever()),
    )
    with pytest.raises(ToolExecutionError, match="fail schema"):
        registry.execute(
            ToolCall(id="x", name="retrieve", arguments={}),
            ConversationState(conversation_id="x"),
            "",
        )


def test_normalize_tolerates_model_parameters_as_argument_values():
    call = normalize_tool_call(
        {
            "type": "function",
            "function": {
                "name": "retrieve",
                "description": "lookup",
                "parameters": {"query": "notification settings", "top_k": 5},
            },
        },
        "fallback",
    )
    assert call.name == "retrieve"
    assert call.arguments == {"query": "notification settings", "top_k": 5}


def test_memory_allowlist_and_read_after_write(tmp_path):
    cfg = make_config(tmp_path)
    registry = ToolRegistry(cfg.tools, ExecutionServices(retriever=FakeRetriever()))
    state = ConversationState(conversation_id="x")
    registry.execute(
        ToolCall(
            id="w",
            name="memory_write",
            arguments={
                "key": "verbosity",
                "value": "concise",
                "scope": "user",
                "reason": "requested",
            },
        ),
        state,
        "",
    )
    read = registry.execute(
        ToolCall(
            id="r",
            name="memory_read",
            arguments={"scope": "user"},
        ),
        state,
        "",
    )
    assert read.payload["verbosity"] == "concise"
    with pytest.raises(ToolExecutionError, match="not allowed"):
        registry.execute(
            ToolCall(
                id="bad",
                name="memory_write",
                arguments={
                    "key": "secret",
                    "value": "x",
                    "scope": "user",
                    "reason": "bad",
                },
            ),
            state,
            "",
        )


def test_simulated_tool_result_is_explicitly_marked():
    result = ToolResult(
        tool_call_id="sim-1",
        name="weather",
        payload={"temperature": 21},
        simulated=True,
    )
    content = json.loads(result.to_message()["content"])
    assert content == {
        "_sdg_simulated": True,
        "value": {"temperature": 21},
    }
