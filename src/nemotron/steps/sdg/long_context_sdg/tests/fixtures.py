from __future__ import annotations

import json
import re
from types import SimpleNamespace
from typing import Any

from long_context_sdg.config import PipelineConfig
from long_context_sdg.executors.base import (
    ConversationState,
    ExecutionContext,
    ExecutionServices,
)
from long_context_sdg.schemas import RetrievalChunk, ToolCall, ToolResult


def make_config(tmp_path, *, judge=False) -> PipelineConfig:
    return PipelineConfig.model_validate(
        {
            "paths": {
                "seeds": str(tmp_path / "queries.jsonl"),
                "enriched_seeds": str(tmp_path / "enriched.jsonl"),
                "artifacts": str(tmp_path / "artifacts"),
                "generated": str(tmp_path / "generated.jsonl"),
                "canonical": str(tmp_path / "canonical.jsonl"),
                "output_dir": str(tmp_path / "evaluated"),
                "export": str(tmp_path / "sft.jsonl"),
            },
            "run": {"mode": "preview", "seed": 11, "num_records": 0},
            "instructions": "Respond in the requested language and use only the configured corpus.",
            "providers": [],
            "models": [
                {"alias": alias, "model": "fake", "provider": "nvidia"}
                for alias in ("assistant", "user", "compressor", "judge")
            ],
            "episode": {
                "turn_budget": {"min": 15, "max": 22},
                "max_retrieval_calls": 3,
                "max_retrieval_calls_per_turn": 1,
                "retrieval_novelty": {
                    "query_lexical_similarity_threshold": 0.80,
                    "evidence_lexical_similarity_threshold": 0.85,
                    "min_new_chunk_fraction": 0.50,
                    "max_low_gain_chain": 1,
                    "low_gain_followup_similarity_threshold": 0.35,
                },
                "max_steps_per_turn": 6,
                "max_tool_calls_per_turn": 2,
                "max_tool_calls_per_conversation": 12,
            },
            "context": {
                "compression_threshold": 300,
                "model_token_limit": 8000,
                "recent_raw_turns": 3,
                "min_turns_between_compression": 3,
                "compression_token_budget": 300,
                "max_reasoning_tokens": 400,
            },
            "retriever": {
                "endpoint": "http://retriever/query",
                "method": "POST",
                "query_field": "query",
                "top_k_field": "num_chunks",
                "top_k": 3,
                "results_path": "chunks",
                "fields": {"id": "id", "text": "text", "source": "source"},
                "selection": "ranked",
                "timeout_seconds": 5,
                "retries": 1,
                "backoff_seconds": 0,
            },
            "tools": [
                {
                    "schema": {
                        "type": "function",
                        "function": {
                            "name": "retrieve",
                            "description": "Retrieve evidence.",
                            "parameters": {
                                "type": "object",
                                "properties": {"query": {"type": "string"}},
                                "required": ["query"],
                            },
                        },
                    },
                    "executor": "long_context_sdg.executors.retrieval:RetrievalExecutor",
                },
                {
                    "schema": {
                        "type": "function",
                        "function": {
                            "name": "memory_read",
                            "description": "Read memory.",
                            "parameters": {
                                "type": "object",
                                "properties": {"scope": {"type": "string"}},
                                "required": ["scope"],
                            },
                        },
                    },
                    "executor": "long_context_sdg.executors.memory:MemoryExecutor",
                },
                {
                    "schema": {
                        "type": "function",
                        "function": {
                            "name": "memory_write",
                            "description": "Write memory.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "key": {"type": "string"},
                                    "value": {},
                                    "scope": {"type": "string"},
                                    "reason": {"type": "string"},
                                },
                                "required": ["key", "value", "scope", "reason"],
                            },
                        },
                    },
                    "executor": "long_context_sdg.executors.memory:MemoryExecutor",
                },
            ],
            "validation": {"require_final_answer_each_turn": True},
            "judge": {
                "enabled": judge,
                "min_score": 3,
                "dimensions": [
                    "grounding",
                    "long_range_coherence",
                    "tool_use",
                    "retrieval_rewrite",
                    "user_realism",
                    "helpfulness",
                    "instruction_adherence",
                    "compaction_continuity",
                ],
            },
            "export": {"format": "messages_and_tools"},
        }
    )


class FakeRetriever:
    def query(self, text: str, *, top_k=None):
        suffix = re.sub(r"\W+", "-", text.lower()).strip("-")[-24:]
        return [
            RetrievalChunk(
                chunk_id=f"chunk-{suffix}",
                content=(f"Authoritative evidence facet {suffix}. " + f"{suffix} " * 30),
                title="Domain document",
                source="domain-corpus",
                score=1.0,
            )
        ]


def _text(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("content", ""))
    return str(getattr(message, "content", ""))


class FakeAssistant:
    def completion(self, messages, **kwargs):
        directive = _text(messages[-1])
        turn_match = re.search(r"Turn (\d+)", directive)
        turn = int(turn_match.group(1)) if turn_match else 1
        value = {
            "reasoning": {
                "think": "I can respond to this conversational turn.",
                "task_understanding": "answer",
                "answer_plan": ["answer"],
            },
            "content": f"Natural final answer for turn {turn}.",
            "tool_calls": [],
        }
        return SimpleNamespace(message=SimpleNamespace(content=json.dumps(value), role="assistant"))


class FakeUser:
    def completion(self, messages, **kwargs):
        return SimpleNamespace(
            message=SimpleNamespace(
                content=json.dumps({"content": "Could you explain how that applies in another realistic case?"}),
                role="assistant",
            )
        )


class FakeCompressor:
    def completion(self, messages, **kwargs):
        value = {
            "summary_id": "placeholder",
            "covers_turns": [1, 1],
            "user_facts": [],
            "key_facts": [],
            "constraints": [],
            "open_questions": [],
            "source_message_ids": [],
            "no_new_claims": True,
        }
        return SimpleNamespace(message=SimpleNamespace(content=json.dumps(value), role="assistant"))


class FakeJudge:
    def completion(self, messages, **kwargs):
        dimensions = [
            "grounding",
            "long_range_coherence",
            "tool_use",
            "retrieval_rewrite",
            "user_realism",
            "helpfulness",
            "instruction_adherence",
            "compaction_continuity",
        ]
        value = {
            "scores": {d: 5 for d in dimensions},
            "rating": "success",
            "explanation": "valid",
        }
        return SimpleNamespace(message=SimpleNamespace(content=json.dumps(value), role="assistant"))


def fake_models(*, judge=True):
    return {
        "assistant": FakeAssistant(),
        "user": FakeUser(),
        "compressor": FakeCompressor(),
        "judge": FakeJudge() if judge else FailingJudge(),
    }


class FailingJudge:
    def completion(self, messages, **kwargs):
        raise TimeoutError("judge timeout")


class CustomExecutor:
    def __init__(self, *, services: ExecutionServices, prefix="custom"):
        self.prefix = prefix

    def execute(self, call: ToolCall, state: ConversationState, context: ExecutionContext) -> ToolResult:
        return ToolResult(tool_call_id=call.id, name=call.name, payload={"value": self.prefix})
