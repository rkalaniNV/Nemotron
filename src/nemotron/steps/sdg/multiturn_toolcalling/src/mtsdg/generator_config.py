"""Config for the live episode-simulator column generator."""

from __future__ import annotations

from typing import List, Literal

from data_designer.config.base import SingleColumnConfig


class EpisodeSimulatorConfig(SingleColumnConfig):
    """Column configuration for the live multi-turn long-context episode simulator.

    Reads one ``queries.jsonl`` row (as JSON in ``episode_input_column``) and
    generates a 20-25 turn trajectory: the assistant LLM drives the live
    retrieve -> assess -> rewrite -> answer loop against the real retriever, the
    user LLM drives natural follow-ups, memory is read/written on request, and the
    context is automatically compacted at the token threshold (never a tool call).
    Emits ``structured_messages``.
    """

    column_type: Literal["episode-simulator"] = "episode-simulator"

    # This column's own model alias (base requires one).
    model_alias: str = "assistant"

    # Input column: one QuerySeed row serialized as JSON.
    episode_input_column: str = "episode_input"

    # Live retriever.
    retriever_url: str = "http://localhost:8000"
    retrieve_top_k: int = 3

    # Model roles.
    user_alias: str = "user"
    assistant_alias: str = "assistant"
    judge_alias: str = "judge"
    compressor_alias: str = "compressor"

    # Behaviour.
    majority_vote_n: int = 1
    max_steps: int = 4
    # Dynamic episode length: when both are set (min <= max), each episode uses a
    # turn count picked in [turn_budget_min, turn_budget_max] — deterministically
    # per query_id (reproducible, but varied across queries) — instead of the
    # query's own turn_budget. 0/0 disables (use each query's turn_budget).
    turn_budget_min: int = 0
    turn_budget_max: int = 0
    # Force the query-rewrite loop (broad retrieve -> refine -> retrieve again, in
    # the SAME turn) on the first N retrieving turns of an episode, so the corpus
    # actually teaches query rewrite even when the model would one-shot the query.
    # 0 disables the deterministic force (rely on prompting only).
    force_rewrite_count: int = 2
    compression_token_budget: int = 400
    max_reasoning_tokens: int = 400
    recent_raw_turns: int = 4
    # Token threshold that triggers automatic compaction (32000 in production;
    # a smaller value simulates it on shorter synthetic turns).
    context_token_threshold: int = 32000
    min_turns_between_compression: int = 3
    run_inline_judge: bool = False
    run_trajectory_judge: bool = True
    # Incremental checkpoint: each finished episode is appended here immediately,
    # so partial results survive a crash/kill and you can inspect mid-run. Empty
    # disables it.
    checkpoint_path: str = ""

    @property
    def required_columns(self) -> List[str]:
        return [self.episode_input_column]

    @property
    def side_effect_columns(self) -> List[str]:
        return [
            "structured_messages",
            "episode_metadata",
            "compaction_events",
            "trajectory_status",
            "trajectory_validation",
            "trajectory_judgment",
        ]
