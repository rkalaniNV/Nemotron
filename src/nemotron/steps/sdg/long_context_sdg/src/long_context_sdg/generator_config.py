"""Data Designer column configuration."""

from __future__ import annotations

from typing import Any, Literal

from data_designer.config.base import SingleColumnConfig


class LongContextEpisodeConfig(SingleColumnConfig):
    column_type: Literal["long-context-episode-simulator"] = (
        "long-context-episode-simulator"
    )
    model_alias: str = "assistant"
    episode_input_column: str = "episode_input"
    pipeline: dict[str, Any]
    checkpoint_path: str
    run_id: str

    @property
    def required_columns(self) -> list[str]:
        return [self.episode_input_column]

    @property
    def side_effect_columns(self) -> list[str]:
        return [
            "canonical_record",
            "trajectory_status",
            "trajectory_validation",
            "structured_messages",
        ]
