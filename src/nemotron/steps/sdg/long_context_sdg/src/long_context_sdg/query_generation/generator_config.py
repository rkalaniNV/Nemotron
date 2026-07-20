"""Data Designer column contract for query generation."""

from __future__ import annotations

from typing import Any, Literal

from data_designer.config.base import SingleColumnConfig


class SyntheticQueryConfig(SingleColumnConfig):
    column_type: Literal["persona-query-generator"] = "persona-query-generator"
    model_alias: str = "assistant"
    candidate_input_column: str = "candidate_input"
    persona_columns: dict[str, str]
    pipeline: dict[str, Any]

    @property
    def required_columns(self) -> list[str]:
        return [self.candidate_input_column, *self.persona_columns.values()]

    @property
    def side_effect_columns(self) -> list[str]:
        return ["synthetic_seed", "query_record", "query_status", "query_validation"]
