"""Service contracts shared by independent generation stages."""

from __future__ import annotations

import os
from typing import Any, Literal

from pydantic import Field

from .config_base import StrictConfigModel


class ProviderConfig(StrictConfigModel):
    name: str
    endpoint: str
    api_key_env: str


class ModelConfig(StrictConfigModel):
    alias: str
    model: str
    provider: str = "primary"
    skip_health_check: bool = False
    inference_parameters: dict[str, Any] = Field(default_factory=dict)


class RetrieverFields(StrictConfigModel):
    id: str = "id"
    text: str = "text"
    title: str = "title"
    source: str = "source"
    score: str = "score"
    url: str = "url"
    date: str = "date"


class RetrieverConfig(StrictConfigModel):
    endpoint: str
    method: Literal["GET", "POST"] = "POST"
    query_field: str = "query"
    top_k_field: str = "top_k"
    top_k: int = Field(4, ge=1, le=100)
    results_path: str = "chunks"
    fields: RetrieverFields = Field(default_factory=RetrieverFields)
    selection: Literal["ranked", "diverse", "sampled"] = "ranked"
    timeout_seconds: float = Field(45, gt=0)
    retries: int = Field(4, ge=1, le=20)
    backoff_seconds: float = Field(1.0, ge=0)
    headers: dict[str, str] = Field(default_factory=dict)
    extra_body: dict[str, Any] = Field(default_factory=dict)


def resolve_api_key(provider: ProviderConfig) -> str:
    """Resolve a provider key reference, preserving an explicit empty value."""
    reference = provider.api_key_env.strip()
    if not reference:
        return ""
    if reference not in os.environ:
        raise ValueError(f"model provider `{provider.name}` needs environment variable `{reference}`")
    return os.environ[reference]


__all__ = [
    "ModelConfig",
    "ProviderConfig",
    "RetrieverConfig",
    "RetrieverFields",
    "resolve_api_key",
]
