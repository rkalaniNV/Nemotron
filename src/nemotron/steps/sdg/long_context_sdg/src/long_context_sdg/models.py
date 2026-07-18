"""OpenAI-compatible judge client used by offline evaluation."""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

import httpx

from .config import PipelineConfig


class JudgeChatFacade:
    def __init__(
        self,
        *,
        model: str,
        endpoint: str,
        api_key: str,
        parameters: dict[str, Any],
        timeout: float = 300,
        connect_timeout: float = 10,
    ):
        self.model = model
        self.endpoint = endpoint.rstrip("/")
        self.parameters = {
            key: value
            for key, value in parameters.items()
            if key not in {"timeout", "max_parallel_requests"}
        }
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self.client = httpx.Client(
            headers=headers,
            timeout=httpx.Timeout(
                timeout=timeout,
                connect=connect_timeout,
                write=30,
                pool=10,
            ),
        )

    def completion(self, messages: list[Any], **kwargs: Any):
        normalized = []
        for message in messages:
            if isinstance(message, dict):
                normalized.append(message)
            elif hasattr(message, "model_dump"):
                normalized.append(message.model_dump(exclude_none=True))
            else:
                normalized.append(
                    {
                        "role": getattr(message, "role", "user"),
                        "content": getattr(message, "content", "") or "",
                    }
                )
        body = {
            "model": self.model,
            "messages": normalized,
            **self.parameters,
            **kwargs,
        }
        response = self.client.post(
            self.endpoint + "/chat/completions",
            json=body,
        )
        response.raise_for_status()
        message = response.json()["choices"][0]["message"]
        return SimpleNamespace(message=SimpleNamespace(**message))

    def close(self) -> None:
        self.client.close()


def evaluation_judge_models(cfg: PipelineConfig) -> dict[str, Any]:
    """Build the judge client used by ``evaluate --rejudge``."""
    providers = {provider.name: provider for provider in cfg.providers}
    model = next(item for item in cfg.models if item.alias == "judge")
    provider = providers.get(model.provider)
    if provider is None:
        raise ValueError(
            f"model `{model.alias}` refers to unknown provider `{model.provider}`"
        )
    key_ref = provider.api_key_env.strip()
    api_key = os.environ.get(key_ref, "").strip() if key_ref else ""
    if key_ref and not api_key:
        raise ValueError(
            f"model provider `{provider.name}` needs environment variable `{key_ref}`"
        )
    return {
        "judge": JudgeChatFacade(
            model=model.model,
            endpoint=provider.endpoint,
            api_key=api_key,
            parameters=model.inference_parameters,
        )
    }
