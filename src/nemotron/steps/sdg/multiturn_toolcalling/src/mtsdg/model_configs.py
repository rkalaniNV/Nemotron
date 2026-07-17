"""Model roles and Data Designer model/provider configuration.

Binds the pipeline's model roles to aliases on a provider:

| Role                | alias          | model                                 |
|---------------------|----------------|---------------------------------------|
| User agent          | ``user``       | teacher-grade model                   |
| Assistant agent     | ``assistant``  | teacher-grade model                   |
| Judge               | ``judge``      | teacher-grade model                   |
| Compression         | ``compressor`` | teacher-grade model                   |
| Bulk candidates     | ``bulk``       | smaller/faster model                  |

Two provider paths are supported: the built-in ``nvidia`` provider (reads the
``NVIDIA_API_KEY`` env var at run time) and a custom OpenAI-compatible provider /
direct HTTP facade for an internal proxy. In every case the API key is read from
the environment — never hard-coded. Data Designer resolves the uppercase string as
an env-var *name*, so no literal key is written into any config artifact; nothing
here needs a real key to import or to build a config.
"""

from __future__ import annotations

import os
from typing import List

# Import lazily-friendly names; these only require data-designer installed.
from data_designer.config.models import (
    ChatCompletionInferenceParams,
    ModelConfig,
    ModelProvider,
)

NVIDIA_ENDPOINT = "https://integrate.api.nvidia.com/v1"
NVIDIA_API_KEY_ENV = "NVIDIA_API_KEY"  # resolved from env at runtime by DD

MODEL_SUPER = "nvidia/nemotron-3-super-120b-a12b"
MODEL_NANO = "nvidia/nemotron-3-nano-30b-a3b"

# Model aliases used across the pipeline.
TEACHER = "teacher"        # high-fidelity generation
USER = "user"              # User Agent (separate-agent loop)
ASSISTANT = "assistant"    # Assistant Agent (majority-voted)
JUDGE = "judge"
COMPRESSOR = "compressor"
BULK = "bulk"


def nvidia_provider() -> ModelProvider:
    """The NVIDIA Build API provider. ``api_key`` is the env-var *name*."""
    return ModelProvider(
        name="nvidia",
        endpoint=NVIDIA_ENDPOINT,
        provider_type="openai",
        api_key=NVIDIA_API_KEY_ENV,
    )


def default_model_configs(
    *,
    teacher_temperature: float = 0.5,
    judge_temperature: float = 0.3,
    bulk_temperature: float = 0.7,
    timeout: int = 180,
) -> List[ModelConfig]:
    """Model configs binding the design's roles to aliases on the nvidia provider."""
    gen_params = ChatCompletionInferenceParams(
        temperature=teacher_temperature, top_p=0.95, timeout=timeout, max_tokens=4096
    )
    judge_params = ChatCompletionInferenceParams(
        temperature=judge_temperature, top_p=0.95, timeout=timeout, max_tokens=2048
    )
    bulk_params = ChatCompletionInferenceParams(
        temperature=bulk_temperature, top_p=0.95, timeout=timeout, max_tokens=4096
    )
    return [
        ModelConfig(alias=TEACHER, model=MODEL_SUPER, provider="nvidia", inference_parameters=gen_params),
        ModelConfig(alias=USER, model=MODEL_SUPER, provider="nvidia", inference_parameters=bulk_params),
        ModelConfig(alias=ASSISTANT, model=MODEL_SUPER, provider="nvidia", inference_parameters=gen_params),
        ModelConfig(alias=JUDGE, model=MODEL_SUPER, provider="nvidia", inference_parameters=judge_params),
        ModelConfig(alias=COMPRESSOR, model=MODEL_SUPER, provider="nvidia", inference_parameters=gen_params),
        ModelConfig(alias=BULK, model=MODEL_NANO, provider="nvidia", inference_parameters=bulk_params),
    ]


def api_key_present() -> bool:
    """True if a real (non-empty) NVIDIA_API_KEY is exported."""
    return bool(os.environ.get(NVIDIA_API_KEY_ENV, "").strip())


class DirectChatFacade:
    """Minimal OpenAI-compatible chat facade — a direct synchronous HTTP client.

    Mimics the model-facade surface that ``mtsdg.core.llm.call_llm`` needs:
    ``completion(chat_messages, **kwargs)`` returning an object with
    ``.choices[0].message.{content, reasoning_content, tool_calls}``. Uses a
    bounded read timeout so a backend that accepts a request but never responds
    fails fast rather than blocking the run (see ``direct_facades_from_env``).

    Temperature/top_p are intentionally not sent (GPT-5-family models reject them).
    """

    def __init__(self, *, model: str, base_url: str, api_key: str, max_tokens: int = 4096, timeout: int = 180):
        self.model = model
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.timeout = timeout

    @staticmethod
    def _to_openai_messages(chat_messages) -> list:
        out = []
        for m in chat_messages:
            get = (lambda k, d=None: m.get(k, d)) if isinstance(m, dict) else (lambda k, d=None: getattr(m, k, d))
            role = get("role") or "user"
            content = get("content")
            if isinstance(content, list):  # ChatML blocks -> flat text
                content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
            msg = {"role": role, "content": content or ""}
            # Preserve tool structure so the conversation history stays coherent
            # (assistant tool_calls paired with tool results).
            tool_calls = get("tool_calls")
            if tool_calls:
                msg["tool_calls"] = tool_calls
            tool_call_id = get("tool_call_id")
            if tool_call_id:
                msg["tool_call_id"] = tool_call_id
            name = get("name")
            if name:
                msg["name"] = name
            out.append(msg)
        return out

    def completion(self, chat_messages, **kwargs):
        import httpx
        from types import SimpleNamespace

        body = {
            "model": self.model,
            "messages": self._to_openai_messages(chat_messages),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        if "response_format" in kwargs and kwargs["response_format"]:
            body["response_format"] = kwargs["response_format"]
        # Granular timeout: a non-responding backend must surface on the READ
        # deadline in bounded time, not block for the whole timeout budget. Some
        # upstream models occasionally accept a request and never send headers.
        resp = httpx.post(
            self.url,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json=body,
            timeout=httpx.Timeout(connect=10.0, read=self.timeout, write=30.0, pool=10.0),
        )
        resp.raise_for_status()
        data = resp.json()
        choices = [
            SimpleNamespace(
                message=SimpleNamespace(
                    role="assistant",
                    content=(c.get("message", {}) or {}).get("content", "") or "",
                    reasoning_content=(c.get("message", {}) or {}).get("reasoning_content"),
                    tool_calls=(c.get("message", {}) or {}).get("tool_calls"),
                )
            )
            for c in data.get("choices", [])
        ]
        return SimpleNamespace(choices=choices)

    async def acompletion(self, chat_messages, **kwargs):
        return self.completion(chat_messages, **kwargs)


def direct_facades_from_env(aliases: List[str]) -> dict:
    """Build ``{alias: DirectChatFacade}`` straight from env — a direct synchronous
    HTTP client to the OpenAI-compatible endpoint.

    Using a direct client (rather than Data Designer's built-in model client) keeps
    the request path simple and gives us a bounded read timeout, so a backend that
    accepts a request but never responds fails fast instead of blocking the run.
    Reads LLM_MODEL / LLM_BASE_URL / LLM_API_KEY; ensures the base URL ends in
    ``/v1``. ``LLM_TIMEOUT`` bounds the per-request read (default 60s).
    """
    model = os.environ["LLM_MODEL"]
    base = os.environ.get("LLM_BASE_URL", "https://inference-api.nvidia.com").rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    key = os.environ.get("LLM_API_KEY", "")
    max_tokens = int(os.environ.get("LLM_MAX_TOKENS", "8192"))
    timeout = int(os.environ.get("LLM_TIMEOUT", "60"))
    return {
        a: DirectChatFacade(model=model, base_url=base, api_key=key, max_tokens=max_tokens, timeout=timeout)
        for a in aliases
    }


# --------------------------------------------------------------------------- #
# Custom OpenAI-compatible provider (e.g. an internal LiteLLM proxy)
# --------------------------------------------------------------------------- #


def custom_openai_provider(
    *,
    name: str = "custom_openai",
    endpoint: str,
    api_key_env: str = "LLM_API_KEY",
) -> ModelProvider:
    """An OpenAI-compatible provider (calls ``{endpoint}/chat/completions``).

    ``api_key_env`` is the uppercase env-var *name* Data Designer resolves at run
    time, so the literal key is never written into any config artifact.
    """
    return ModelProvider(
        name=name,
        endpoint=endpoint,
        provider_type="openai",
        api_key=api_key_env,
    )


def custom_openai_model_configs(
    *,
    model: str,
    provider_name: str = "custom_openai",
    temperature: float = 0.5,
    judge_temperature: float = 0.3,
    max_tokens: int = 8192,
    timeout: int = 300,
    max_parallel_requests: int = 2,
) -> List[ModelConfig]:
    """Bind every pipeline role (teacher/judge/compressor/bulk) to one proxy model.

    ``temperature``/``top_p`` are intentionally NOT sent: GPT-5-family models on
    this proxy reject them (only default sampling is supported). Kept as function
    args for API symmetry but not forwarded.

    ``max_parallel_requests`` is kept low (2) so concurrent episodes do not burst
    the shared retriever's vLLM backend into a stall.
    """
    gen = ChatCompletionInferenceParams(timeout=timeout, max_tokens=max_tokens, max_parallel_requests=max_parallel_requests)
    judge = ChatCompletionInferenceParams(timeout=timeout, max_tokens=2048, max_parallel_requests=max_parallel_requests)
    return [
        ModelConfig(alias=TEACHER, model=model, provider=provider_name, inference_parameters=gen, skip_health_check=True),
        ModelConfig(alias=USER, model=model, provider=provider_name, inference_parameters=gen, skip_health_check=True),
        ModelConfig(alias=ASSISTANT, model=model, provider=provider_name, inference_parameters=gen, skip_health_check=True),
        ModelConfig(alias=JUDGE, model=model, provider=provider_name, inference_parameters=judge, skip_health_check=True),
        ModelConfig(alias=COMPRESSOR, model=model, provider=provider_name, inference_parameters=gen, skip_health_check=True),
        ModelConfig(alias=BULK, model=model, provider=provider_name, inference_parameters=gen, skip_health_check=True),
    ]
