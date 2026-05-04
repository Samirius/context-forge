"""OpenAI-compatible LLM provider (works with OpenAI, OpenRouter, any compatible API)."""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from ctxf.llm.base import LLMProvider, LLMResponse, Message

logger = logging.getLogger(__name__)


class OpenAICompatProvider(LLMProvider):
    """LLM provider using the OpenAI chat completions API format."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
        api_key: str = "",
        timeout: float = 120.0,
    ) -> None:
        super().__init__(model=model, base_url=base_url.rstrip("/"), api_key=api_key)
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    async def complete(self, messages: list[Message], **kwargs: Any) -> LLMResponse:
        payload = {
            "model": kwargs.pop("model", self.model),
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": kwargs.pop("temperature", 0.3),
            **kwargs,
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return LLMResponse(
            content=content,
            model=data.get("model", self.model),
            usage=usage,
            raw=data,
        )

    async def complete_json(self, messages: list[Message], **kwargs: Any) -> dict:
        kwargs.setdefault("response_format", {"type": "json_object"})
        resp = await self.complete(messages, **kwargs)
        try:
            return json.loads(resp.content)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            text = resp.content
            if "```json" in text:
                start = text.index("```json") + 7
                end = text.index("```", start)
                return json.loads(text[start:end])
            if "```" in text:
                start = text.index("```") + 3
                end = text.index("```", start)
                return json.loads(text[start:end])
            raise
