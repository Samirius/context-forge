"""Abstract base for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Message:
    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class LLMResponse:
    content: str
    model: str = ""
    usage: dict = field(default_factory=dict)
    raw: Any = None


class LLMProvider(ABC):
    """Base class for LLM providers."""

    def __init__(self, model: str, base_url: str, api_key: str = "") -> None:
        self.model = model
        self.base_url = base_url
        self.api_key = api_key

    @abstractmethod
    async def complete(self, messages: list[Message], **kwargs: Any) -> LLMResponse:
        """Send a chat completion request."""
        ...

    @abstractmethod
    async def complete_json(self, messages: list[Message], **kwargs: Any) -> dict:
        """Send a chat completion request expecting JSON output."""
        ...

    async def simple(self, system: str, user: str, **kwargs: Any) -> str:
        """Convenience: system + user → content string."""
        resp = await self.complete([
            Message(role="system", content=system),
            Message(role="user", content=user),
        ], **kwargs)
        return resp.content
