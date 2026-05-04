"""Ollama LLM provider — uses OpenAI-compatible endpoint served by Ollama."""

from __future__ import annotations

from ctxf.llm.openai_compat import OpenAICompatProvider


class OllamaProvider(OpenAICompatProvider):
    """LLM provider for local Ollama models.

    Ollama exposes an OpenAI-compatible endpoint at http://localhost:11434/v1
    so we can reuse the OpenAI provider with different defaults.
    """

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434/v1",
        **kwargs,
    ) -> None:
        super().__init__(model=model, base_url=base_url, api_key="ollama", **kwargs)
