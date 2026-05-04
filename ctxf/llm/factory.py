"""LLM provider factory — select provider based on config."""

from __future__ import annotations

from ctxf.config import Settings
from ctxf.llm.base import LLMProvider
from ctxf.llm.openai_compat import OpenAICompatProvider
from ctxf.llm.ollama import OllamaProvider


def create_llm(settings: Settings | None = None, role: str = "default") -> LLMProvider:
    """Create an LLM provider from settings.

    If the base_url contains 'ollama' or '11434', uses OllamaProvider.
    Otherwise, uses OpenAICompatProvider.
    """
    if settings is None:
        settings = Settings()

    model = settings.get_model(role)
    base_url = settings.llm_base_url
    api_key = settings.llm_api_key

    if "ollama" in base_url.lower() or "11434" in base_url:
        return OllamaProvider(model=model, base_url=base_url)

    return OpenAICompatProvider(model=model, base_url=base_url, api_key=api_key)
