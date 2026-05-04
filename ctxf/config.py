"""Configuration via environment variables and .env files."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment / .env file."""

    # --- Storage ---
    db_path: str = str(Path.home() / ".ctxf" / "playbook.db")

    # --- LLM ---
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    llm_default_model: str = "gpt-4o-mini"

    # Per-role model overrides (fall back to llm_default_model if empty)
    generator_model: str = ""
    reflector_model: str = ""
    curator_model: str = ""

    # --- Embeddings ---
    embed_base_url: str = ""  # defaults to llm_base_url
    embed_api_key: str = ""   # defaults to llm_api_key
    embed_model: str = "text-embedding-3-small"
    embed_dim: int = 1536

    # --- Retrieval ---
    retrieval_top_k: int = 10
    hybrid_alpha: float = 0.5  # weight for semantic vs keyword (0=key, 1=semantic)

    # --- Server ---
    server_host: str = "0.0.0.0"
    server_port: int = 8000

    # --- General ---
    log_level: str = "INFO"

    model_config = {
        "env_prefix": "CTXF_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    # --- helpers ---
    def get_model(self, role: str) -> str:
        override = getattr(self, f"{role}_model", "")
        return override or self.llm_default_model

    @property
    def effective_embed_base_url(self) -> str:
        return self.embed_base_url or self.llm_base_url

    @property
    def effective_embed_api_key(self) -> str:
        return self.embed_api_key or self.llm_api_key

    @property
    def embeddings_enabled(self) -> bool:
        return bool(self.effective_embed_api_key)


def get_settings() -> Settings:
    """Return a cached Settings singleton."""
    return Settings()
