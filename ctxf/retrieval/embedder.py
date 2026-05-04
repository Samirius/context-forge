"""Embedding provider — calls OpenAI-compatible /embeddings endpoint."""

from __future__ import annotations

import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


class Embedder:
    """Generates embeddings via an OpenAI-compatible API."""

    def __init__(
        self,
        base_url: str = "https://api.openai.com/v1",
        api_key: str = "",
        model: str = "text-embedding-3-small",
        dim: int = 1536,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.dim = dim
        self.timeout = timeout
        self._available: Optional[bool] = None

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns list of embedding vectors."""
        if not self.api_key:
            return [[] for _ in texts]

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    f"{self.base_url}/embeddings",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "model": self.model,
                        "input": texts,
                        "dimensions": self.dim,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                embeddings = [d["embedding"] for d in sorted(data["data"], key=lambda x: x["index"])]
                self._available = True
                return embeddings
        except Exception as e:
            logger.warning("Embedding failed: %s", e)
            self._available = False
            return [[] for _ in texts]

    async def embed_single(self, text: str) -> list[float]:
        results = await self.embed([text])
        return results[0] if results else []

    @property
    def available(self) -> bool:
        if self._available is None:
            return bool(self.api_key)
        return self._available
