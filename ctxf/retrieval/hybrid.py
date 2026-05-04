"""Hybrid retrieval: BM25 keyword search + cosine similarity on embeddings."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi

from ctxf.models.bullet import Bullet
from ctxf.retrieval.embedder import Embedder

logger = logging.getLogger(__name__)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    va = np.array(a)
    vb = np.array(b)
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercasing tokenizer."""
    return text.lower().split()


class HybridRetriever:
    """Retrieves relevant playbook bullets using BM25 + embedding similarity."""

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        top_k: int = 10,
        alpha: float = 0.5,
    ) -> None:
        """
        Args:
            embedder: Embedding provider (optional — keyword-only if None).
            top_k: Number of results to return.
            alpha: Weight for semantic vs keyword scores (0=pure keyword, 1=pure semantic).
        """
        self.embedder = embedder
        self.top_k = top_k
        self.alpha = alpha

    async def retrieve(self, query: str, bullets: list[Bullet]) -> list[tuple[Bullet, float]]:
        """Return top-k bullets scored by hybrid relevance.

        Returns list of (bullet, score) tuples, sorted descending by score.
        """
        if not bullets:
            return []

        # --- BM25 keyword scores ---
        bm25_scores: dict[str, float] = {}
        if query.strip():
            tokenized_corpus = [_tokenize(b.content) for b in bullets]
            valid_indices = [i for i, t in enumerate(tokenized_corpus) if t]
            if valid_indices:
                valid_corpus = [tokenized_corpus[i] for i in valid_indices]
                valid_bullets = [bullets[i] for i in valid_indices]
                bm25 = BM25Okapi(valid_corpus)
                tokenized_query = _tokenize(query)
                scores = bm25.get_scores(tokenized_query)
                max_bm25 = max(scores) if len(scores) > 0 and max(scores) > 0 else 1.0
                for i, s in enumerate(scores):
                    bm25_scores[valid_bullets[i].id] = s / max_bm25

        # --- Embedding scores ---
        sem_scores: dict[str, float] = {}
        if self.embedder and self.embedder.available and self.alpha > 0:
            query_emb = await self.embedder.embed_single(query)
            if query_emb:
                # Embed bullets that don't have embeddings cached
                to_embed = [b for b in bullets if not b.embedding]
                if to_embed:
                    texts = [b.content for b in to_embed]
                    embeddings = await self.embedder.embed(texts)
                    for b, emb in zip(to_embed, embeddings):
                        b.embedding = emb

                for b in bullets:
                    if b.embedding:
                        sem_scores[b.id] = _cosine_similarity(query_emb, b.embedding)

                # Normalize
                max_sem = max(sem_scores.values()) if sem_scores else 1.0
                if max_sem > 0:
                    sem_scores = {k: v / max_sem for k, v in sem_scores.items()}

        # --- Combine ---
        results: list[tuple[Bullet, float]] = []
        for b in bullets:
            ks = bm25_scores.get(b.id, 0.0)
            ss = sem_scores.get(b.id, 0.0)
            if self.alpha == 0:
                combined = ks
            elif not sem_scores:
                combined = ks  # fallback to keyword only
            else:
                combined = (1 - self.alpha) * ks + self.alpha * ss
            results.append((b, combined))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[: self.top_k]
