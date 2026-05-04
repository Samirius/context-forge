"""Refiner — dedup/merge/prune bullets based on similarity."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ctxf.models.bullet import Bullet
from ctxf.models.delta import Delta
from ctxf.retrieval.embedder import Embedder

logger = logging.getLogger(__name__)


class Refiner:
    """Periodic pass that finds near-duplicate bullets via embeddings and merges them."""

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        similarity_threshold: float = 0.92,
        min_helpful_ratio: float = 0.2,
        min_total_feedback: int = 5,
    ) -> None:
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold
        self.min_helpful_ratio = min_helpful_ratio
        self.min_total_feedback = min_total_feedback

    async def find_duplicates(self, bullets: list[Bullet]) -> list[tuple[int, int, float]]:
        """Find pairs of near-duplicate bullets.

        Returns list of (i, j, similarity) tuples.
        """
        if len(bullets) < 2:
            return []

        # Compute embeddings if needed
        if self.embedder and self.embedder.available:
            to_embed = [b for b in bullets if not b.embedding]
            if to_embed:
                texts = [b.content for b in to_embed]
                embeddings = await self.embedder.embed(texts)
                for b, emb in zip(to_embed, embeddings):
                    b.embedding = emb

            # Check which bullets have embeddings
            has_emb = [(i, b) for i, b in enumerate(bullets) if b.embedding]
            pairs = []
            for ai in range(len(has_emb)):
                for bi in range(ai + 1, len(has_emb)):
                    i, ba = has_emb[ai]
                    j, bb = has_emb[bi]
                    if ba.section != bb.section:
                        continue
                    sim = float(np.dot(ba.embedding, bb.embedding) /
                                (np.linalg.norm(ba.embedding) * np.linalg.norm(bb.embedding) + 1e-8))
                    if sim >= self.similarity_threshold:
                        pairs.append((i, j, sim))
            return pairs

        # Fallback: simple text overlap heuristic
        pairs = []
        for i in range(len(bullets)):
            for j in range(i + 1, len(bullets)):
                if bullets[i].section != bullets[j].section:
                    continue
                # Jaccard on words
                words_a = set(bullets[i].content.lower().split())
                words_b = set(bullets[j].content.lower().split())
                if not words_a or not words_b:
                    continue
                overlap = len(words_a & words_b) / min(len(words_a), len(words_b))
                if overlap >= 0.7:
                    pairs.append((i, j, overlap))
        return pairs

    async def find_deprecated(self, bullets: list[Bullet]) -> list[Bullet]:
        """Find bullets that should be deprecated (consistently harmful)."""
        to_deprecate = []
        for b in bullets:
            total = b.helpful + b.harmful
            if total >= self.min_total_feedback:
                ratio = b.helpful / total
                if ratio < self.min_helpful_ratio:
                    to_deprecate.append(b)
        return to_deprecate

    async def refine(self, bullets: list[Bullet]) -> list[Delta]:
        """Run full refinement pass: merge duplicates + deprecate low-quality bullets."""
        deltas: list[Delta] = []

        # Find and merge duplicates
        duplicates = await self.find_duplicates(bullets)
        merged_indices: set[int] = set()
        for i, j, sim in duplicates:
            if i in merged_indices or j in merged_indices:
                continue
            bi, bj = bullets[i], bullets[j]
            # Keep the one with more helpful votes
            if bi.helpful >= bj.helpful:
                kept, removed = bi, bj
                merged_indices.add(j)
            else:
                kept, removed = bj, bi
                merged_indices.add(i)

            merged_content = kept.content
            if len(removed.content) > len(kept.content):
                merged_content = removed.content

            deltas.append(Delta.merge(
                bullet_ids=[bi.id, bj.id],
                merged_content=merged_content,
                reason=f"Merged near-duplicates (similarity={sim:.3f})",
            ))

        # Find bullets to deprecate
        to_deprecate = await self.find_deprecated(bullets)
        for b in to_deprecate:
            deltas.append(Delta.deprecate(
                bullet_id=b.id,
                reason=f"Low helpful ratio ({b.helpful}/{b.helpful + b.harmful})",
            ))

        return deltas
