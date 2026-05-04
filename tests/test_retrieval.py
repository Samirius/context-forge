"""Tests for hybrid retrieval engine."""
import os
import tempfile

import pytest

from ctxf.models.bullet import Bullet
from ctxf.models.delta import Delta
from ctxf.store.sqlite_store import SqliteStore
from ctxf.retrieval.hybrid import HybridRetriever


@pytest.fixture
def seeded_store():
    """Create a store with seeded bullets."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    s = SqliteStore(path)
    pb = s.create_playbook(name="test-retrieval")

    bullets_data = [
        ("technical", "Use exponential backoff for retry logic on external API calls"),
        ("technical", "Always validate input before processing API requests"),
        ("governance", "Log all payment transactions for audit trail"),
        ("domain", "Stripe webhooks need signature verification"),
        ("style", "Use structured error responses with error codes"),
        ("workflow", "Break complex tasks into smaller, verifiable steps"),
        ("technical", "Cache frequently accessed data with appropriate TTL"),
        ("governance", "Rotate secrets and credentials regularly"),
    ]

    deltas = [Delta.add(section, content) for section, content in bullets_data]
    s.apply_deltas(pb.id, deltas)

    yield s, pb.id
    s.close()
    os.unlink(path)


class TestHybridRetriever:
    @pytest.mark.asyncio
    async def test_retrieve_keyword_match(self, seeded_store):
        store, pb_id = seeded_store
        pb = store.get_playbook(pb_id)
        bullets = pb.active_bullets()

        retriever = HybridRetriever(embedder=None, top_k=5, alpha=0.0)
        results = await retriever.retrieve("API retry logic", bullets)

        assert len(results) > 0
        # Top result should mention retry or API
        top_bullet = results[0][0]
        assert "retry" in top_bullet.content.lower() or "api" in top_bullet.content.lower()

    @pytest.mark.asyncio
    async def test_retrieve_returns_top_k(self, seeded_store):
        store, pb_id = seeded_store
        pb = store.get_playbook(pb_id)
        bullets = pb.active_bullets()

        retriever = HybridRetriever(embedder=None, top_k=3, alpha=0.0)
        results = await retriever.retrieve("payment processing", bullets)

        assert len(results) <= 3

    @pytest.mark.asyncio
    async def test_retrieve_empty_playbook(self):
        retriever = HybridRetriever(embedder=None, top_k=5)
        results = await retriever.retrieve("test query", [])
        assert results == []

    @pytest.mark.asyncio
    async def test_retrieve_scores_non_negative(self, seeded_store):
        store, pb_id = seeded_store
        pb = store.get_playbook(pb_id)
        bullets = pb.active_bullets()

        retriever = HybridRetriever(embedder=None, top_k=5, alpha=0.0)
        results = await retriever.retrieve("error handling", bullets)

        for bullet, score in results:
            assert score >= 0.0

    @pytest.mark.asyncio
    async def test_retrieve_sorted_descending(self, seeded_store):
        store, pb_id = seeded_store
        pb = store.get_playbook(pb_id)
        bullets = pb.active_bullets()

        retriever = HybridRetriever(embedder=None, top_k=5, alpha=0.0)
        results = await retriever.retrieve("API calls", bullets)

        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)
