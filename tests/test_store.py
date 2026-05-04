"""Tests for SQLite store and data models."""
import json
import os
import tempfile

import pytest

from ctxf.models.bullet import Bullet
from ctxf.models.delta import Delta, DeltaOp
from ctxf.models.playbook import Playbook
from ctxf.store.sqlite_store import SqliteStore


@pytest.fixture
def store():
    """Create a temporary SQLite store."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    s = SqliteStore(path)
    yield s
    s.close()
    os.unlink(path)


@pytest.fixture
def playbook_id(store):
    """Create a test playbook."""
    pb = store.create_playbook(name="test")
    return pb.id


class TestPlaybook:
    def test_create_playbook(self, store):
        pb = store.create_playbook(name="my-project")
        assert pb.name == "my-project"
        assert pb.version == 1
        assert len(pb.sections) > 0
        assert pb.id is not None

    def test_get_playbook(self, store, playbook_id):
        pb = store.get_playbook(playbook_id)
        assert pb is not None
        assert pb.id == playbook_id

    def test_get_default_playbook(self, store, playbook_id):
        pb = store.get_playbook()  # No ID = get first
        assert pb is not None

    def test_active_bullets_empty(self, store, playbook_id):
        pb = store.get_playbook(playbook_id)
        assert pb.active_bullets() == []


class TestDelta:
    def test_add_delta(self):
        d = Delta.add("technical", "Always validate inputs", reason="test")
        assert d.op == DeltaOp.ADD
        assert d.section == "technical"
        assert d.content == "Always validate inputs"

    def test_incr_delta(self):
        d = Delta.incr("TEC-00001", "helpful", reason="worked well")
        assert d.op == DeltaOp.INCR
        assert d.bullet_id == "TEC-00001"

    def test_deprecate_delta(self):
        d = Delta.deprecate("TEC-00001", reason="outdated")
        assert d.op == DeltaOp.DEPRECATE

    def test_delta_roundtrip(self):
        d = Delta.add("governance", "Log everything", reason="seed")
        serialized = d.to_dict()
        restored = Delta.from_dict(serialized)
        assert restored.op == d.op
        assert restored.section == d.section
        assert restored.content == d.content


class TestApplyDeltas:
    def test_add_bullet(self, store, playbook_id):
        delta = Delta.add("technical", "Use retries on API failures")
        results = store.apply_deltas(playbook_id, [delta])
        assert len(results) == 1

        pb = store.get_playbook(playbook_id)
        bullets = pb.active_bullets()
        assert len(bullets) == 1
        assert "retries" in bullets[0].content

    def test_add_multiple_bullets(self, store, playbook_id):
        deltas = [
            Delta.add("technical", "Strategy A"),
            Delta.add("technical", "Strategy B"),
            Delta.add("governance", "Policy X"),
        ]
        store.apply_deltas(playbook_id, deltas)
        pb = store.get_playbook(playbook_id)
        assert len(pb.active_bullets()) == 3

    def test_incr_helpful(self, store, playbook_id):
        # Add a bullet first
        store.apply_deltas(playbook_id, [Delta.add("technical", "Test strategy")])
        pb = store.get_playbook(playbook_id)
        bullet_id = pb.active_bullets()[0].id

        # Increment helpful
        store.apply_deltas(playbook_id, [Delta.incr(bullet_id, "helpful")])
        pb = store.get_playbook(playbook_id)
        assert pb.active_bullets()[0].helpful == 1

    def test_deprecate_bullet(self, store, playbook_id):
        store.apply_deltas(playbook_id, [Delta.add("technical", "Old strategy")])
        pb = store.get_playbook(playbook_id)
        bullet_id = pb.active_bullets()[0].id

        store.apply_deltas(playbook_id, [Delta.deprecate(bullet_id)])
        pb = store.get_playbook(playbook_id)
        assert len(pb.active_bullets()) == 0

    def test_stats(self, store, playbook_id):
        store.apply_deltas(playbook_id, [
            Delta.add("technical", "A"),
            Delta.add("governance", "B"),
        ])
        stats = store.get_stats(playbook_id)
        assert stats["active_bullets"] == 2
        assert "technical" in stats["by_section"]
        assert "governance" in stats["by_section"]
