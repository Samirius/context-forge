"""Tests for API endpoints."""
import json
import os
import tempfile

import pytest
from fastapi.testclient import TestClient

from ctxf.api.server import create_app
from ctxf.store.sqlite_store import SqliteStore
from ctxf.models.delta import Delta


@pytest.fixture
def client():
    """Create a test client with a temp database."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)

    os.environ["CTXF_DB_PATH"] = path

    # Seed some data
    store = SqliteStore(path)
    pb = store.create_playbook(name="test-api")
    store.apply_deltas(pb.id, [
        Delta.add("technical", "Use retries on API failures"),
        Delta.add("governance", "Log all operations"),
    ])
    store.close()

    app = create_app()
    tc = TestClient(app)
    tc._db_path = path
    tc._playbook_id = pb.id
    yield tc

    os.unlink(path)
    del os.environ["CTXF_DB_PATH"]


class TestHealthEndpoint:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"


class TestPlaybookEndpoints:
    def test_list_playbooks(self, client):
        resp = client.get("/v1/playbook")
        assert resp.status_code == 200
        data = resp.json()
        assert "playbooks" in data
        assert len(data["playbooks"]) >= 1

    def test_get_playbook(self, client):
        resp = client.get(f"/v1/playbook/{client._playbook_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "test-api"

    def test_get_stats(self, client):
        resp = client.get(f"/v1/playbook/{client._playbook_id}/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["active_bullets"] == 2


class TestRetrieveEndpoint:
    def test_retrieve(self, client):
        resp = client.post("/v1/retrieve", json={
            "playbook_id": client._playbook_id,
            "query": "API failures",
            "top_k": 5,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data


class TestFeedbackEndpoint:
    def test_add_insight(self, client):
        resp = client.post("/v1/feedback", json={
            "playbook_id": client._playbook_id,
            "new_insights": [{"section": "technical", "content": "New insight"}],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["applied"] == 1

    def test_helpful_feedback(self, client):
        # First get a bullet ID
        resp = client.get(f"/v1/playbook/{client._playbook_id}")
        bullets = resp.json().get("bullets", [])

        if bullets:
            resp = client.post("/v1/feedback", json={
                "playbook_id": client._playbook_id,
                "helpful_bullet_ids": [bullets[0]["id"]],
            })
            assert resp.status_code == 200
