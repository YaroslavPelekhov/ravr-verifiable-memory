from __future__ import annotations

import importlib
import os
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient


def test_http_contract_and_authentication() -> None:
    with tempfile.TemporaryDirectory() as directory:
        os.environ["RAVR_MEMORY_DB"] = str(Path(directory) / "memory.sqlite3")
        os.environ["RAVR_MEMORY_API_KEY"] = "test-secret"
        import ravr_science.aml_service as service

        service.store.cache_clear()
        importlib.reload(service)
        client = TestClient(service.app)
        assert client.get("/health").status_code == 200
        assert client.post("/add", json={}).status_code == 401
        headers = {"X-Api-Key": "test-secret"}
        payload = {
            "request_id": "r1",
            "messages": [{"role": "user", "content": "Mira moved to Lisbon."}],
            "user_id": "u1",
            "session_id": "s1",
        }
        added = client.post("/add", headers=headers, json=payload)
        assert added.status_code == 200
        assert added.json() == {
            "success": True,
            "request_id": "r1",
            "user_id": "u1",
            "session_id": "s1",
        }
        found = client.post(
            "/search",
            headers={"Authorization": "Bearer test-secret"},
            json={"query": "Where did Mira move?", "user_id": "u1", "top_k": 100},
        )
        assert found.status_code == 200
        assert "Lisbon" in found.json()["data"][0]["content"]


def test_service_fails_closed_when_api_key_is_missing() -> None:
    with tempfile.TemporaryDirectory() as directory:
        os.environ["RAVR_MEMORY_DB"] = str(Path(directory) / "memory.sqlite3")
        os.environ.pop("RAVR_MEMORY_API_KEY", None)
        import ravr_science.aml_service as service

        service.store.cache_clear()
        importlib.reload(service)
        client = TestClient(service.app)
        assert client.get("/health").status_code == 200
        assert client.post("/add", headers={"X-Api-Key": "anything"}, json={}).status_code == 401
        assert client.post("/search", headers={"X-Api-Key": "anything"}, json={}).status_code == 401
