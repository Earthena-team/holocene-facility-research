"""API dedupe and force bypass tests."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from api.main import app
from shared.config import Settings
from shared.redis_queue import JobQueue


@pytest.fixture
async def api_client(queue: JobQueue, settings: Settings):
    app.state.queue = queue
    app.state.settings = settings
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.mark.asyncio
async def test_dedupe_returns_duplicate(api_client: AsyncClient):
    r1 = await api_client.post(
        "/jobs",
        json={"supplier_name": "Acme", "product": "steel"},
    )
    assert r1.status_code == 201
    job_id = r1.json()["job_id"]

    r2 = await api_client.post(
        "/jobs",
        json={"supplier_name": "Acme", "product": "steel"},
    )
    assert r2.status_code == 200
    body = r2.json()
    assert body["status"] == "duplicate"
    assert body["existing_job_id"] == job_id

    # Queue should still have only one job
    stats = await api_client.get("/stats")
    assert stats.json()["pending_normal"] == 1


@pytest.mark.asyncio
async def test_force_bypasses_dedupe(api_client: AsyncClient):
    r1 = await api_client.post(
        "/jobs",
        json={"supplier_name": "Acme", "product": "copper"},
    )
    assert r1.status_code == 201

    r2 = await api_client.post(
        "/jobs",
        json={"supplier_name": "Acme", "product": "copper", "force": True},
    )
    assert r2.status_code == 201
    assert r2.json()["job_id"] != r1.json()["job_id"]

    stats = await api_client.get("/stats")
    assert stats.json()["pending_normal"] == 2
