"""Catalog API — suppliers and facilities read endpoints."""

from __future__ import annotations

from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from api.main import app
from shared.config import Settings
from shared.redis_queue import JobQueue


class FakePersist:
    def __init__(self) -> None:
        self.suppliers: dict[str, dict[str, Any]] = {
            "s1": {
                "id": "s1",
                "name": "Acme Corp",
                "normalized_name": "acme-corp",
                "created_at": "2026-01-01T00:00:00+00:00",
            },
            "s2": {
                "id": "s2",
                "name": "Beta Steel",
                "normalized_name": "beta-steel",
                "created_at": "2026-01-02T00:00:00+00:00",
            },
        }
        self.facilities: dict[str, dict[str, Any]] = {
            "f1": {
                "id": "f1",
                "supplier_id": "s1",
                "supplier_name": "Acme Corp",
                "facility_name": "Plant A",
                "facility_address": "1 Main St",
                "product": "steel",
                "latitude": 40.0,
                "longitude": -74.0,
                "facility_type": "manufacturing",
                "confidence": "high",
                "source_url": "https://example.com",
                "geocode_status": "ok",
                "created_at": "2026-01-03T00:00:00+00:00",
            },
            "f2": {
                "id": "f2",
                "supplier_id": "s1",
                "supplier_name": "Acme Corp",
                "facility_name": "Warehouse B",
                "facility_address": "2 Side St",
                "product": "copper",
                "latitude": None,
                "longitude": None,
                "facility_type": "logistics",
                "confidence": "medium",
                "source_url": None,
                "geocode_status": "pending",
                "created_at": "2026-01-04T00:00:00+00:00",
            },
            "f3": {
                "id": "f3",
                "supplier_id": "s2",
                "supplier_name": "Beta Steel",
                "facility_name": "Mill C",
                "facility_address": "3 Mill Rd",
                "product": "steel",
                "latitude": 41.0,
                "longitude": -75.0,
                "facility_type": "manufacturing",
                "confidence": "low",
                "source_url": None,
                "geocode_status": "approximate",
                "created_at": "2026-01-05T00:00:00+00:00",
            },
        }

    def get_supplier(self, supplier_id: str) -> dict[str, Any] | None:
        return self.suppliers.get(supplier_id)

    def list_suppliers(
        self,
        *,
        q: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        rows = sorted(
            self.suppliers.values(),
            key=lambda r: r["created_at"],
            reverse=True,
        )
        if q:
            q_lower = q.lower()
            rows = [r for r in rows if q_lower in r["name"].lower()]
        return rows[offset : offset + limit]

    def get_facility(self, facility_id: str) -> dict[str, Any] | None:
        return self.facilities.get(facility_id)

    def list_facilities(
        self,
        *,
        supplier_id: str | None = None,
        product: str | None = None,
        facility_type: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        rows = sorted(
            self.facilities.values(),
            key=lambda r: r["created_at"],
            reverse=True,
        )
        if supplier_id:
            rows = [r for r in rows if r["supplier_id"] == supplier_id]
        if product is not None and product != "":
            rows = [r for r in rows if r["product"] == product]
        if facility_type:
            rows = [r for r in rows if r["facility_type"] == facility_type]
        return rows[offset : offset + limit]


@pytest.fixture
async def api_client(queue: JobQueue, settings: Settings):
    app.state.queue = queue
    app.state.settings = settings
    app.state.persist = FakePersist()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.mark.asyncio
async def test_list_suppliers(api_client: AsyncClient):
    r = await api_client.get("/suppliers")
    assert r.status_code == 200
    body = r.json()
    assert len(body["suppliers"]) == 2
    assert body["next_offset"] is None
    assert body["suppliers"][0]["name"] == "Beta Steel"


@pytest.mark.asyncio
async def test_list_suppliers_q_and_pagination(api_client: AsyncClient):
    r = await api_client.get("/suppliers", params={"q": "acme", "limit": 1, "offset": 0})
    assert r.status_code == 200
    body = r.json()
    assert len(body["suppliers"]) == 1
    assert body["suppliers"][0]["id"] == "s1"
    assert body["next_offset"] == 1


@pytest.mark.asyncio
async def test_get_supplier_ok_and_404(api_client: AsyncClient):
    ok = await api_client.get("/suppliers/s1")
    assert ok.status_code == 200
    assert ok.json()["normalized_name"] == "acme-corp"

    missing = await api_client.get("/suppliers/missing")
    assert missing.status_code == 404
    assert missing.headers["content-type"].startswith("application/problem+json")


@pytest.mark.asyncio
async def test_list_facilities_filters(api_client: AsyncClient):
    r = await api_client.get(
        "/facilities",
        params={"supplier_id": "s1", "product": "steel", "facility_type": "manufacturing"},
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body["facilities"]) == 1
    assert body["facilities"][0]["id"] == "f1"
    assert body["facilities"][0]["supplier_name"] == "Acme Corp"
    assert body["next_offset"] is None


@pytest.mark.asyncio
async def test_list_facilities_pagination(api_client: AsyncClient):
    r = await api_client.get("/facilities", params={"limit": 2, "offset": 0})
    assert r.status_code == 200
    body = r.json()
    assert len(body["facilities"]) == 2
    assert body["next_offset"] == 2


@pytest.mark.asyncio
async def test_get_facility_ok_and_404(api_client: AsyncClient):
    ok = await api_client.get("/facilities/f2")
    assert ok.status_code == 200
    assert ok.json()["facility_type"] == "logistics"

    missing = await api_client.get("/facilities/missing")
    assert missing.status_code == 404
