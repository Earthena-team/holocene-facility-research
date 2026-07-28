"""DISCOVER_PRODUCT: runs only when a job is submitted without a product."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from api.main import app
from shared.config import Settings
from shared.models import Facility, FacilityType, JobStatus
from shared.redis_queue import JobQueue
from worker.gemini import GeminiClient, GeminiResult, ProductDiscoveryResult
from worker.geocode import GeocodeClient
from worker.graph import PipelineGraph
from worker.main import process_job

from tests.test_integration import InMemoryPersist


@pytest.fixture
async def api_client(queue: JobQueue, settings: Settings):
    app.state.queue = queue
    app.state.settings = settings
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.mark.asyncio
async def test_job_accepted_without_product(api_client: AsyncClient):
    resp = await api_client.post("/jobs", json={"supplier_name": "Acme"})
    assert resp.status_code == 201
    job = await api_client.get(f"/jobs/{resp.json()['job_id']}")
    assert job.json()["product"] == ""


@pytest.mark.asyncio
async def test_repeat_blank_product_submissions_dedupe(api_client: AsyncClient):
    r1 = await api_client.post("/jobs", json={"supplier_name": "Acme"})
    r2 = await api_client.post("/jobs", json={"supplier_name": "Acme"})
    assert r1.status_code == 201
    assert r2.status_code == 200
    assert r2.json()["status"] == "duplicate"


@pytest.mark.asyncio
async def test_discover_product_then_research_runs_with_resolved_product(
    queue: JobQueue, settings: Settings
):
    result = await queue.enqueue("Acme Corp", "")
    job_id = result["job_id"]

    gemini = MagicMock(spec=GeminiClient)
    gemini.discover_products = AsyncMock(
        return_value=ProductDiscoveryResult(
            products=["Cold-rolled steel coil", "Steel wire"],
            input_tokens=50,
            output_tokens=30,
            thinking_tokens=10,
            search_queries=2,
        )
    )
    gemini.research = AsyncMock(
        return_value=GeminiResult(
            facilities_raw=[
                {
                    "facility_name": "Plant One",
                    "facility_address": "500 Factory Lane, Cleveland, OH",
                    "facility_type": "manufacturing",
                    "confidence": "high",
                }
            ],
            input_tokens=1000,
            output_tokens=500,
            thinking_tokens=200,
            search_queries=4,
        )
    )

    geocode = MagicMock(spec=GeocodeClient)
    geocode.geocode_calls = 0

    async def fake_geocode(facilities):
        return [f.model_copy(update={"geocode_status": "ok"}) for f in facilities]

    geocode.geocode_facilities = AsyncMock(side_effect=fake_geocode)

    persist = InMemoryPersist()
    graph = PipelineGraph(queue, gemini, geocode, persist, settings)  # type: ignore[arg-type]

    await queue.dequeue(timeout=1)
    await process_job(queue, graph, job_id, settings)

    gemini.discover_products.assert_called_once_with("Acme Corp")
    gemini.research.assert_called_once_with("Acme Corp", "Cold-rolled steel coil")

    job = await queue.get_job(job_id)
    assert job["status"] == JobStatus.succeeded.value
    assert job["product"] == "Cold-rolled steel coil"
    assert job["discovered_products"]
    # One supplier row for the company; product lives on facilities/runs, not the supplier.
    assert len(persist.suppliers) == 1
    supplier = next(iter(persist.suppliers.values()))
    assert "product" not in supplier
    # The discovered product is tied through to each facility found for it, and audited.
    assert persist.facilities
    assert all(f["product"] == "Cold-rolled steel coil" for f in persist.facilities)
    assert all(r["product"] == "Cold-rolled steel coil" for r in persist.runs)


@pytest.mark.asyncio
async def test_cache_hit_after_discovery_skips_research(queue: JobQueue, settings: Settings):
    persist = InMemoryPersist()
    supplier = persist.upsert_supplier(name="Cached Co", normalized_name="cached-co")
    persist.upsert_facilities(
        supplier["id"],
        [
            Facility(
                facility_name="Existing",
                facility_address="1 Known St",
                product="springs",
                facility_type=FacilityType.manufacturing,
                confidence="high",
                geocode_status="ok",
            )
        ],
    )

    result = await queue.enqueue("Cached Co", "")
    job_id = result["job_id"]

    gemini = MagicMock(spec=GeminiClient)
    gemini.discover_products = AsyncMock(
        return_value=ProductDiscoveryResult(products=["springs"])
    )
    gemini.research = AsyncMock()
    geocode = MagicMock(spec=GeocodeClient)
    geocode.geocode_calls = 0

    graph = PipelineGraph(queue, gemini, geocode, persist, settings)  # type: ignore[arg-type]
    await queue.dequeue(timeout=1)
    await process_job(queue, graph, job_id, settings)

    gemini.research.assert_not_called()
    job = await queue.get_job(job_id)
    assert job["status"] == JobStatus.succeeded.value
    assert job["served_from_cache"] == "true"
    assert job["product"] == "springs"


@pytest.mark.asyncio
async def test_discover_products_retries_on_empty(monkeypatch):
    settings = Settings(gemini_api_key="test-key")
    client = GeminiClient(settings)
    calls: list[str] = []

    async def fake_call(user_message: str) -> ProductDiscoveryResult:
        calls.append(user_message)
        if len(calls) == 1:
            return ProductDiscoveryResult(products=[], search_queries=1)
        return ProductDiscoveryResult(products=["Widgets"], search_queries=2)

    monkeypatch.setattr(client, "_call_product_discovery", fake_call)
    result = await client.discover_products("Obscure Co")

    assert len(calls) == 2
    assert result.products == ["Widgets"]
    assert result.search_queries == 3
