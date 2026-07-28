"""Integration: full job lifecycle with mocked Gemini/Geocode and fake Supabase."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from shared.config import Settings
from shared.models import Facility, FacilityType, JobStatus
from shared.redis_queue import JobQueue
from worker.gemini import GeminiClient, GeminiResult
from worker.geocode import GeocodeClient
from worker.graph import PipelineGraph
from worker.main import process_job


class InMemoryPersist:
    def __init__(self) -> None:
        self.suppliers: dict[str, dict[str, Any]] = {}
        self.facilities: list[dict[str, Any]] = []
        self.runs: list[dict[str, Any]] = []
        self._id = 0

    def find_cached_supplier(self, normalized_name: str, product: str):
        for s in self.suppliers.values():
            if s["normalized_name"] == normalized_name:
                facs = [
                    f
                    for f in self.facilities
                    if f["supplier_id"] == s["id"] and f.get("product") == product
                ]
                if facs:
                    return s, facs
        return None

    def upsert_supplier(self, *, name: str, normalized_name: str):
        for s in self.suppliers.values():
            if s["normalized_name"] == normalized_name:
                return s
        self._id += 1
        row = {
            "id": f"sup-{self._id}",
            "name": name,
            "normalized_name": normalized_name,
            "created_at": "2099-01-01T00:00:00+00:00",
        }
        self.suppliers[row["id"]] = row
        return row

    def upsert_facilities(self, supplier_id: str, facilities: list[Facility]) -> int:
        for f in facilities:
            key = (supplier_id, f.facility_name, f.facility_address)
            existing = next(
                (
                    i
                    for i, row in enumerate(self.facilities)
                    if (row["supplier_id"], row["facility_name"], row["facility_address"])
                    == key
                ),
                None,
            )
            row = {
                "supplier_id": supplier_id,
                "facility_name": f.facility_name,
                "facility_address": f.facility_address,
                "product": f.product,
                "latitude": f.latitude,
                "longitude": f.longitude,
                "facility_type": f.facility_type.value,
                "confidence": f.confidence,
                "source_url": f.source_url,
                "geocode_status": f.geocode_status,
            }
            if existing is not None:
                self.facilities[existing] = row
            else:
                self.facilities.append(row)
        return len(facilities)

    def insert_research_run(self, **kwargs):
        self.runs.append(kwargs)


@pytest.mark.asyncio
async def test_full_lifecycle_cost_math(queue: JobQueue, settings: Settings):
    result = await queue.enqueue("Integration Co", "bearings")
    job_id = result["job_id"]

    gemini = MagicMock(spec=GeminiClient)
    gemini.research = AsyncMock(
        return_value=GeminiResult(
            facilities_raw=[
                {
                    "facility_name": "Plant One",
                    "facility_address": "500 Factory Lane, Cleveland, OH",
                    "facility_type": "manufacturing",
                    "confidence": "high",
                    "latitude": None,
                    "longitude": None,
                    "source_url": "https://example.com",
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
        geocode.geocode_calls += 1
        await queue.incr_metric("geocodes", 1)
        return [
            f.model_copy(
                update={
                    "latitude": 41.5,
                    "longitude": -81.7,
                    "geocode_status": "ok",
                }
            )
            for f in facilities
        ]

    geocode.geocode_facilities = AsyncMock(side_effect=fake_geocode)

    persist = InMemoryPersist()
    graph = PipelineGraph(queue, gemini, geocode, persist, settings)  # type: ignore[arg-type]

    dequeued = await queue.dequeue(timeout=1)
    assert dequeued == job_id
    await process_job(queue, graph, job_id, settings)

    job = await queue.get_job(job_id)
    assert job["status"] == JobStatus.succeeded.value
    assert int(job["facilities_found"]) == 1
    assert len(persist.facilities) == 1
    assert persist.facilities[0]["geocode_status"] == "ok"
    # Facility is tagged with the product it was researched for.
    assert persist.facilities[0]["product"] == "bearings"
    assert len(persist.runs) == 1

    run = persist.runs[0]
    expected = queue.estimate_cost(
        input_tokens=1000,
        output_tokens=500,
        thinking_tokens=200,
        search_queries=4,
        geocode_calls=1,
    )
    assert run["est_cost_usd"] == expected
    assert run["input_tokens"] == 1000
    assert run["search_queries"] == 4
    assert run["geocode_calls"] == 1


@pytest.mark.asyncio
async def test_one_supplier_row_across_multiple_products(queue: JobQueue, settings: Settings):
    """Same company, two products → one supplier row, facilities partitioned by product."""
    persist = InMemoryPersist()
    geocode = MagicMock(spec=GeocodeClient)
    geocode.geocode_calls = 0
    geocode.geocode_facilities = AsyncMock(side_effect=lambda fs: fs)

    async def run_product(product: str, plant: str):
        gemini = MagicMock(spec=GeminiClient)
        gemini.research = AsyncMock(
            return_value=GeminiResult(
                facilities_raw=[
                    {
                        "facility_name": plant,
                        "facility_address": f"{plant} Rd, Town",
                        "facility_type": "manufacturing",
                        "confidence": "high",
                    }
                ],
                input_tokens=100,
                output_tokens=50,
                thinking_tokens=10,
                search_queries=2,
            )
        )
        graph = PipelineGraph(queue, gemini, geocode, persist, settings)  # type: ignore[arg-type]
        result = await queue.enqueue("Acme Corp", product, force=True)
        await queue.dequeue(timeout=1)
        await process_job(queue, graph, result["job_id"], settings)

    await run_product("steel", "Steel Plant")
    await run_product("bearings", "Bearing Plant")

    # One company → one supplier row.
    assert len(persist.suppliers) == 1
    supplier_id = next(iter(persist.suppliers))
    # Facilities tie to that single supplier, partitioned by product.
    assert {f["product"] for f in persist.facilities} == {"steel", "bearings"}
    assert all(f["supplier_id"] == supplier_id for f in persist.facilities)
    # Each run is audited with its own product.
    assert {r["product"] for r in persist.runs} == {"steel", "bearings"}


@pytest.mark.asyncio
async def test_cache_hit_skips_gemini(queue: JobQueue, settings: Settings):
    persist = InMemoryPersist()
    supplier = persist.upsert_supplier(name="Cached Co", normalized_name="cached-co")
    persist.upsert_facilities(
        supplier["id"],
        [
            Facility(
                facility_name="Existing",
                facility_address="1 Known St",
                product="springs",
                latitude=1.0,
                longitude=2.0,
                facility_type=FacilityType.manufacturing,
                confidence="high",
                geocode_status="ok",
            )
        ],
    )

    result = await queue.enqueue("Cached Co", "springs", force=True)
    job_id = result["job_id"]
    await queue.dequeue(timeout=1)

    gemini = MagicMock(spec=GeminiClient)
    gemini.research = AsyncMock()
    geocode = MagicMock(spec=GeocodeClient)
    geocode.geocode_calls = 0

    graph = PipelineGraph(queue, gemini, geocode, persist, settings)  # type: ignore[arg-type]
    await process_job(queue, graph, job_id, settings)

    gemini.research.assert_not_called()
    job = await queue.get_job(job_id)
    assert job["status"] == JobStatus.succeeded.value
    assert job["served_from_cache"] == "true"
    assert int(job["facilities_found"]) == 1
