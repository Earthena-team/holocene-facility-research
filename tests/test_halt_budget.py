"""Halt mid-run and budget circuit breaker tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from shared.config import Settings
from shared.redis_queue import JobQueue
from worker.gemini import GeminiClient, GeminiResult
from worker.geocode import GeocodeClient
from worker.graph import BudgetPausedError, HaltedError, PipelineGraph


class FakePersist:
    def find_cached_supplier(self, normalized_name, product):
        return None

    def upsert_supplier(self, **kwargs):
        raise AssertionError("persist should not be called when halted before PERSIST")

    def upsert_facilities(self, *args, **kwargs):
        raise AssertionError("persist should not be called when halted before PERSIST")

    def insert_research_run(self, **kwargs):
        raise AssertionError("persist should not be called when halted before PERSIST")


@pytest.mark.asyncio
async def test_halt_before_next_step(queue: JobQueue, settings: Settings):
    result = await queue.enqueue("Halt Co", "pipes")
    job_id = result["job_id"]
    await queue.dequeue(timeout=1)

    gemini = MagicMock(spec=GeminiClient)
    gemini.research = AsyncMock(
        return_value=GeminiResult(
            facilities_raw=[
                {
                    "facility_name": "Plant",
                    "facility_address": "1 Rd",
                    "facility_type": "manufacturing",
                    "confidence": "high",
                }
            ],
            input_tokens=10,
            output_tokens=20,
            thinking_tokens=5,
            search_queries=3,
        )
    )
    geocode = MagicMock(spec=GeocodeClient)
    geocode.geocode_calls = 0
    geocode.geocode_facilities = AsyncMock(side_effect=lambda fs: fs)

    # Set halt after research would complete — we set it before run and
    # graph checks halt before every step including the first.
    await queue.set_halt(job_id)

    graph = PipelineGraph(queue, gemini, geocode, FakePersist(), settings)  # type: ignore[arg-type]
    with pytest.raises(HaltedError):
        await graph.run(job_id)

    gemini.research.assert_not_called()


@pytest.mark.asyncio
async def test_halt_between_steps_no_persist(queue: JobQueue, settings: Settings):
    result = await queue.enqueue("Halt Mid", "valves")
    job_id = result["job_id"]
    await queue.dequeue(timeout=1)

    async def research_then_halt(supplier_name, product):
        # After research, set halt so EXTRACT/PERSIST never run
        await queue.set_halt(job_id)
        return GeminiResult(
            facilities_raw=[
                {
                    "facility_name": "Plant",
                    "facility_address": "1 Rd, City",
                    "facility_type": "manufacturing",
                    "confidence": "high",
                }
            ],
            input_tokens=10,
            output_tokens=20,
            thinking_tokens=0,
            search_queries=3,
        )

    gemini = MagicMock(spec=GeminiClient)
    gemini.research = AsyncMock(side_effect=research_then_halt)
    geocode = MagicMock(spec=GeocodeClient)
    geocode.geocode_calls = 0

    persist = FakePersist()
    graph = PipelineGraph(queue, gemini, geocode, persist, settings)  # type: ignore[arg-type]

    with pytest.raises(HaltedError):
        await graph.run(job_id)

    gemini.research.assert_called_once()


@pytest.mark.asyncio
async def test_budget_breaker_skips_gemini(queue: JobQueue, settings: Settings):
    settings.daily_budget_usd = 1.0
    await queue.add_budget(5.0)  # already over budget

    result = await queue.enqueue("Budget Co", "gaskets")
    job_id = result["job_id"]
    await queue.dequeue(timeout=1)

    gemini = MagicMock(spec=GeminiClient)
    gemini.research = AsyncMock()
    geocode = MagicMock(spec=GeocodeClient)
    geocode.geocode_calls = 0

    graph = PipelineGraph(queue, gemini, geocode, FakePersist(), settings)  # type: ignore[arg-type]
    with pytest.raises(BudgetPausedError):
        await graph.run(job_id)

    gemini.research.assert_not_called()
