"""RESEARCH step: conditional retry now fires on low yield, not just zero."""

import pytest

from shared.config import Settings
from worker.gemini import GeminiClient, GeminiResult


def _client(min_facilities_before_retry: int = 3) -> GeminiClient:
    settings = Settings(
        gemini_api_key="test-key",
        min_facilities_before_retry=min_facilities_before_retry,
    )
    return GeminiClient(settings)


@pytest.mark.asyncio
async def test_retry_fires_below_threshold_and_merges_results(monkeypatch):
    client = _client(min_facilities_before_retry=3)
    calls: list[str] = []

    async def fake_call(user_message: str) -> GeminiResult:
        calls.append(user_message)
        if len(calls) == 1:
            return GeminiResult(
                facilities_raw=[{"facility_name": "Plant A"}],
                search_queries=3,
            )
        return GeminiResult(
            facilities_raw=[{"facility_name": "Plant B"}],
            search_queries=2,
        )

    monkeypatch.setattr(client, "_call", fake_call)
    result = await client.research("Acme Corp", "Widgets")

    assert len(calls) == 2
    # first-call facility is kept, not discarded, and the retry's facility is appended
    assert [f["facility_name"] for f in result.facilities_raw] == ["Plant A", "Plant B"]
    assert result.search_queries == 5
    # retry prompt names the already-found facility so it isn't re-surfaced
    assert "Plant A" in calls[1]


@pytest.mark.asyncio
async def test_no_retry_when_yield_meets_threshold(monkeypatch):
    client = _client(min_facilities_before_retry=3)
    calls: list[str] = []

    async def fake_call(user_message: str) -> GeminiResult:
        calls.append(user_message)
        return GeminiResult(
            facilities_raw=[{"facility_name": f"Plant {i}"} for i in range(3)],
            search_queries=4,
        )

    monkeypatch.setattr(client, "_call", fake_call)
    result = await client.research("Acme Corp", "Widgets")

    assert len(calls) == 1
    assert len(result.facilities_raw) == 3
