"""RESEARCH step: conditional retry on low yield or missing upstream (origin) sites."""

import pytest

from shared.config import Settings
from worker.gemini import GeminiClient, GeminiResult, _has_upstream_sites


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
    # low yield + no upstream → prefer upstream angle
    assert "Plant A" in calls[1]
    assert "plantation" in calls[1].lower()
    assert "OpenSupplyHub" not in calls[1]


@pytest.mark.asyncio
async def test_no_retry_when_yield_meets_threshold_with_upstream(monkeypatch):
    client = _client(min_facilities_before_retry=3)
    calls: list[str] = []

    async def fake_call(user_message: str) -> GeminiResult:
        calls.append(user_message)
        return GeminiResult(
            facilities_raw=[
                {"facility_name": "North Plantation"},
                {"facility_name": "River Mill"},
                {"facility_name": "West Estate"},
            ],
            search_queries=4,
        )

    monkeypatch.setattr(client, "_call", fake_call)
    result = await client.research("Acme Corp", "Widgets")

    assert len(calls) == 1
    assert len(result.facilities_raw) == 3


@pytest.mark.asyncio
async def test_retry_when_refinery_only_first_pass(monkeypatch):
    """Enough facilities but only refineries → upstream coverage retry."""
    client = _client(min_facilities_before_retry=3)
    calls: list[str] = []

    async def fake_call(user_message: str) -> GeminiResult:
        calls.append(user_message)
        if len(calls) == 1:
            return GeminiResult(
                facilities_raw=[
                    {
                        "facility_name": "Rotterdam Palm Oil Refinery",
                        "facility_type": "raw_material",
                    },
                    {
                        "facility_name": "Port Klang Refining Plant",
                        "facility_type": "raw_material",
                    },
                    {
                        "facility_name": "Wichita Manufacturing Hub",
                        "facility_type": "manufacturing",
                    },
                ],
                search_queries=4,
            )
        return GeminiResult(
            facilities_raw=[{"facility_name": "Sumatra Palm Estate"}],
            search_queries=3,
        )

    monkeypatch.setattr(client, "_call", fake_call)
    result = await client.research("Cargill", "palm oil")

    assert len(calls) == 2
    assert "plantation" in calls[1].lower()
    assert "farm" in calls[1].lower()
    assert "cooperative" in calls[1].lower()
    assert "OpenSupplyHub" not in calls[1]
    assert [f["facility_name"] for f in result.facilities_raw] == [
        "Rotterdam Palm Oil Refinery",
        "Port Klang Refining Plant",
        "Wichita Manufacturing Hub",
        "Sumatra Palm Estate",
    ]


@pytest.mark.asyncio
async def test_retry_when_nursery_or_mill_only_first_pass(monkeypatch):
    """Nurseries/mills/processing do not satisfy origin coverage → upstream retry."""
    client = _client(min_facilities_before_retry=3)
    calls: list[str] = []

    async def fake_call(user_message: str) -> GeminiResult:
        calls.append(user_message)
        if len(calls) == 1:
            return GeminiResult(
                facilities_raw=[
                    {
                        "facility_name": "Mondelez India Foods Ltd. Cocoa Operations & Nursery",
                        "facility_type": "raw_material",
                    },
                    {
                        "facility_name": "Cadbury Nigeria Plc Cocoa Processing Plant",
                        "facility_type": "raw_material",
                    },
                    {
                        "facility_name": "Acme Palm Oil Mill",
                        "facility_type": "raw_material",
                    },
                ],
                search_queries=4,
            )
        return GeminiResult(
            facilities_raw=[{"facility_name": "Ghana Cocoa Grower Cooperative"}],
            search_queries=3,
        )

    monkeypatch.setattr(client, "_call", fake_call)
    result = await client.research("Cadbury", "Chocolate")

    assert len(calls) == 2
    assert "farm" in calls[1].lower()
    assert "cooperative" in calls[1].lower()
    assert "Cocoa Life" in calls[1]
    assert "OpenSupplyHub" not in calls[1]
    assert result.facilities_raw[-1]["facility_name"] == "Ghana Cocoa Grower Cooperative"


@pytest.mark.asyncio
async def test_no_retry_when_upstream_present_among_enough_facilities(monkeypatch):
    client = _client(min_facilities_before_retry=3)
    calls: list[str] = []

    async def fake_call(user_message: str) -> GeminiResult:
        calls.append(user_message)
        return GeminiResult(
            facilities_raw=[
                {"facility_name": "Rotterdam Refinery"},
                {"facility_name": "Kalimantan Palm Plantation"},
                {"facility_name": "Wichita Plant"},
            ],
            search_queries=4,
        )

    monkeypatch.setattr(client, "_call", fake_call)
    result = await client.research("Cargill", "palm oil")

    assert len(calls) == 1
    assert len(result.facilities_raw) == 3


@pytest.mark.asyncio
async def test_no_retry_when_cooperative_present(monkeypatch):
    client = _client(min_facilities_before_retry=3)
    calls: list[str] = []

    async def fake_call(user_message: str) -> GeminiResult:
        calls.append(user_message)
        return GeminiResult(
            facilities_raw=[
                {"facility_name": "Bournville Factory"},
                {"facility_name": "Ashanti Cocoa Cooperative"},
                {"facility_name": "Coolock Factory"},
            ],
            search_queries=4,
        )

    monkeypatch.setattr(client, "_call", fake_call)
    result = await client.research("Cadbury", "Chocolate")

    assert len(calls) == 1
    assert len(result.facilities_raw) == 3


@pytest.mark.asyncio
async def test_low_yield_with_upstream_uses_broaden_angle(monkeypatch):
    """Low yield but already has origin upstream → broaden third-party sources."""
    client = _client(min_facilities_before_retry=3)
    calls: list[str] = []

    async def fake_call(user_message: str) -> GeminiResult:
        calls.append(user_message)
        if len(calls) == 1:
            return GeminiResult(
                facilities_raw=[{"facility_name": "Acme Palm Plantation"}],
                search_queries=2,
            )
        return GeminiResult(
            facilities_raw=[{"facility_name": "Contract Co-packer"}],
            search_queries=2,
        )

    monkeypatch.setattr(client, "_call", fake_call)
    result = await client.research("Acme Corp", "palm oil")

    assert len(calls) == 2
    assert "OpenSupplyHub" in calls[1]
    assert "Cocoa Life" not in calls[1]
    assert [f["facility_name"] for f in result.facilities_raw] == [
        "Acme Palm Plantation",
        "Contract Co-packer",
    ]


def test_has_upstream_sites_origin_only():
    assert not _has_upstream_sites(
        [
            {"facility_name": "Palm Oil Refinery", "facility_type": "raw_material"},
            {"facility_name": "Refining Hub", "facility_address": "Rotterdam"},
        ]
    )
    assert not _has_upstream_sites(
        [{"facility_name": "Hindoli Palm Oil Mill", "facility_type": "raw_material"}]
    )
    assert not _has_upstream_sites(
        [
            {
                "facility_name": "Mondelez India Foods Ltd. Cocoa Operations & Nursery",
                "facility_type": "raw_material",
            }
        ]
    )
    assert not _has_upstream_sites(
        [
            {
                "facility_name": "Cadbury Nigeria Plc Cocoa Processing Plant",
                "facility_type": "raw_material",
            }
        ]
    )
    assert _has_upstream_sites(
        [{"facility_name": "Site A", "facility_address": "Near Sungai Lilin Estate"}]
    )
    assert _has_upstream_sites(
        [{"facility_name": "Kalimantan Palm Plantation"}]
    )
    assert _has_upstream_sites(
        [{"facility_name": "Ashanti Cocoa Grower Cooperative"}]
    )
