"""Google Geocoding API with Redis cache."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
from typing import Any

import httpx

from shared.config import Settings, get_settings
from shared.models import Facility
from shared.normalize import normalize_address
from shared.redis_queue import JobQueue

logger = logging.getLogger("frp.geocode")

GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"


def address_hash(address: str) -> str:
    return hashlib.sha1(normalize_address(address).encode("utf-8")).hexdigest()


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def map_location_type(location_type: str | None) -> str:
    if location_type in {"ROOFTOP", "RANGE_INTERPOLATED"}:
        return "ok"
    if location_type in {"GEOMETRIC_CENTER", "APPROXIMATE"}:
        return "approximate"
    return "failed"


class GeocodeClient:
    def __init__(
        self,
        queue: JobQueue,
        settings: Settings | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.queue = queue
        self.settings = settings or get_settings()
        self._http = http_client
        self._owns_http = http_client is None
        self.geocode_calls = 0

    async def _client(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=30.0)
        return self._http

    async def close(self) -> None:
        if self._owns_http and self._http is not None:
            await self._http.aclose()

    async def geocode_facilities(self, facilities: list[Facility]) -> list[Facility]:
        sem = asyncio.Semaphore(self.settings.geocode_concurrency)
        results = await asyncio.gather(
            *[self._geocode_one(f, sem) for f in facilities]
        )
        return list(results)

    async def _geocode_one(self, facility: Facility, sem: asyncio.Semaphore) -> Facility:
        # Skip if already has coords and we still verify against geocoder when needed
        needs_geocode = facility.latitude is None or facility.longitude is None
        async with sem:
            result = await self._lookup(facility.facility_address)

        if result is None:
            return facility.model_copy(
                update={
                    "latitude": None if needs_geocode else facility.latitude,
                    "longitude": None if needs_geocode else facility.longitude,
                    "geocode_status": "failed" if needs_geocode else facility.geocode_status,
                }
            )

        g_lat = result["lat"]
        g_lng = result["lng"]
        status = map_location_type(result.get("location_type"))

        if facility.latitude is not None and facility.longitude is not None:
            dist = haversine_km(facility.latitude, facility.longitude, g_lat, g_lng)
            if dist > self.settings.geocode_distance_km_threshold:
                # Trust the geocoder over model-provided coords
                return facility.model_copy(
                    update={
                        "latitude": g_lat,
                        "longitude": g_lng,
                        "geocode_status": status,
                    }
                )
            # Model coords are close enough; keep them but mark geocode ok/approx
            return facility.model_copy(update={"geocode_status": status})

        return facility.model_copy(
            update={
                "latitude": g_lat,
                "longitude": g_lng,
                "geocode_status": status,
            }
        )

    async def _lookup(self, address: str) -> dict[str, Any] | None:
        h = address_hash(address)
        cached = await self.queue.get_geocode_cache(h)
        if cached is not None:
            if cached.get("failed"):
                return None
            return cached

        client = await self._client()
        data = await self._call_api(client, address)
        self.geocode_calls += 1
        await self.queue.incr_metric("geocodes", 1)

        if data is None:
            await self.queue.set_geocode_cache(h, {"failed": True})
            return None

        await self.queue.set_geocode_cache(h, data)
        return data

    async def _call_api(
        self, client: httpx.AsyncClient, address: str, attempt: int = 0
    ) -> dict[str, Any] | None:
        params = {"address": address, "key": self.settings.geocoding_api_key}
        try:
            resp = await client.get(GEOCODE_URL, params=params)
        except httpx.HTTPError:
            logger.exception("geocode HTTP error for %s", address)
            return None

        if resp.status_code == 429:
            if attempt >= 3:
                return None
            await asyncio.sleep(2 ** attempt)
            return await self._call_api(client, address, attempt + 1)

        if resp.status_code != 200:
            logger.warning("geocode status %s for %s", resp.status_code, address)
            return None

        payload = resp.json()
        results = payload.get("results") or []
        if not results:
            return None
        loc = results[0]["geometry"]["location"]
        location_type = results[0]["geometry"].get("location_type")
        return {
            "lat": float(loc["lat"]),
            "lng": float(loc["lng"]),
            "location_type": location_type,
        }
