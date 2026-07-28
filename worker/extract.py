"""Validate and clean facility records — no LLM."""

from __future__ import annotations

import logging
from typing import Any

from rapidfuzz import fuzz

from shared.models import Facility, FacilityType

logger = logging.getLogger("frp.extract")

TYPE_SYNONYMS: dict[str, FacilityType] = {
    "manufacturing": FacilityType.manufacturing,
    "factory": FacilityType.manufacturing,
    "plant": FacilityType.manufacturing,
    "production": FacilityType.manufacturing,
    "assembly": FacilityType.manufacturing,
    "fabrication": FacilityType.manufacturing,
    "contract manufactur": FacilityType.manufacturing,
    "toll manufactur": FacilityType.manufacturing,
    "oem": FacilityType.manufacturing,
    "logistics": FacilityType.logistics,
    "warehouse": FacilityType.logistics,
    "dc": FacilityType.logistics,
    "distribution": FacilityType.logistics,
    "distribution center": FacilityType.logistics,
    "fulfillment": FacilityType.logistics,
    "port": FacilityType.logistics,
    "terminal": FacilityType.logistics,
    "freight": FacilityType.logistics,
    "hub": FacilityType.logistics,
    "depot": FacilityType.logistics,
    "cross-dock": FacilityType.logistics,
    "cross dock": FacilityType.logistics,
    "storage": FacilityType.logistics,
    "yard": FacilityType.logistics,
    "3pl": FacilityType.logistics,
    "third-party logistics": FacilityType.logistics,
    "raw_material": FacilityType.raw_material,
    "raw material": FacilityType.raw_material,
    "mine": FacilityType.raw_material,
    "quarry": FacilityType.raw_material,
    "well": FacilityType.raw_material,
    "extraction": FacilityType.raw_material,
    "smelter": FacilityType.raw_material,
    "refinery": FacilityType.raw_material,
    "mill": FacilityType.raw_material,
    "tannery": FacilityType.raw_material,
    "farm": FacilityType.raw_material,
    "plantation": FacilityType.raw_material,
    "primary processing": FacilityType.raw_material,
}


def _map_facility_type(raw: Any) -> FacilityType | None:
    if isinstance(raw, FacilityType):
        return raw
    if not isinstance(raw, str):
        return None
    key = raw.strip().lower()
    if key in TYPE_SYNONYMS:
        return TYPE_SYNONYMS[key]
    for synonym, mapped in TYPE_SYNONYMS.items():
        if synonym in key:
            return mapped
    return None


def _clamp_coords(
    lat: Any, lng: Any
) -> tuple[float | None, float | None]:
    try:
        lat_f = float(lat) if lat is not None else None
    except (TypeError, ValueError):
        lat_f = None
    try:
        lng_f = float(lng) if lng is not None else None
    except (TypeError, ValueError):
        lng_f = None
    if lat_f is not None and not -90.0 <= lat_f <= 90.0:
        lat_f = None
    if lng_f is not None and not -180.0 <= lng_f <= 180.0:
        lng_f = None
    return lat_f, lng_f


def extract_facilities(
    raw_items: list[dict[str, Any]], *, max_items: int = 40, product: str = ""
) -> list[Facility]:
    cleaned: list[Facility] = []
    for item in raw_items[:max_items]:
        name = (item.get("facility_name") or "").strip()
        address = (item.get("facility_address") or "").strip()
        if not name or not address:
            continue

        ftype = _map_facility_type(item.get("facility_type"))
        if ftype is None:
            logger.info("dropping facility with unknown type: %s / %s", name, item.get("facility_type"))
            continue

        confidence = item.get("confidence", "low")
        if confidence not in {"high", "medium", "low"}:
            confidence = "low"

        lat, lng = _clamp_coords(item.get("latitude"), item.get("longitude"))
        source_url = item.get("source_url")
        if source_url is not None:
            source_url = str(source_url).strip() or None

        try:
            facility = Facility(
                facility_name=name[:300],
                facility_address=address,
                product=product,
                latitude=lat,
                longitude=lng,
                facility_type=ftype,
                confidence=confidence,
                source_url=source_url,
                geocode_status="pending",
            )
        except Exception:
            logger.exception("skipping invalid facility %s", name)
            continue
        cleaned.append(facility)

    # Exact dedupe on (lower name, lower address)
    seen: set[tuple[str, str]] = set()
    unique: list[Facility] = []
    for f in cleaned:
        key = (f.facility_name.lower(), f.facility_address.lower())
        if key in seen:
            continue
        seen.add(key)
        unique.append(f)

    # Near-duplicate collapse on address similarity ≥ 0.9
    final: list[Facility] = []
    for f in unique:
        is_dup = False
        for kept in final:
            if fuzz.ratio(f.facility_address.lower(), kept.facility_address.lower()) >= 90:
                # Prefer the one with higher confidence / coordinates
                is_dup = True
                break
        if not is_dup:
            final.append(f)

    return final[:max_items]
