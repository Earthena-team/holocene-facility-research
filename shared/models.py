import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class FacilityType(str, Enum):
    manufacturing = "manufacturing"
    logistics = "logistics"
    raw_material = "raw_material"


class Facility(BaseModel):
    facility_name: str = Field(..., min_length=1, max_length=300)
    facility_address: str = Field(..., min_length=1)
    # The product this facility was researched for. Stamped from the job's resolved
    # product (not returned by Gemini) — a job is scoped to one product, so every
    # facility it finds relates to that product.
    product: str = ""
    latitude: float | None = None
    longitude: float | None = None
    facility_type: FacilityType
    confidence: Literal["high", "medium", "low"]
    source_url: str | None = None
    geocode_status: Literal["ok", "approximate", "failed", "pending"] = "pending"

    @field_validator("latitude")
    @classmethod
    def validate_lat(cls, v: float | None) -> float | None:
        if v is None:
            return v
        if not -90.0 <= v <= 90.0:
            return None
        return v

    @field_validator("longitude")
    @classmethod
    def validate_lng(cls, v: float | None) -> float | None:
        if v is None:
            return v
        if not -180.0 <= v <= 180.0:
            return None
        return v


class JobPriority(str, Enum):
    high = "high"
    normal = "normal"


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    halted = "halted"
    dead = "dead"
    deleted = "deleted"
    budget_paused = "budget_paused"


class PipelineStep(str, Enum):
    discover_product = "discover_product"
    research = "research"
    extract = "extract"
    geocode = "geocode"
    persist = "persist"


class CreateJobRequest(BaseModel):
    supplier_name: str = Field(..., min_length=1)
    # Optional: if omitted/blank, the worker runs a DISCOVER_PRODUCT step first to
    # find the supplier's most popular/sourced product before researching facilities.
    product: str = Field(default="")
    priority: JobPriority = JobPriority.normal
    force: bool = False

    @field_validator("product")
    @classmethod
    def strip_product(cls, v: str) -> str:
        return v.strip()


class CreateJobResponse(BaseModel):
    job_id: str


class DuplicateJobResponse(BaseModel):
    status: Literal["duplicate"] = "duplicate"
    existing_job_id: str


class RetryJobRequest(BaseModel):
    reset_attempts: bool = False


class JobView(BaseModel):
    id: str
    supplier_name: str
    product: str
    status: str
    step: str | None = None
    attempts: int = 0
    priority: str = "normal"
    enqueued_at: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    search_queries: int = 0
    geocode_calls: int = 0
    est_cost_usd: float = 0.0
    facilities_found: int = 0
    error: str | None = None
    worker_id: str | None = None
    heartbeat_at: str | None = None
    served_from_cache: bool = False
    elapsed_seconds: float | None = None
    note: str | None = None
    # Populated only when the job ran DISCOVER_PRODUCT (product was omitted at submission);
    # ranked most-to-least significant, `product` above is discovered_products[0].
    discovered_products: list[str] = Field(default_factory=list)


class StatsResponse(BaseModel):
    pending_high: int
    pending_normal: int
    processing: int
    dead: int
    jobs_by_status_24h: dict[str, int]
    budget_usd_today: float
    tokens_in_today: int
    tokens_out_today: int
    searches_today: int
    geocodes_today: int
    budget_paused: bool = False


class SupplierView(BaseModel):
    id: str
    name: str
    normalized_name: str
    created_at: str | None = None


class FacilityView(BaseModel):
    id: str
    supplier_id: str
    supplier_name: str | None = None
    facility_name: str
    facility_address: str
    product: str = ""
    latitude: float | None = None
    longitude: float | None = None
    facility_type: str
    confidence: str
    source_url: str | None = None
    geocode_status: str = "pending"
    created_at: str | None = None


class SupplierListResponse(BaseModel):
    suppliers: list[SupplierView]
    next_offset: int | None = None


class FacilityListResponse(BaseModel):
    facilities: list[FacilityView]
    next_offset: int | None = None


def _json_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return []
    return [str(x) for x in parsed] if isinstance(parsed, list) else []


def job_hash_to_view(data: dict[str, Any]) -> JobView:
    """Convert a Redis job hash (string values) into a JobView with derived fields."""
    enqueued_at = data.get("enqueued_at") or None
    started_at = data.get("started_at") or None
    finished_at = data.get("finished_at") or None

    elapsed: float | None = None
    start_ref = started_at or enqueued_at
    if start_ref:
        try:
            start_dt = datetime.fromisoformat(start_ref)
            end_dt = (
                datetime.fromisoformat(finished_at)
                if finished_at
                else datetime.now(timezone.utc)
            )
            # Job hashes store timezone-aware ISO timestamps; normalize both sides.
            if start_dt.tzinfo is None:
                start_dt = start_dt.replace(tzinfo=timezone.utc)
            if end_dt.tzinfo is None:
                end_dt = end_dt.replace(tzinfo=timezone.utc)
            elapsed = max(0.0, (end_dt - start_dt).total_seconds())
        except (ValueError, TypeError):
            elapsed = None

    def _int(key: str, default: int = 0) -> int:
        raw = data.get(key)
        if raw is None or raw == "":
            return default
        return int(float(raw))

    def _float(key: str, default: float = 0.0) -> float:
        raw = data.get(key)
        if raw is None or raw == "":
            return default
        return float(raw)

    return JobView(
        id=data.get("id", ""),
        supplier_name=data.get("supplier_name", ""),
        product=data.get("product", ""),
        status=data.get("status", ""),
        step=data.get("step") or None,
        attempts=_int("attempts"),
        priority=data.get("priority") or "normal",
        enqueued_at=enqueued_at,
        started_at=started_at,
        finished_at=finished_at,
        input_tokens=_int("input_tokens"),
        output_tokens=_int("output_tokens"),
        thinking_tokens=_int("thinking_tokens"),
        search_queries=_int("search_queries"),
        geocode_calls=_int("geocode_calls"),
        est_cost_usd=_float("est_cost_usd"),
        facilities_found=_int("facilities_found"),
        error=data.get("error") or None,
        worker_id=data.get("worker_id") or None,
        heartbeat_at=data.get("heartbeat_at") or None,
        served_from_cache=data.get("served_from_cache") == "true",
        elapsed_seconds=elapsed,
        note=data.get("note") or None,
        discovered_products=_json_list(data.get("discovered_products")),
    )


# JSON schema for Gemini structured output (array of facilities).
# google.genai Schema rejects JSON Schema union types like ["number","null"];
# use a single type + nullable instead.
# Do not set maxItems here: Gemini 3.5 rejects maxItems on this complex item
# schema with 400 INVALID_ARGUMENT. Cap length in parse/extract instead.
FACILITY_LIST_SCHEMA: dict[str, Any] = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "facility_name": {"type": "string"},
            "facility_address": {"type": "string"},
            "latitude": {"type": "number", "nullable": True},
            "longitude": {"type": "number", "nullable": True},
            "facility_type": {
                "type": "string",
                "enum": ["manufacturing", "logistics", "raw_material"],
            },
            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            "source_url": {"type": "string", "nullable": True},
        },
        "required": [
            "facility_name",
            "facility_address",
            "facility_type",
            "confidence",
        ],
    },
}

# JSON schema for Gemini structured output in the DISCOVER_PRODUCT step:
# a short, ranked list of product names — deliberately just strings, no object
# wrapper, to keep this call's output tokens (and cost) minimal.
PRODUCT_LIST_SCHEMA: dict[str, Any] = {
    "type": "array",
    "items": {"type": "string"},
}
