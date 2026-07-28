"""Supabase persistence — upsert suppliers/facilities and write research_runs."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from supabase import Client, ClientOptions, create_client

from shared.config import Settings, get_settings
from shared.models import Facility

logger = logging.getLogger("frp.persist")

_FACILITY_SELECT = "*,suppliers(name)"


def _flatten_facility(row: dict[str, Any]) -> dict[str, Any]:
    """Lift embedded suppliers(name) into supplier_name."""
    out = dict(row)
    embedded = out.pop("suppliers", None)
    if isinstance(embedded, dict):
        out["supplier_name"] = embedded.get("name")
    elif isinstance(embedded, list) and embedded:
        out["supplier_name"] = embedded[0].get("name")
    else:
        out.setdefault("supplier_name", None)
    return out


class PersistClient:
    def __init__(self, settings: Settings | None = None, client: Client | None = None) -> None:
        self.settings = settings or get_settings()
        if client is not None:
            self.client = client
        else:
            if not self.settings.supabase_url or not self.settings.supabase_service_key:
                raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_KEY are required")
            schema = self.settings.supabase_schema or "public"
            self.client = create_client(
                self.settings.supabase_url,
                self.settings.supabase_service_key,
                options=ClientOptions(schema=schema),
            )
            logger.info("Supabase client using schema=%s", schema)

    def find_cached_supplier(
        self, normalized_name: str, product: str
    ) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
        """Return (supplier, facilities-for-this-product) when a fresh cache exists.

        Supplier identity is the company (normalized_name); freshness is per product,
        judged by the most recent successful research_run for (supplier, product) —
        `suppliers.created_at` is no longer a valid signal now that one supplier row
        is reused across many products and many runs.
        """
        resp = (
            self.client.table("suppliers")
            .select("*")
            .eq("normalized_name", normalized_name)
            .limit(1)
            .execute()
        )
        rows = resp.data or []
        if not rows:
            return None
        supplier = rows[0]

        fac_resp = (
            self.client.table("facilities")
            .select("*")
            .eq("supplier_id", supplier["id"])
            .eq("product", product)
            .execute()
        )
        facilities = fac_resp.data or []
        if not facilities:
            return None

        # Per-(supplier, product) freshness from the latest successful run.
        run_resp = (
            self.client.table("research_runs")
            .select("finished_at")
            .eq("supplier_id", supplier["id"])
            .eq("product", product)
            .eq("status", "succeeded")
            .order("finished_at", desc=True)
            .limit(1)
            .execute()
        )
        runs = run_resp.data or []
        if runs and runs[0].get("finished_at"):
            try:
                finished = datetime.fromisoformat(runs[0]["finished_at"].replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                finished = datetime.now(timezone.utc)
            cutoff = datetime.now(timezone.utc) - timedelta(days=self.settings.refresh_days)
            if finished < cutoff:
                return None

        return supplier, facilities

    def upsert_supplier(self, *, name: str, normalized_name: str) -> dict[str, Any]:
        """Upsert one row per company (keyed on normalized_name), independent of product."""
        existing = (
            self.client.table("suppliers")
            .select("*")
            .eq("normalized_name", normalized_name)
            .limit(1)
            .execute()
        )
        if existing.data:
            return existing.data[0]

        resp = (
            self.client.table("suppliers")
            .insert({"name": name, "normalized_name": normalized_name})
            .execute()
        )
        return resp.data[0]

    def upsert_facilities(self, supplier_id: str, facilities: list[Facility]) -> int:
        if not facilities:
            return 0
        rows = [
            {
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
            for f in facilities
        ]
        self.client.table("facilities").upsert(
            rows,
            on_conflict="supplier_id,product,facility_name,facility_address",
        ).execute()
        return len(rows)

    def insert_research_run(
        self,
        *,
        job_id: str,
        supplier_id: str | None,
        product: str,
        status: str,
        input_tokens: int,
        output_tokens: int,
        thinking_tokens: int,
        search_queries: int,
        geocode_calls: int,
        est_cost_usd: float,
        error: str | None,
        started_at: str | None,
        finished_at: str | None,
    ) -> None:
        row = {
            "id": job_id,
            "supplier_id": supplier_id,
            "product": product,
            "status": status,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "thinking_tokens": thinking_tokens,
            "search_queries": search_queries,
            "geocode_calls": geocode_calls,
            "est_cost_usd": est_cost_usd,
            "error": error,
            "started_at": started_at,
            "finished_at": finished_at,
        }
        self.client.table("research_runs").upsert(row, on_conflict="id").execute()

    def get_supplier(self, supplier_id: str) -> dict[str, Any] | None:
        resp = (
            self.client.table("suppliers")
            .select("*")
            .eq("id", supplier_id)
            .limit(1)
            .execute()
        )
        rows = resp.data or []
        return rows[0] if rows else None

    def list_suppliers(
        self,
        *,
        q: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        query = self.client.table("suppliers").select("*")
        if q:
            query = query.ilike("name", f"%{q}%")
        resp = (
            query.order("created_at", desc=True)
            .range(offset, offset + limit - 1)
            .execute()
        )
        return resp.data or []

    def get_facility(self, facility_id: str) -> dict[str, Any] | None:
        resp = (
            self.client.table("facilities")
            .select(_FACILITY_SELECT)
            .eq("id", facility_id)
            .limit(1)
            .execute()
        )
        rows = resp.data or []
        return _flatten_facility(rows[0]) if rows else None

    def list_facilities(
        self,
        *,
        supplier_id: str | None = None,
        product: str | None = None,
        facility_type: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        query = self.client.table("facilities").select(_FACILITY_SELECT)
        if supplier_id:
            query = query.eq("supplier_id", supplier_id)
        if product is not None and product != "":
            query = query.eq("product", product)
        if facility_type:
            query = query.eq("facility_type", facility_type)
        resp = (
            query.order("created_at", desc=True)
            .range(offset, offset + limit - 1)
            .execute()
        )
        return [_flatten_facility(r) for r in (resp.data or [])]
