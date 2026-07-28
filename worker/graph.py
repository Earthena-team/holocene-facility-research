"""Explicit resumable pipeline state machine:
(DISCOVER_PRODUCT →) RESEARCH → EXTRACT → GEOCODE → PERSIST.

DISCOVER_PRODUCT only runs when the job was submitted without a product."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from shared.config import Settings, get_settings
from shared.models import Facility, JobStatus, PipelineStep
from shared.normalize import make_normalized_key, make_normalized_name
from shared.redis_queue import JobQueue
from worker.extract import extract_facilities
from worker.gemini import GeminiClient
from worker.geocode import GeocodeClient
from shared.persist import PersistClient

logger = logging.getLogger("frp.graph")

STEP_BACKOFFS = (2.0, 8.0)


class HaltedError(Exception):
    """Raised when a halt flag is observed before a pipeline step."""


class BudgetPausedError(Exception):
    """Raised when the daily budget circuit breaker trips."""


@dataclass
class JobState:
    job_id: str
    supplier_name: str
    product: str
    normalized_key: str  # (supplier, product) — per-run dedupe / cache lookup
    normalized_name: str = ""  # supplier identity (company), independent of product
    facilities: list[Facility] = field(default_factory=list)
    facilities_raw: list[dict[str, Any]] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    search_queries: int = 0
    geocode_calls: int = 0
    est_cost_usd: float = 0.0
    supplier_id: str | None = None
    served_from_cache: bool = False
    started_at: str | None = None
    discovered_products: list[str] = field(default_factory=list)


class PipelineGraph:
    def __init__(
        self,
        queue: JobQueue,
        gemini: GeminiClient,
        geocode: GeocodeClient,
        persist: PersistClient,
        settings: Settings | None = None,
        worker_id: str = "worker",
    ) -> None:
        self.queue = queue
        self.gemini = gemini
        self.geocode = geocode
        self.persist = persist
        self.settings = settings or get_settings()
        self.worker_id = worker_id

    async def run(self, job_id: str) -> JobState:
        data = await self.queue.get_job(job_id)
        if not data:
            raise RuntimeError(f"job {job_id} not found")

        started_at = data.get("started_at") or datetime.now(timezone.utc).isoformat()
        await self.queue.update_job(
            job_id,
            status=JobStatus.running.value,
            started_at=started_at,
            worker_id=self.worker_id,
            heartbeat_at=datetime.now(timezone.utc).isoformat(),
        )

        state = JobState(
            job_id=job_id,
            supplier_name=data["supplier_name"],
            product=data["product"],
            normalized_key=data.get("normalized_key")
            or make_normalized_key(data["supplier_name"], data["product"]),
            normalized_name=make_normalized_name(data["supplier_name"]),
            started_at=started_at,
            input_tokens=int(data.get("input_tokens") or 0),
            output_tokens=int(data.get("output_tokens") or 0),
            thinking_tokens=int(data.get("thinking_tokens") or 0),
            search_queries=int(data.get("search_queries") or 0),
            geocode_calls=int(data.get("geocode_calls") or 0),
            est_cost_usd=float(data.get("est_cost_usd") or 0),
        )
        if data.get("facilities_json"):
            try:
                raw = json.loads(data["facilities_json"])
                state.facilities = [Facility.model_validate(x) for x in raw]
            except Exception:
                logger.exception("failed to restore facilities_json")
        if data.get("discovered_products"):
            try:
                state.discovered_products = json.loads(data["discovered_products"])
            except Exception:
                logger.exception("failed to restore discovered_products")

        needs_product = not state.product.strip()

        # Cost-saving: serve from Supabase cache inside refresh window. Skipped while
        # the product is still unknown — DISCOVER_PRODUCT re-checks this once it
        # resolves state.product/normalized_key (see _step_discover_product).
        if not needs_product:
            await self._check_halt(job_id)
            cached = self.persist.find_cached_supplier(state.normalized_name, state.product)
            if cached is not None:
                await self._serve_from_cache(state, cached, started_at)
                return state

        steps: list[str] = []
        if needs_product:
            steps.append(PipelineStep.discover_product.value)
        steps += [
            PipelineStep.research.value,
            PipelineStep.extract.value,
            PipelineStep.geocode.value,
            PipelineStep.persist.value,
        ]
        resume_step = data.get("step") or steps[0]
        try:
            start_idx = steps.index(resume_step)
        except ValueError:
            start_idx = 0

        # If we already have facilities from a prior partial run, skip earlier steps
        # that produced them — but always respect resume_step from Redis.
        for step in steps[start_idx:]:
            await self._check_halt(job_id)
            await self.queue.update_job(job_id, step=step)
            await self._run_step_with_retry(step, state)
            if step == PipelineStep.discover_product.value and state.served_from_cache:
                return state

        await self.queue.mark_terminal(
            job_id,
            JobStatus.succeeded.value,
            facilities_found=len(state.facilities),
            extra={
                "est_cost_usd": state.est_cost_usd,
                "input_tokens": state.input_tokens,
                "output_tokens": state.output_tokens,
                "thinking_tokens": state.thinking_tokens,
                "search_queries": state.search_queries,
                "geocode_calls": state.geocode_calls,
            },
        )
        return state

    async def _serve_from_cache(
        self,
        state: JobState,
        cached: tuple[dict[str, Any], list[dict[str, Any]]],
        started_at: str,
    ) -> None:
        """Populate state from a Supabase cache hit and mark the job terminal."""
        supplier, facilities = cached
        state.supplier_id = supplier["id"]
        state.served_from_cache = True
        state.facilities = [
            Facility(
                facility_name=f["facility_name"],
                facility_address=f["facility_address"],
                # Older cached rows predate the product column — fall back to the
                # job's product (identical, since the supplier row is product-scoped).
                product=f.get("product") or state.product,
                latitude=f.get("latitude"),
                longitude=f.get("longitude"),
                facility_type=f["facility_type"],
                confidence=f.get("confidence") or "medium",
                source_url=f.get("source_url"),
                geocode_status=f.get("geocode_status") or "ok",
            )
            for f in facilities
        ]
        await self.queue.mark_terminal(
            state.job_id,
            JobStatus.succeeded.value,
            facilities_found=len(state.facilities),
            extra={"served_from_cache": "true", "note": "served_from_cache"},
        )
        self.persist.insert_research_run(
            job_id=state.job_id,
            supplier_id=state.supplier_id,
            product=state.product,
            status=JobStatus.succeeded.value,
            input_tokens=state.input_tokens,
            output_tokens=state.output_tokens,
            thinking_tokens=state.thinking_tokens,
            search_queries=state.search_queries,
            geocode_calls=state.geocode_calls,
            est_cost_usd=state.est_cost_usd,
            error=None,
            started_at=started_at,
            finished_at=datetime.now(timezone.utc).isoformat(),
        )

    async def _check_halt(self, job_id: str) -> None:
        if await self.queue.is_halted(job_id):
            raise HaltedError(job_id)

    async def _run_step_with_retry(self, step: str, state: JobState) -> None:
        last_exc: Exception | None = None
        for attempt in range(3):  # initial + 2 retries
            try:
                if step == PipelineStep.discover_product.value:
                    await self._step_discover_product(state)
                elif step == PipelineStep.research.value:
                    await self._step_research(state)
                elif step == PipelineStep.extract.value:
                    await self._step_extract(state)
                elif step == PipelineStep.geocode.value:
                    await self._step_geocode(state)
                elif step == PipelineStep.persist.value:
                    await self._step_persist(state)
                return
            except (HaltedError, BudgetPausedError):
                raise
            except Exception as exc:
                last_exc = exc
                logger.exception("step %s failed attempt %s for %s", step, attempt, state.job_id)
                if attempt < 2:
                    await asyncio.sleep(STEP_BACKOFFS[attempt])
        assert last_exc is not None
        raise last_exc

    async def _step_discover_product(self, state: JobState) -> None:
        """Resolve state.product when the job was submitted without one — a single
        small grounded call (see GeminiClient.discover_products), same budget/cost
        accounting path as RESEARCH."""
        budget = await self.queue.get_budget()
        if budget >= self.settings.daily_budget_usd:
            raise BudgetPausedError(
                f"daily budget {budget} >= {self.settings.daily_budget_usd}"
            )

        result = await self.gemini.discover_products(state.supplier_name)
        state.input_tokens += result.input_tokens
        state.output_tokens += result.output_tokens
        state.thinking_tokens += result.thinking_tokens
        state.search_queries += result.search_queries

        cost = self.queue.estimate_cost(
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            thinking_tokens=result.thinking_tokens,
            search_queries=result.search_queries,
        )
        state.est_cost_usd = round(state.est_cost_usd + cost, 5)
        await self.queue.add_budget(cost)
        await self.queue.incr_metric("tokens_in", result.input_tokens)
        await self.queue.incr_metric("tokens_out", result.output_tokens + result.thinking_tokens)
        await self.queue.incr_metric("searches", result.search_queries)

        state.discovered_products = result.products
        state.product = result.products[0] if result.products else "general operations"
        state.normalized_key = make_normalized_key(state.supplier_name, state.product)

        await self.queue.update_job(
            state.job_id,
            product=state.product,
            normalized_key=state.normalized_key,
            discovered_products=json.dumps(state.discovered_products),
            input_tokens=state.input_tokens,
            output_tokens=state.output_tokens,
            thinking_tokens=state.thinking_tokens,
            search_queries=state.search_queries,
            est_cost_usd=state.est_cost_usd,
        )
        # Claim the resolved (supplier, product) key so later explicit-product
        # submissions for the same pair dedupe against this job too.
        await self.queue.set_dedupe(state.normalized_key, state.job_id)

        # Re-check the Supabase cache now that the real product is known — avoids
        # paying for RESEARCH if this exact (supplier, discovered product) was
        # already researched under a prior explicit-product job.
        cached = self.persist.find_cached_supplier(state.normalized_name, state.product)
        if cached is not None:
            await self._serve_from_cache(state, cached, state.started_at or "")

    async def _step_research(self, state: JobState) -> None:
        budget = await self.queue.get_budget()
        if budget >= self.settings.daily_budget_usd:
            raise BudgetPausedError(
                f"daily budget {budget} >= {self.settings.daily_budget_usd}"
            )

        result = await self.gemini.research(state.supplier_name, state.product)
        state.facilities_raw = result.facilities_raw
        state.input_tokens += result.input_tokens
        state.output_tokens += result.output_tokens
        state.thinking_tokens += result.thinking_tokens
        state.search_queries += result.search_queries

        cost = self.queue.estimate_cost(
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            thinking_tokens=result.thinking_tokens,
            search_queries=result.search_queries,
        )
        state.est_cost_usd = round(state.est_cost_usd + cost, 5)
        await self.queue.add_budget(cost)
        await self.queue.incr_metric("tokens_in", result.input_tokens)
        await self.queue.incr_metric("tokens_out", result.output_tokens + result.thinking_tokens)
        await self.queue.incr_metric("searches", result.search_queries)

        await self.queue.update_job(
            state.job_id,
            input_tokens=state.input_tokens,
            output_tokens=state.output_tokens,
            thinking_tokens=state.thinking_tokens,
            search_queries=state.search_queries,
            est_cost_usd=state.est_cost_usd,
            facilities_json=json.dumps(state.facilities_raw),
        )

    async def _step_extract(self, state: JobState) -> None:
        raw = state.facilities_raw
        if not raw:
            # Try restoring from job hash
            data = await self.queue.get_job(state.job_id)
            if data and data.get("facilities_json"):
                try:
                    parsed = json.loads(data["facilities_json"])
                    if parsed and isinstance(parsed[0], dict) and "facility_type" in parsed[0]:
                        # Could be raw or already Facility dicts
                        raw = parsed
                except Exception:
                    pass
        state.facilities = extract_facilities(
            raw, max_items=self.settings.max_facilities, product=state.product
        )
        await self.queue.update_job(
            state.job_id,
            facilities_json=json.dumps([f.model_dump(mode="json") for f in state.facilities]),
            facilities_found=len(state.facilities),
        )

    async def _step_geocode(self, state: JobState) -> None:
        if not state.facilities:
            data = await self.queue.get_job(state.job_id)
            if data and data.get("facilities_json"):
                try:
                    state.facilities = [
                        Facility.model_validate(x) for x in json.loads(data["facilities_json"])
                    ]
                except Exception:
                    logger.exception("failed to load facilities for geocode")

        before_calls = self.geocode.geocode_calls
        state.facilities = await self.geocode.geocode_facilities(state.facilities)
        new_calls = self.geocode.geocode_calls - before_calls
        state.geocode_calls += new_calls

        geo_cost = self.queue.estimate_cost(geocode_calls=new_calls)
        state.est_cost_usd = round(state.est_cost_usd + geo_cost, 5)
        if geo_cost:
            await self.queue.add_budget(geo_cost)

        await self.queue.update_job(
            state.job_id,
            geocode_calls=state.geocode_calls,
            est_cost_usd=state.est_cost_usd,
            facilities_json=json.dumps([f.model_dump(mode="json") for f in state.facilities]),
        )

    async def _step_persist(self, state: JobState) -> None:
        supplier = self.persist.upsert_supplier(
            name=state.supplier_name,
            normalized_name=state.normalized_name,
        )
        state.supplier_id = supplier["id"]
        self.persist.upsert_facilities(state.supplier_id, state.facilities)

        data = await self.queue.get_job(state.job_id)
        self.persist.insert_research_run(
            job_id=state.job_id,
            supplier_id=state.supplier_id,
            product=state.product,
            status=JobStatus.succeeded.value,
            input_tokens=state.input_tokens,
            output_tokens=state.output_tokens,
            thinking_tokens=state.thinking_tokens,
            search_queries=state.search_queries,
            geocode_calls=state.geocode_calls,
            est_cost_usd=state.est_cost_usd,
            error=None,
            started_at=state.started_at or (data or {}).get("started_at"),
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
