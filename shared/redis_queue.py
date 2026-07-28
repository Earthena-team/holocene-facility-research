"""Async Redis job queue for the facility research pipeline.

Reliable-queue pattern with priority lists, reaper, dedupe, geocode cache,
budget counters, and daily metrics. All keys are namespaced under ``frp:``.
"""

from __future__ import annotations

import json
import uuid
from datetime import date, datetime, timezone
from typing import Any

import redis.asyncio as redis

from shared.config import Settings, get_settings
from shared.models import JobPriority, JobStatus
from shared.normalize import make_normalized_key

PREFIX = "frp"

QUEUE_PENDING_HIGH = f"{PREFIX}:queue:pending:high"
QUEUE_PENDING_NORMAL = f"{PREFIX}:queue:pending:normal"
QUEUE_PROCESSING = f"{PREFIX}:queue:processing"
QUEUE_DEAD = f"{PREFIX}:queue:dead"
INDEX_JOBS = f"{PREFIX}:index:jobs"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _today() -> str:
    return date.today().isoformat()


def job_key(job_id: str) -> str:
    return f"{PREFIX}:job:{job_id}"


def halt_key(job_id: str) -> str:
    return f"{PREFIX}:halt:{job_id}"


def dedupe_key(normalized_key: str) -> str:
    return f"{PREFIX}:dedupe:{normalized_key}"


def geocode_key(address_hash: str) -> str:
    return f"{PREFIX}:geocode:{address_hash}"


def budget_key(day: str | None = None) -> str:
    return f"{PREFIX}:budget:{day or _today()}"


def metrics_key(metric: str, day: str | None = None) -> str:
    return f"{PREFIX}:metrics:{day or _today()}:{metric}"


class JobQueue:
    def __init__(self, client: redis.Redis, settings: Settings | None = None) -> None:
        self.r = client
        self.settings = settings or get_settings()

    @classmethod
    async def create(cls, settings: Settings | None = None) -> JobQueue:
        settings = settings or get_settings()
        # socket_timeout must exceed BLMOVE block time or empty-queue waits raise
        # TimeoutError instead of returning None.
        client = redis.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_timeout=30,
            socket_connect_timeout=5,
        )
        return cls(client, settings)

    async def ping(self) -> bool:
        return bool(await self.r.ping())

    async def close(self) -> None:
        await self.r.aclose()

    # ------------------------------------------------------------------ enqueue
    async def enqueue(
        self,
        supplier_name: str,
        product: str,
        *,
        priority: JobPriority | str = JobPriority.normal,
        force: bool = False,
        job_id: str | None = None,
    ) -> dict[str, Any]:
        """Enqueue a job. Returns either a new job or a duplicate marker.

        `product` may be blank — the worker then runs DISCOVER_PRODUCT first. Dedupe
        uses a placeholder key in that case, so repeat no-product submissions for the
        same supplier collapse onto one discovery job instead of paying for N of them.
        """
        nkey = make_normalized_key(supplier_name, product or "__auto__")
        if not force:
            existing = await self.r.get(dedupe_key(nkey))
            if existing:
                return {"status": "duplicate", "existing_job_id": existing}

        job_id = job_id or str(uuid.uuid4())
        now = _utcnow()
        priority_val = (
            priority.value if isinstance(priority, JobPriority) else str(priority)
        )
        fields: dict[str, str] = {
            "id": job_id,
            "supplier_name": supplier_name,
            "product": product,
            "normalized_key": nkey,
            "status": JobStatus.queued.value,
            "step": "",
            "attempts": "0",
            "priority": priority_val,
            "enqueued_at": now,
            "started_at": "",
            "finished_at": "",
            "input_tokens": "0",
            "output_tokens": "0",
            "thinking_tokens": "0",
            "search_queries": "0",
            "geocode_calls": "0",
            "est_cost_usd": "0",
            "facilities_found": "0",
            "error": "",
            "worker_id": "",
            "heartbeat_at": "",
            "served_from_cache": "false",
            "note": "",
            "facilities_json": "",
            "discovered_products": "",
        }
        pipe = self.r.pipeline()
        pipe.hset(job_key(job_id), mapping=fields)
        queue = (
            QUEUE_PENDING_HIGH
            if priority_val == JobPriority.high.value
            else QUEUE_PENDING_NORMAL
        )
        pipe.lpush(queue, job_id)
        score = datetime.now(timezone.utc).timestamp()
        pipe.zadd(INDEX_JOBS, {job_id: score})
        pipe.set(
            dedupe_key(nkey),
            job_id,
            ex=self.settings.dedupe_ttl_days * 24 * 3600,
        )
        await pipe.execute()
        return {"status": "created", "job_id": job_id}

    async def set_dedupe(self, normalized_key: str, job_id: str) -> None:
        """Claim the dedupe key for a (supplier, product) pair.

        Used by DISCOVER_PRODUCT to register the *resolved* key once the product is
        known, so later explicit-product submissions for the same pair dedupe too.
        """
        await self.r.set(
            dedupe_key(normalized_key),
            job_id,
            ex=self.settings.dedupe_ttl_days * 24 * 3600,
        )

    # ------------------------------------------------------------------ dequeue / ack
    async def dequeue(self, timeout: float = 5.0) -> str | None:
        """BLMOVE from high (short timeout) then normal. Returns job_id or None."""
        high_timeout = min(0.1, timeout)
        try:
            result = await self.r.blmove(
                QUEUE_PENDING_HIGH,
                QUEUE_PROCESSING,
                timeout=high_timeout,
                src="RIGHT",
                dest="LEFT",
            )
        except redis.TimeoutError:
            result = None
        if result is not None:
            return result

        remaining = max(0.0, timeout - high_timeout)
        if remaining <= 0:
            return None
        try:
            result = await self.r.blmove(
                QUEUE_PENDING_NORMAL,
                QUEUE_PROCESSING,
                timeout=remaining,
                src="RIGHT",
                dest="LEFT",
            )
        except redis.TimeoutError:
            return None
        return result

    async def ack(self, job_id: str) -> None:
        await self.r.lrem(QUEUE_PROCESSING, 0, job_id)

    async def requeue(
        self,
        job_id: str,
        *,
        status: str = JobStatus.queued.value,
        note: str | None = None,
        increment_attempts: bool = False,
    ) -> None:
        """Remove from processing and push back to the appropriate pending list."""
        data = await self.get_job(job_id)
        priority = (data or {}).get("priority", JobPriority.normal.value)
        queue = (
            QUEUE_PENDING_HIGH
            if priority == JobPriority.high.value
            else QUEUE_PENDING_NORMAL
        )
        updates: dict[str, str] = {"status": status, "step": "", "worker_id": ""}
        if note is not None:
            updates["note"] = note
        if increment_attempts and data:
            updates["attempts"] = str(int(data.get("attempts") or 0) + 1)

        pipe = self.r.pipeline()
        pipe.lrem(QUEUE_PROCESSING, 0, job_id)
        # Also remove from pending in case of double-presence
        pipe.lrem(QUEUE_PENDING_HIGH, 0, job_id)
        pipe.lrem(QUEUE_PENDING_NORMAL, 0, job_id)
        pipe.hset(job_key(job_id), mapping=updates)
        pipe.lpush(queue, job_id)
        await pipe.execute()

    async def move_to_dead(self, job_id: str, error: str | None = None) -> None:
        updates: dict[str, str] = {
            "status": JobStatus.dead.value,
            "finished_at": _utcnow(),
        }
        if error:
            updates["error"] = error
        pipe = self.r.pipeline()
        pipe.lrem(QUEUE_PROCESSING, 0, job_id)
        pipe.lrem(QUEUE_PENDING_HIGH, 0, job_id)
        pipe.lrem(QUEUE_PENDING_NORMAL, 0, job_id)
        pipe.lpush(QUEUE_DEAD, job_id)
        pipe.hset(job_key(job_id), mapping=updates)
        pipe.expire(job_key(job_id), self.settings.job_ttl_seconds)
        await pipe.execute()

    # ------------------------------------------------------------------ job state
    async def get_job(self, job_id: str) -> dict[str, str] | None:
        data = await self.r.hgetall(job_key(job_id))
        return data or None

    async def update_job(self, job_id: str, **fields: Any) -> None:
        mapping = {k: "" if v is None else str(v) for k, v in fields.items()}
        if mapping:
            await self.r.hset(job_key(job_id), mapping=mapping)

    async def mark_terminal(
        self,
        job_id: str,
        status: str,
        *,
        error: str | None = None,
        facilities_found: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        mapping: dict[str, str] = {
            "status": status,
            "finished_at": _utcnow(),
        }
        if error is not None:
            mapping["error"] = error
        if facilities_found is not None:
            mapping["facilities_found"] = str(facilities_found)
        if extra:
            mapping.update({k: "" if v is None else str(v) for k, v in extra.items()})
        await self.r.hset(job_key(job_id), mapping=mapping)
        await self.r.expire(job_key(job_id), self.settings.job_ttl_seconds)

    async def heartbeat(self, job_id: str, worker_id: str) -> None:
        await self.r.hset(
            job_key(job_id),
            mapping={"heartbeat_at": _utcnow(), "worker_id": worker_id},
        )

    async def set_halt(self, job_id: str) -> None:
        await self.r.set(halt_key(job_id), "1", ex=self.settings.halt_ttl_seconds)

    async def is_halted(self, job_id: str) -> bool:
        return bool(await self.r.get(halt_key(job_id)))

    async def clear_halt(self, job_id: str) -> None:
        await self.r.delete(halt_key(job_id))

    async def delete_queued(self, job_id: str) -> bool:
        """Delete a queued job. Returns False if not queued."""
        data = await self.get_job(job_id)
        if not data or data.get("status") != JobStatus.queued.value:
            return False
        pipe = self.r.pipeline()
        pipe.lrem(QUEUE_PENDING_HIGH, 0, job_id)
        pipe.lrem(QUEUE_PENDING_NORMAL, 0, job_id)
        pipe.hset(
            job_key(job_id),
            mapping={"status": JobStatus.deleted.value, "finished_at": _utcnow()},
        )
        pipe.expire(job_key(job_id), self.settings.job_ttl_seconds)
        await pipe.execute()
        return True

    async def retry_job(self, job_id: str, *, reset_attempts: bool = False) -> bool:
        data = await self.get_job(job_id)
        if not data:
            return False
        status = data.get("status")
        if status not in {
            JobStatus.failed.value,
            JobStatus.dead.value,
            JobStatus.halted.value,
        }:
            return False
        await self.clear_halt(job_id)
        updates: dict[str, str] = {
            "status": JobStatus.queued.value,
            "step": "",
            "error": "",
            "finished_at": "",
            "note": "",
            "worker_id": "",
        }
        if reset_attempts:
            updates["attempts"] = "0"
        priority = data.get("priority", JobPriority.normal.value)
        queue = (
            QUEUE_PENDING_HIGH
            if priority == JobPriority.high.value
            else QUEUE_PENDING_NORMAL
        )
        pipe = self.r.pipeline()
        pipe.lrem(QUEUE_DEAD, 0, job_id)
        pipe.lrem(QUEUE_PROCESSING, 0, job_id)
        pipe.hset(job_key(job_id), mapping=updates)
        pipe.lpush(queue, job_id)
        await pipe.execute()
        return True

    # ------------------------------------------------------------------ listing
    async def list_jobs(
        self,
        *,
        status: str | None = None,
        limit: int = 50,
        cursor: float | None = None,
    ) -> tuple[list[dict[str, str]], float | None]:
        """Paginate jobs from the ZSET index by enqueued_at score (newest first)."""
        max_score = cursor if cursor is not None else float("+inf")
        # Fetch a window; filter by status in Python (index is small / operational).
        ids = await self.r.zrevrangebyscore(
            INDEX_JOBS,
            max=max_score,
            min="-inf",
            start=0,
            num=limit + 20,
            withscores=True,
        )
        results: list[dict[str, str]] = []
        next_cursor: float | None = None
        for jid, score in ids:
            if cursor is not None and score >= cursor:
                # Exclusive upper bound for pagination
                continue
            job = await self.get_job(jid)
            if not job:
                continue
            if status and job.get("status") != status:
                continue
            results.append(job)
            next_cursor = score
            if len(results) >= limit:
                break
        if len(results) < limit:
            next_cursor = None
        return results, next_cursor

    # ------------------------------------------------------------------ reaper
    async def reap_stale(self) -> list[str]:
        """Re-queue or dead-letter jobs in processing with stale heartbeats."""
        processing = await self.r.lrange(QUEUE_PROCESSING, 0, -1)
        now = datetime.now(timezone.utc)
        acted: list[str] = []
        for job_id in processing:
            data = await self.get_job(job_id)
            if not data:
                await self.r.lrem(QUEUE_PROCESSING, 0, job_id)
                continue
            hb = data.get("heartbeat_at") or data.get("started_at") or ""
            if not hb:
                # Just moved into processing; give it a grace window via started_at check
                started = data.get("started_at") or data.get("enqueued_at") or ""
                if not started:
                    continue
                hb = started
            try:
                hb_dt = datetime.fromisoformat(hb)
                if hb_dt.tzinfo is None:
                    hb_dt = hb_dt.replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            age = (now - hb_dt).total_seconds()
            if age < self.settings.heartbeat_stale_seconds:
                continue
            attempts = int(data.get("attempts") or 0) + 1
            if attempts >= self.settings.max_attempts:
                await self.move_to_dead(job_id, error="reaper: max attempts exceeded")
            else:
                await self.requeue(
                    job_id,
                    status=JobStatus.queued.value,
                    note="reaper: requeued after stale heartbeat",
                    increment_attempts=True,
                )
            acted.append(job_id)
        return acted

    # ------------------------------------------------------------------ geocode cache
    async def get_geocode_cache(self, address_hash: str) -> dict[str, Any] | None:
        raw = await self.r.get(geocode_key(address_hash))
        if not raw:
            return None
        return json.loads(raw)

    async def set_geocode_cache(self, address_hash: str, payload: dict[str, Any]) -> None:
        await self.r.set(
            geocode_key(address_hash),
            json.dumps(payload),
            ex=self.settings.geocode_ttl_seconds,
        )

    # ------------------------------------------------------------------ budget / metrics
    async def get_budget(self, day: str | None = None) -> float:
        raw = await self.r.get(budget_key(day))
        return float(raw) if raw else 0.0

    async def add_budget(self, amount: float, day: str | None = None) -> float:
        key = budget_key(day)
        pipe = self.r.pipeline()
        pipe.incrbyfloat(key, amount)
        pipe.expire(key, self.settings.budget_ttl_seconds)
        results = await pipe.execute()
        return float(results[0])

    async def incr_metric(self, metric: str, amount: int | float = 1) -> None:
        key = metrics_key(metric)
        pipe = self.r.pipeline()
        if isinstance(amount, float):
            pipe.incrbyfloat(key, amount)
        else:
            pipe.incrby(key, int(amount))
        pipe.expire(key, self.settings.metrics_ttl_seconds)
        await pipe.execute()

    async def get_metric(self, metric: str, day: str | None = None) -> int:
        raw = await self.r.get(metrics_key(metric, day))
        return int(float(raw)) if raw else 0

    async def get_stats(self) -> dict[str, Any]:
        pipe = self.r.pipeline()
        pipe.llen(QUEUE_PENDING_HIGH)
        pipe.llen(QUEUE_PENDING_NORMAL)
        pipe.llen(QUEUE_PROCESSING)
        pipe.llen(QUEUE_DEAD)
        pipe.get(budget_key())
        pipe.get(metrics_key("tokens_in"))
        pipe.get(metrics_key("tokens_out"))
        pipe.get(metrics_key("searches"))
        pipe.get(metrics_key("geocodes"))
        results = await pipe.execute()

        # Jobs by status in last 24h from ZSET
        cutoff = datetime.now(timezone.utc).timestamp() - 24 * 3600
        recent_ids = await self.r.zrangebyscore(INDEX_JOBS, min=cutoff, max="+inf")
        by_status: dict[str, int] = {}
        for jid in recent_ids:
            job = await self.get_job(jid)
            if not job:
                continue
            st = job.get("status", "unknown")
            by_status[st] = by_status.get(st, 0) + 1

        budget = float(results[4]) if results[4] else 0.0
        return {
            "pending_high": int(results[0]),
            "pending_normal": int(results[1]),
            "processing": int(results[2]),
            "dead": int(results[3]),
            "jobs_by_status_24h": by_status,
            "budget_usd_today": budget,
            "tokens_in_today": int(float(results[5])) if results[5] else 0,
            "tokens_out_today": int(float(results[6])) if results[6] else 0,
            "searches_today": int(float(results[7])) if results[7] else 0,
            "geocodes_today": int(float(results[8])) if results[8] else 0,
            "budget_paused": budget >= self.settings.daily_budget_usd,
        }

    def estimate_cost(
        self,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        thinking_tokens: int = 0,
        search_queries: int = 0,
        geocode_calls: int = 0,
    ) -> float:
        s = self.settings
        cost = 0.0
        cost += (input_tokens / 1_000_000) * s.price_in_per_m
        # thinking tokens billed as output
        cost += ((output_tokens + thinking_tokens) / 1_000_000) * s.price_out_per_m
        cost += (search_queries / 1000) * s.price_per_1k_search
        cost += (geocode_calls / 1000) * s.price_per_1k_geocode
        return round(cost, 5)
