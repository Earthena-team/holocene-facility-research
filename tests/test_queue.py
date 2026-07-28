"""Unit tests for JobQueue: enqueue/dequeue/ack, priority, reaper, dead-letter."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from shared.models import JobPriority, JobStatus
from shared.redis_queue import QUEUE_PROCESSING, JobQueue


@pytest.mark.asyncio
async def test_enqueue_dequeue_ack(queue: JobQueue):
    result = await queue.enqueue("Acme Corp", "widgets")
    assert result["status"] == "created"
    job_id = result["job_id"]

    dequeued = await queue.dequeue(timeout=1.0)
    assert dequeued == job_id

    job = await queue.get_job(job_id)
    assert job is not None
    assert job["status"] == JobStatus.queued.value

    await queue.ack(job_id)
    processing = await queue.r.lrange(QUEUE_PROCESSING, 0, -1)
    assert job_id not in processing


@pytest.mark.asyncio
async def test_priority_ordering(queue: JobQueue):
    normal = await queue.enqueue("Normal Co", "bolts", priority=JobPriority.normal)
    high = await queue.enqueue("High Co", "screws", priority=JobPriority.high)

    first = await queue.dequeue(timeout=1.0)
    second = await queue.dequeue(timeout=1.0)
    assert first == high["job_id"]
    assert second == normal["job_id"]


@pytest.mark.asyncio
async def test_reaper_requeues_stale(queue: JobQueue, settings):
    result = await queue.enqueue("Stale Co", "nails")
    job_id = result["job_id"]
    # Simulate in-flight with stale heartbeat
    await queue.r.lpush(QUEUE_PROCESSING, job_id)
    # Remove from pending
    await queue.r.lrem("frp:queue:pending:normal", 0, job_id)
    stale = (datetime.now(timezone.utc) - timedelta(minutes=15)).isoformat()
    await queue.update_job(
        job_id,
        status=JobStatus.running.value,
        heartbeat_at=stale,
        attempts=0,
        started_at=stale,
    )

    acted = await queue.reap_stale()
    assert job_id in acted
    job = await queue.get_job(job_id)
    assert job["status"] == JobStatus.queued.value
    assert int(job["attempts"]) == 1


@pytest.mark.asyncio
async def test_reaper_dead_letters_after_max_attempts(queue: JobQueue):
    result = await queue.enqueue("Dead Co", "rivets")
    job_id = result["job_id"]
    await queue.r.lpush(QUEUE_PROCESSING, job_id)
    await queue.r.lrem("frp:queue:pending:normal", 0, job_id)
    stale = (datetime.now(timezone.utc) - timedelta(minutes=15)).isoformat()
    await queue.update_job(
        job_id,
        status=JobStatus.running.value,
        heartbeat_at=stale,
        attempts=2,  # +1 in reaper → 3 == max
        started_at=stale,
    )

    acted = await queue.reap_stale()
    assert job_id in acted
    job = await queue.get_job(job_id)
    assert job["status"] == JobStatus.dead.value
    dead = await queue.r.lrange("frp:queue:dead", 0, -1)
    assert job_id in dead
