"""Worker main loop: dequeue → run graph → ack, with reaper, heartbeats, SIGTERM, /healthz."""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
import uuid
from datetime import datetime, timezone
from typing import Any

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from shared.config import get_settings
from shared.models import JobStatus
from shared.redis_queue import JobQueue
from worker.gemini import GeminiClient
from worker.geocode import GeocodeClient
from worker.graph import BudgetPausedError, HaltedError, PipelineGraph
from shared.persist import PersistClient

logger = logging.getLogger("frp.worker")

# Track last loop activity for Cloud Run health checks
_last_heartbeat = datetime.now(timezone.utc)
_shutdown = asyncio.Event()


def _touch_heartbeat() -> None:
    global _last_heartbeat
    _last_heartbeat = datetime.now(timezone.utc)


async def healthz_handler(_request: Request) -> JSONResponse:
    age = (datetime.now(timezone.utc) - _last_heartbeat).total_seconds()
    if age > 60:
        return JSONResponse({"status": "stale", "age_seconds": age}, status_code=503)
    return JSONResponse({"status": "ok", "age_seconds": age})


async def start_health_server(port: int) -> uvicorn.Server:
    app = Starlette(routes=[Route("/healthz", healthz_handler)])
    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="warning")
    server = uvicorn.Server(config)
    asyncio.create_task(server.serve())
    logger.info("health server listening on :%s", port)
    return server


async def process_job(
    queue: JobQueue,
    graph: PipelineGraph,
    job_id: str,
    settings,
) -> None:
    hb_task: asyncio.Task | None = None

    async def _heartbeat_loop() -> None:
        while True:
            await queue.heartbeat(job_id, graph.worker_id)
            _touch_heartbeat()
            await asyncio.sleep(settings.heartbeat_interval_seconds)

    try:
        hb_task = asyncio.create_task(_heartbeat_loop())
        await graph.run(job_id)
        await queue.ack(job_id)
    except HaltedError:
        logger.info("job %s halted", job_id)
        await queue.mark_terminal(job_id, JobStatus.halted.value, error="halted by operator")
        await queue.ack(job_id)
    except BudgetPausedError as exc:
        logger.warning("budget paused for job %s: %s", job_id, exc)
        await queue.requeue(
            job_id,
            status=JobStatus.budget_paused.value,
            note="budget_paused",
        )
        await asyncio.sleep(15 * 60)
    except asyncio.CancelledError:
        # SIGTERM path: re-enqueue current job
        logger.info("job %s cancelled — re-enqueueing", job_id)
        await queue.requeue(
            job_id,
            status=JobStatus.queued.value,
            note="worker shutdown requeue",
        )
        raise
    except Exception as exc:
        logger.exception("job %s failed", job_id)
        data = await queue.get_job(job_id)
        attempts = int((data or {}).get("attempts") or 0) + 1
        await queue.update_job(job_id, attempts=attempts, error=str(exc))
        if attempts >= settings.max_attempts:
            await queue.move_to_dead(job_id, error=str(exc))
        else:
            await queue.requeue(
                job_id,
                status=JobStatus.queued.value,
                note=f"retry after error: {exc}",
            )
    finally:
        if hb_task:
            hb_task.cancel()
            try:
                await hb_task
            except asyncio.CancelledError:
                pass


async def worker_loop(
    queue: JobQueue,
    graph: PipelineGraph,
    settings,
    slot: int,
) -> None:
    logger.info("worker slot %s starting", slot)
    while not _shutdown.is_set():
        _touch_heartbeat()
        try:
            job_id = await queue.dequeue(timeout=5.0)
        except Exception:
            logger.exception("dequeue failed")
            await asyncio.sleep(1)
            continue
        if job_id is None:
            continue
        if _shutdown.is_set():
            await queue.requeue(job_id, status=JobStatus.queued.value, note="shutdown before start")
            break
        logger.info("slot %s picked job %s", slot, job_id)
        await process_job(queue, graph, job_id, settings)


async def reaper_loop(queue: JobQueue) -> None:
    while not _shutdown.is_set():
        try:
            acted = await queue.reap_stale()
            if acted:
                logger.info("reaper acted on %s", acted)
        except Exception:
            logger.exception("reaper failed")
        try:
            await asyncio.wait_for(_shutdown.wait(), timeout=60)
        except asyncio.TimeoutError:
            pass


async def async_main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    settings = get_settings()
    worker_id = f"worker-{uuid.uuid4().hex[:8]}"

    queue = await JobQueue.create(settings)
    gemini = GeminiClient(settings)
    geocode = GeocodeClient(queue, settings)
    persist = PersistClient(settings)
    graph = PipelineGraph(queue, gemini, geocode, persist, settings, worker_id=worker_id)

    health_runner = await start_health_server(settings.worker_health_port)

    loop = asyncio.get_running_loop()

    def _handle_sig(*_args) -> None:
        logger.info("SIGTERM/SIGINT received — shutting down")
        _shutdown.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _handle_sig)
        except NotImplementedError:
            signal.signal(sig, lambda *_: _handle_sig())

    # Reap stale jobs on startup
    acted = await queue.reap_stale()
    if acted:
        logger.info("startup reaper acted on %s", acted)

    tasks = [
        asyncio.create_task(worker_loop(queue, graph, settings, slot=i))
        for i in range(settings.worker_concurrency)
    ]
    tasks.append(asyncio.create_task(reaper_loop(queue)))

    await _shutdown.wait()
    logger.info("waiting for in-flight work to finish/requeue…")
    # Give workers a moment to requeue; then cancel
    await asyncio.sleep(1)
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    await geocode.close()
    await queue.close()
    health_runner.should_exit = True
    logger.info("worker exited cleanly")


def main() -> None:
    try:
        # aiohttp is used for the health listener; fall back note if missing
        asyncio.run(async_main())
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
