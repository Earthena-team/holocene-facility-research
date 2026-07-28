"""Control API — FastAPI service. Jobs via Redis; catalog reads via Supabase."""

from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.errors import ProblemDetail, problem_exception_handler
from api.routes import router
from shared.config import get_settings
from shared.persist import PersistClient
from shared.redis_queue import JobQueue

logger = logging.getLogger("frp.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    queue = await JobQueue.create(settings)
    persist = PersistClient(settings)
    app.state.queue = queue
    app.state.persist = persist
    app.state.settings = settings
    yield
    await queue.close()


app = FastAPI(
    title="Facility Research Control API",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_exception_handler(ProblemDetail, problem_exception_handler)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "type": "about:blank",
            "title": "Validation Error",
            "status": 422,
            "detail": str(exc.errors()),
            "instance": str(request.url.path),
        },
        media_type="application/problem+json",
    )


@app.middleware("http")
async def structured_logging(request: Request, call_next):
    start = time.perf_counter()
    response: Response = await call_next(request)
    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    job_id = request.path_params.get("job_id")
    log = {
        "method": request.method,
        "path": request.url.path,
        "status": response.status_code,
        "latency_ms": latency_ms,
    }
    if job_id:
        log["job_id"] = job_id
    logger.info(json.dumps(log))
    return response


app.include_router(router)


def create_app() -> FastAPI:
    return app
