import asyncio

from fastapi import APIRouter, Query, Request, Response

from api.errors import ProblemDetail
from shared.models import (
    CreateJobRequest,
    CreateJobResponse,
    DuplicateJobResponse,
    FacilityListResponse,
    FacilityType,
    FacilityView,
    RetryJobRequest,
    StatsResponse,
    SupplierListResponse,
    SupplierView,
    job_hash_to_view,
)
from shared.persist import PersistClient
from shared.redis_queue import JobQueue

router = APIRouter()


def _queue(request: Request) -> JobQueue:
    return request.app.state.queue


def _persist(request: Request) -> PersistClient:
    return request.app.state.persist


def _next_offset(offset: int, limit: int, n: int) -> int | None:
    return offset + n if n == limit else None


@router.get("/healthz")
async def healthz(request: Request) -> dict[str, str]:
    queue = _queue(request)
    ok = await queue.ping()
    if not ok:
        raise ProblemDetail(status=503, title="Unavailable", detail="Redis PING failed")
    return {"status": "ok"}


@router.post("/jobs", status_code=201)
async def create_job(body: CreateJobRequest, request: Request, response: Response):
    queue = _queue(request)
    result = await queue.enqueue(
        body.supplier_name,
        body.product,
        priority=body.priority,
        force=body.force,
    )
    if result.get("status") == "duplicate":
        response.status_code = 200
        return DuplicateJobResponse(existing_job_id=result["existing_job_id"])
    response.status_code = 201
    return CreateJobResponse(job_id=result["job_id"])


@router.get("/jobs/{job_id}")
async def get_job(job_id: str, request: Request):
    queue = _queue(request)
    data = await queue.get_job(job_id)
    if not data:
        raise ProblemDetail(status=404, title="Not Found", detail=f"Job {job_id} not found")
    return job_hash_to_view(data)


@router.get("/jobs")
async def list_jobs(
    request: Request,
    status: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    cursor: float | None = None,
):
    queue = _queue(request)
    jobs, next_cursor = await queue.list_jobs(status=status, limit=limit, cursor=cursor)
    return {
        "jobs": [job_hash_to_view(j) for j in jobs],
        "next_cursor": next_cursor,
    }


@router.delete("/jobs/{job_id}", status_code=204)
async def delete_job(job_id: str, request: Request):
    queue = _queue(request)
    data = await queue.get_job(job_id)
    if not data:
        raise ProblemDetail(status=404, title="Not Found", detail=f"Job {job_id} not found")
    ok = await queue.delete_queued(job_id)
    if not ok:
        raise ProblemDetail(
            status=409,
            title="Conflict",
            detail="Only queued jobs can be deleted; halt a running job instead",
        )
    return Response(status_code=204)


@router.post("/jobs/{job_id}/halt", status_code=202)
async def halt_job(job_id: str, request: Request):
    queue = _queue(request)
    data = await queue.get_job(job_id)
    if not data:
        raise ProblemDetail(status=404, title="Not Found", detail=f"Job {job_id} not found")
    await queue.set_halt(job_id)
    return {"status": "halt_requested", "job_id": job_id}


@router.post("/jobs/{job_id}/retry")
async def retry_job(job_id: str, request: Request, body: RetryJobRequest | None = None):
    queue = _queue(request)
    data = await queue.get_job(job_id)
    if not data:
        raise ProblemDetail(status=404, title="Not Found", detail=f"Job {job_id} not found")
    reset = body.reset_attempts if body else False
    ok = await queue.retry_job(job_id, reset_attempts=reset)
    if not ok:
        raise ProblemDetail(
            status=409,
            title="Conflict",
            detail="Retry only allowed for failed, dead, or halted jobs",
        )
    return {"status": "queued", "job_id": job_id}


@router.get("/stats", response_model=StatsResponse)
async def stats(request: Request):
    queue = _queue(request)
    return StatsResponse(**await queue.get_stats())


@router.get("/suppliers", response_model=SupplierListResponse)
async def list_suppliers(
    request: Request,
    q: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    persist = _persist(request)
    rows = await asyncio.to_thread(
        persist.list_suppliers, q=q, limit=limit, offset=offset
    )
    suppliers = [SupplierView.model_validate(r) for r in rows]
    return SupplierListResponse(
        suppliers=suppliers,
        next_offset=_next_offset(offset, limit, len(suppliers)),
    )


@router.get("/suppliers/{supplier_id}", response_model=SupplierView)
async def get_supplier(supplier_id: str, request: Request):
    persist = _persist(request)
    row = await asyncio.to_thread(persist.get_supplier, supplier_id)
    if not row:
        raise ProblemDetail(
            status=404, title="Not Found", detail=f"Supplier {supplier_id} not found"
        )
    return SupplierView.model_validate(row)


@router.get("/facilities", response_model=FacilityListResponse)
async def list_facilities(
    request: Request,
    supplier_id: str | None = None,
    product: str | None = None,
    facility_type: FacilityType | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    persist = _persist(request)
    rows = await asyncio.to_thread(
        persist.list_facilities,
        supplier_id=supplier_id,
        product=product,
        facility_type=facility_type.value if facility_type else None,
        limit=limit,
        offset=offset,
    )
    facilities = [FacilityView.model_validate(r) for r in rows]
    return FacilityListResponse(
        facilities=facilities,
        next_offset=_next_offset(offset, limit, len(facilities)),
    )


@router.get("/facilities/{facility_id}", response_model=FacilityView)
async def get_facility(facility_id: str, request: Request):
    persist = _persist(request)
    row = await asyncio.to_thread(persist.get_facility, facility_id)
    if not row:
        raise ProblemDetail(
            status=404, title="Not Found", detail=f"Facility {facility_id} not found"
        )
    return FacilityView.model_validate(row)
