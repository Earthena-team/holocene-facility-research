from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse


class ProblemDetail(Exception):
    def __init__(
        self,
        *,
        status: int,
        title: str,
        detail: str,
        type_: str = "about:blank",
        extra: dict[str, Any] | None = None,
    ) -> None:
        self.status = status
        self.title = title
        self.detail = detail
        self.type_ = type_
        self.extra = extra or {}
        super().__init__(detail)


async def problem_exception_handler(request: Request, exc: ProblemDetail) -> JSONResponse:
    body: dict[str, Any] = {
        "type": exc.type_,
        "title": exc.title,
        "status": exc.status,
        "detail": exc.detail,
        "instance": str(request.url.path),
    }
    body.update(exc.extra)
    return JSONResponse(
        status_code=exc.status,
        content=body,
        media_type="application/problem+json",
    )
