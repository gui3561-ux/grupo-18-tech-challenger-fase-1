import time
from collections.abc import Awaitable, Callable

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from src.metrics import http_request_duration_seconds, http_requests_total

logger = structlog.get_logger(__name__)


class LatencyLoggerMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        start_time = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

        http_requests_total.labels(
            method=request.method,
            endpoint=request.url.path,
            status_code=str(response.status_code),
        ).inc()
        http_request_duration_seconds.labels(
            method=request.method,
            endpoint=request.url.path,
        ).observe(elapsed_ms / 1000)

        logger.info(
            "http_request",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            latency_ms=elapsed_ms,
        )

        return response
