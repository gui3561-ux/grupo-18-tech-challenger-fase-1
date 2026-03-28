import time

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, Request, Response

logger = structlog.get_logger(__name__)


class LatencyLoggerMiddleware(BaseHTTPMiddleware):

    async def dispatch(self, request: Request, call_next) -> Response:
        start_time = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        response.headers["X-Process-Time"] = str(elapsed_ms)
        logger.info(
            "http_request",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            latency_ms=elapsed_ms
        )

        return response