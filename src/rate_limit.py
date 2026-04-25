import structlog
from fastapi import Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

logger = structlog.get_logger(__name__)

limiter = Limiter(key_func=get_remote_address)


async def rate_limit_exceeded_handler(request: Request, exc: Exception) -> JSONResponse:
    rate_exc = exc if isinstance(exc, RateLimitExceeded) else None
    client_ip = request.client.host if request.client else "unknown"
    logger.warning(
        "rate_limit_exceeded",
        path=request.url.path,
        method=request.method,
        client_ip=client_ip,
        limit=str(rate_exc.detail) if rate_exc else "unknown",
    )
    retry_after = getattr(rate_exc, "retry_after", 60) if rate_exc else 60
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit excedido. Tente novamente mais tarde."},
        headers={"Retry-After": str(retry_after)},
    )
