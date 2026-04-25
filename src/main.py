from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from src.api.v1.router import api_router
from src.core.config import settings
from src.core.logging import configure_logging
from src.metrics import model_loaded
from src.middleware import LatencyLoggerMiddleware
from src.rate_limit import limiter, rate_limit_exceeded_handler
from src.services.inference_service import ChurnInferenceService, ModelNotLoadedError

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging(settings.log_level)

    try:
        predictor = ChurnInferenceService(settings.model_path)
        app.state.predictor = predictor
        app.state.model_loaded = True
        model_loaded.set(1)
        logger.info("startup_complete", model_path=settings.model_path)
    except ModelNotLoadedError as exc:
        logger.info("startup_model_not_found", error=str(exc))
        app.state.predictor = None
        app.state.model_loaded = False
        model_loaded.set(0)

    yield

    logger.info("shutdown")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.api_title,
        description=settings.api_description,
        version=settings.api_version,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)
    app.add_middleware(LatencyLoggerMiddleware)
    app.include_router(api_router)

    return app


app = create_app()
