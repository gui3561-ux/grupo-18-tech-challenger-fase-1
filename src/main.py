from contextlib import asynccontextmanager
from fastapi import FastAPI
from src.api.v1.router import api_router
from src.core.config import settings
from src.core.logging import configure_logging
from src.middleware import LatencyLoggerMiddleware
from src.services.inference_service import ChurnInferenceService
from src.services.inference_service import ModelNotLoadedError
from pathlib import Path
import structlog

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(settings.log_level)

    try:
        predictor = ChurnInferenceService(settings.model_path)
        app.state.predictor = predictor
        app.state.model_loaded = True
        logger.info("startup_complete", model_path=settings.model_path)
    except ModelNotLoadedError as exc:
        logger.info("startup_model_not_found", error=str(exc))
        app.state.predictor = None
        app.state.model_loaded = False

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

    app.add_middleware(LatencyLoggerMiddleware)
    app.include_router(api_router)

    return app


app = create_app()
