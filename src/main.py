from fastapi import FastAPI
import logging
import sys
from src.api.v1.router import api_router
from src.core.config import settings
from src.core.logging import configure_logging




def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.api_title,
        description=settings.api_description,
        version=settings.api_version,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.include_router(api_router)

    return app


app = create_app()
