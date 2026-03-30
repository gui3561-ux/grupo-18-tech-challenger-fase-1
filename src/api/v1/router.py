from fastapi import APIRouter

from src.routers import health, inference, metrics

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(health.router)
api_router.include_router(inference.router)
api_router.include_router(metrics.router)
