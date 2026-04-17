from fastapi import APIRouter, Request

from src.core.config import settings
from src.schemas.health import HealthResponse

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Healthcheck",
    description="Verifica se a API está operacional.",
)
async def health_check(request: Request) -> HealthResponse:
    model_loaded: bool = getattr(request.app.state, "model_loaded", False)
    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        model_loaded=model_loaded,
        version=settings.api_version,
    )
