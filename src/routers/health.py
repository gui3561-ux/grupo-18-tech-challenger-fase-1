from fastapi import APIRouter

from src.schemas.health import HealthResponse

router = APIRouter(tags=["Health"])

API_VERSION = "1.0.0"


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Healthcheck",
    description="Verifica se a API está operacional.",
)
def healthcheck() -> HealthResponse:
    return HealthResponse(status="ok", version=API_VERSION)
