"""Router para expor métricas no formato Prometheus."""

from fastapi import APIRouter, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from src.metrics import registry

router = APIRouter(prefix="/metrics", tags=["Monitoring"])


@router.get("/", include_in_schema=False)
async def metrics():
    """Endpoint para expor métricas no formato Prometheus.

    Retorna todas as métricas coletadas no formato texto do Prometheus.
    Coleta automatizada: Azure Monitor Managed Prometheus ou scraping direto.
    """
    return Response(
        content=generate_latest(registry),
        media_type=CONTENT_TYPE_LATEST,
    )


@router.get("/health", include_in_schema=False)
async def metrics_health():
    """Health check para o endpoint de métricas."""
    return {"status": "ok", "metrics_endpoint": "/metrics"}
