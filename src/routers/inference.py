from fastapi import APIRouter, Depends

from src.schemas.inference import InferenceResponse
from src.services.inference_service import HelloWorldInferenceService, InferenceServiceInterface

router = APIRouter(prefix="/inference", tags=["Inference"])


def get_inference_service() -> InferenceServiceInterface:
    """Factory de dependência — troque a implementação aqui quando o modelo ML estiver pronto."""
    return HelloWorldInferenceService()


@router.get(
    "/hello",
    response_model=InferenceResponse,
    summary="Hello World",
    description="Endpoint de exemplo. Será substituído pela inferência real do modelo de ML.",
)
def hello_world(
    service: InferenceServiceInterface = Depends(get_inference_service),
) -> InferenceResponse:
    return service.predict()
