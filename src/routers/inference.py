from fastapi import APIRouter
from src.schemas.inference import ChurnRequest, ChurnResponse
from src.services.inference_service import ChurnInferenceService

router = APIRouter(prefix="/inference", tags=["Inference"])

_service = ChurnInferenceService()


@router.post("/predict", response_model=ChurnResponse)
def predict_churn(request: ChurnRequest) -> ChurnResponse:
    return _service.predict(request)