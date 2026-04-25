from fastapi import APIRouter, Request

from src.core.config import settings
from src.rate_limit import limiter
from src.schemas.inference import ChurnRequest, ChurnResponse
from src.services.inference_service import ChurnInferenceService

router = APIRouter(prefix="/inference", tags=["Inference"])

_service = ChurnInferenceService()


@router.post("/predict", response_model=ChurnResponse)
@limiter.limit(settings.rate_limit_predict)
def predict_churn(request: Request, body: ChurnRequest) -> ChurnResponse:
    return _service.predict(body)
