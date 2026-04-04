import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import numpy as np

from src.api.v1.router import api_router
from src.schemas.inference import ChurnResponse


def _create_test_app(model_loaded: bool) -> FastAPI:
    app = FastAPI()
    app.state.model_loaded = model_loaded
    app.include_router(api_router)
    return app


def _sample_churn_payload() -> dict:
    return {
        "tenure_months": 2,
        "monthly_charges": 70.70,
        "total_charges": 151.65,
        "gender": "Female",
        "senior_citizen": "No",
        "partner": "No",
        "dependents": "No",
        "state": "CA",
        "phone_service": "Yes",
        "multiple_lines": "No",
        "internet_service": "Fiber optic",
        "online_security": "No",
        "online_backup": "No",
        "device_protection": "No",
        "tech_support": "No",
        "streaming_tv": "No",
        "streaming_movies": "No",
        "contract": "Month-to-month",
        "paperless_billing": "Yes",
        "payment_method": "Electronic check",
    }


@pytest.fixture
def sample_payload() -> dict:
    return _sample_churn_payload()


@pytest.fixture
def app_model_loaded():
    return _create_test_app(model_loaded=True)


@pytest.fixture
def app_model_degraded():
    return _create_test_app(model_loaded=False)


@pytest.fixture
def client_loaded(app_model_loaded):
    with TestClient(app_model_loaded) as client:
        yield client


@pytest.fixture
def client_degraded(app_model_degraded):
    with TestClient(app_model_degraded) as client:
        yield client


@pytest.fixture
def mock_inference_service():
    """Mock do ChurnInferenceService para evitar carregar o modelo real."""
    mock_service = MagicMock()
    mock_service.predict.return_value = ChurnResponse(
        churn_probability=0.7523,
        churn_prediction=True,
        model="neural_network",
    )
    return mock_service


@pytest.fixture
def client_with_mock_model(mock_inference_service):
    """Client com modelo mockado para testes de inferência."""
    with patch(
        "src.routers.inference._service", mock_inference_service
    ):
        app = _create_test_app(model_loaded=True)
        with TestClient(app) as client:
            yield client
