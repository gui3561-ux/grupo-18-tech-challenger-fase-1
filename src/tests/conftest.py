import pytest
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

# Mockar o lifespan ANTES de importar create_app
# para evitar carregamento do modelo durante import

_mock_predictor = MagicMock()
_mock_predictor.predict.return_value = MagicMock()

@pytest.fixture
def app_model_loaded():
    app = FastAPI()
    app.state.model_loaded = True
    app.state.predictor = _mock_predictor
    return app

@pytest.fixture
def app_model_degraded():
    app = FastAPI()
    app.state.model_loaded = False
    app.state.predictor = None
    return app

@pytest.fixture
def client_loaded(app_model_loaded):
    with TestClient(app_model_loaded) as client:
        yield client

@pytest.fixture
def client_degraded(app_model_degraded):
    with TestClient(app_model_degraded) as client:
        yield client