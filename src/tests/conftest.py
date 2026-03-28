import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock
from src.main import create_app

@pytest.fixture
def app_model_loaded():
    app = create_app()

    async def mock_lifespan(app):
        app.state.model_loaded = True
        app.state.predictor = MagicMock()
        yield

    app.router.lifespan_context = mock_lifespan
    return app

@pytest.fixture
def app_model_degraded():
    app = create_app()

    async def mock_lifespan(app):
        app.state.model_loaded = False
        app.state.predictor = None
        yield

    app.router.lifespan_context = mock_lifespan
    return app

@pytest.fixture
def client_loaded(app_model_loaded):
    with TestClient(app_model_loaded) as client:
        yield client


@pytest.fixture
def client_degraded(app_model_degraded):
    with TestClient(app_model_degraded) as client:
        yield client