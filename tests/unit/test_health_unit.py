import asyncio
from unittest.mock import MagicMock

from starlette.requests import Request as StarletteRequest

from src.core.config import settings
from src.routers.health import health_check


def build_mock_request(model_loaded: bool) -> StarletteRequest:
    mock_app = MagicMock()
    mock_app.state.model_loaded = model_loaded

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/health",
        "query_string": b"",
        "headers": [],
        "app": mock_app,
    }
    return StarletteRequest(scope)


class TestHealthUnit:
    def test_status_ok_when_model_is_loaded(self):
        request = build_mock_request(model_loaded=True)
        response = asyncio.run(health_check(request))

        assert response.status == "ok"
        assert response.model_loaded is True
        assert response.version == settings.api_version

    def test_status_degraded_when_model_is_not_loaded(self):
        request = build_mock_request(model_loaded=False)
        response = asyncio.run(health_check(request))

        assert response.status == "degraded"
        assert response.model_loaded is False

    def test_status_degraded_when_state_attribute_is_missing(self):
        mock_app = MagicMock()
        mock_app.state = MagicMock(spec=[])

        scope = {
            "type": "http",
            "method": "GET",
            "path": "/health",
            "query_string": b"",
            "headers": [],
            "app": mock_app,
        }
        request = StarletteRequest(scope)
        response = asyncio.run(health_check(request))

        assert response.status == "degraded"
        assert response.model_loaded is False
