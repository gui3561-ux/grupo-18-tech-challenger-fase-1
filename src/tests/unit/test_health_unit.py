import pytest
from unittest.mock import MagicMock
from src.routers.health import health_check
from src.core.config import settings


async def build_mock_request(model_loaded: bool) -> MagicMock:
    mock_request = MagicMock()
    mock_request.app.state.model_loaded = model_loaded
    return mock_request


class TestHealthUnit:

    @pytest.mark.asyncio
    async def test_status_ok_when_model_is_loaded(self):
        request = await build_mock_request(model_loaded=True)
        response = await health_check(request)

        assert response.status == "ok"
        assert response.model_loaded is True
        assert response.version == settings.api_version

    @pytest.mark.asyncio
    async def test_status_degraded_when_model_is_not_loaded(self):
        request = await build_mock_request(model_loaded=False)
        response = await health_check(request)

        assert response.status == "degraded"
        assert response.model_loaded is False

    @pytest.mark.asyncio
    async def test_status_degraded_when_state_attribute_is_missing(self):
        mock_request = MagicMock()
        mock_request.app.state = MagicMock(spec=[])

        response = await health_check(mock_request)

        assert response.status == "degraded"
        assert response.model_loaded is False