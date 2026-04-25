from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from src.rate_limit import rate_limit_exceeded_handler


def _make_rate_limited_app(limit: str) -> TestClient:
    test_limiter = Limiter(key_func=get_remote_address)

    app = FastAPI()
    app.state.limiter = test_limiter
    app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)

    @app.get("/probe")
    @test_limiter.limit(limit)
    async def probe(request: Request):
        return {"ok": True}

    return TestClient(app)


class TestRateLimitBehavior:
    def test_requests_within_limit_succeed(self):
        client = _make_rate_limited_app("3/minute")
        for _ in range(3):
            assert client.get("/probe").status_code == 200

    def test_requests_exceeding_limit_returns_429(self):
        client = _make_rate_limited_app("2/minute")
        client.get("/probe")
        client.get("/probe")
        response = client.get("/probe")
        assert response.status_code == 429

    def test_rate_limit_response_is_json(self):
        client = _make_rate_limited_app("1/minute")
        client.get("/probe")
        response = client.get("/probe")
        assert response.headers["content-type"].startswith("application/json")

    def test_rate_limit_response_has_detail_field(self):
        client = _make_rate_limited_app("1/minute")
        client.get("/probe")
        response = client.get("/probe")
        assert "detail" in response.json()

    def test_rate_limit_response_includes_retry_after_header(self):
        client = _make_rate_limited_app("1/minute")
        client.get("/probe")
        response = client.get("/probe")
        assert "retry-after" in response.headers


class TestProductionEndpointsHaveRateLimits:
    def test_health_endpoint_is_wrapped_by_rate_limit(self):
        from src.routers.health import health_check

        assert hasattr(health_check, "__wrapped__"), (
            "/health deve ter @limiter.limit aplicado"
        )

    def test_predict_endpoint_is_wrapped_by_rate_limit(self):
        from src.routers.inference import predict_churn

        assert hasattr(predict_churn, "__wrapped__"), (
            "/predict deve ter @limiter.limit aplicado"
        )

    def test_limiter_is_registered_on_app_state(self):
        from slowapi import Limiter

        from src.main import app

        assert isinstance(app.state.limiter, Limiter), (
            "app.state.limiter deve ser uma instância de Limiter",
        )

    def test_rate_limit_settings_are_valid_strings(self):
        from src.core.config import settings

        assert "/" in settings.rate_limit_predict, (
            "rate_limit_predict deve ser no formato 'N/period', ex: '10/minute'"
        )

        assert "/" in settings.rate_limit_health, (
            "rate_limit_health deve ter formato 'N/period', ex: '60/minute'"
        )
