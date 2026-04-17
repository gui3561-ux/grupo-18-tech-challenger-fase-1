from src.core.config import settings


class TestHealthEndpoint:
    """Testes para o endpoint GET /api/v1/health."""

    def test_health_ok_when_model_loaded(self, client_loaded):
        response = client_loaded.get("/api/v1/health")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True
        assert body["version"] == settings.api_version

    def test_health_degraded_when_model_not_loaded(self, client_degraded):
        response = client_degraded.get("/api/v1/health")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "degraded"
        assert body["model_loaded"] is False

    def test_health_response_has_all_fields(self, client_loaded):
        response = client_loaded.get("/api/v1/health")
        body = response.json()

        assert set(body.keys()) == {"status", "model_loaded", "version"}

    def test_health_version_is_semver(self, client_loaded):
        response = client_loaded.get("/api/v1/health")
        version = response.json()["version"]
        parts = version.split(".")

        assert len(parts) == 3
        assert all(part.isdigit() for part in parts)

    def test_health_wrong_method_returns_405(self, client_loaded):
        response = client_loaded.post("/api/v1/health")
        assert response.status_code == 405
