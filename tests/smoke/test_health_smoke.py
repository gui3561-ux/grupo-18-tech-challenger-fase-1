HEALTH_URL = "/api/v1/health"


class TestHealthSmoke:
    def test_health_endpoint_is_reachable(self, client_loaded):
        response = client_loaded.get(HEALTH_URL)
        assert response.status_code == 200

    def test_health_endpoint_returns_json(self, client_loaded):
        response = client_loaded.get(HEALTH_URL)
        assert response.headers["content-type"] == "application/json"
        assert response.json() is not None

    def test_health_endpoint_responds_fast(self, client_loaded):
        import time

        start = time.perf_counter()
        client_loaded.get(HEALTH_URL)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 500

    def test_health_endpoint_does_not_return_500(self, client_loaded):
        response = client_loaded.get(HEALTH_URL)
        assert response.status_code != 500
