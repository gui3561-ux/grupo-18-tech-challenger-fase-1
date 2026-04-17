class TestMetricsEndpoint:
    """Testes para os endpoints de métricas Prometheus."""

    def test_metrics_returns_200(self, client_loaded):
        response = client_loaded.get("/api/v1/metrics/")
        assert response.status_code == 200

    def test_metrics_content_type_is_prometheus(self, client_loaded):
        response = client_loaded.get("/api/v1/metrics/")
        content_type = response.headers["content-type"]

        assert "text/plain" in content_type or "text/openmetrics" in content_type

    def test_metrics_body_contains_expected_metrics(self, client_loaded):
        response = client_loaded.get("/api/v1/metrics/")
        body = response.text

        assert "churn_predictions_total" in body
        assert "model_inference_seconds" in body
        assert "churn_probability_histogram" in body

    def test_metrics_body_contains_http_metrics(self, client_loaded):
        response = client_loaded.get("/api/v1/metrics/")
        body = response.text

        assert "http_requests_total" in body
        assert "http_request_duration_seconds" in body

    def test_metrics_body_contains_model_loaded_gauge(self, client_loaded):
        response = client_loaded.get("/api/v1/metrics/")
        body = response.text

        assert "model_loaded" in body

    def test_metrics_health_returns_200(self, client_loaded):
        response = client_loaded.get("/api/v1/metrics/health")
        assert response.status_code == 200

    def test_metrics_health_response_body(self, client_loaded):
        response = client_loaded.get("/api/v1/metrics/health")
        body = response.json()

        assert body["status"] == "ok"
        assert body["metrics_endpoint"] == "/metrics"

    def test_metrics_wrong_method_returns_405(self, client_loaded):
        response = client_loaded.post("/api/v1/metrics/")
        assert response.status_code == 405
