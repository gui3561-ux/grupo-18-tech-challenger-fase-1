import time

INFERENCE_URL = "/api/v1/inference/predict"


class TestInferenceSmoke:
    def test_inference_endpoint_is_reachable(
        self, client_with_mock_model, sample_payload
    ):
        response = client_with_mock_model.post(INFERENCE_URL, json=sample_payload)
        assert response.status_code != 404

    def test_inference_endpoint_returns_json(
        self, client_with_mock_model, sample_payload
    ):
        response = client_with_mock_model.post(INFERENCE_URL, json=sample_payload)
        assert "application/json" in response.headers["content-type"]
        assert response.json() is not None

    def test_inference_endpoint_responds_fast(
        self, client_with_mock_model, sample_payload
    ):
        start = time.perf_counter()
        client_with_mock_model.post(INFERENCE_URL, json=sample_payload)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 500

    def test_inference_endpoint_does_not_return_500(
        self, client_with_mock_model, sample_payload
    ):
        response = client_with_mock_model.post(INFERENCE_URL, json=sample_payload)
        assert response.status_code != 500
