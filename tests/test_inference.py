import pytest
from unittest.mock import MagicMock, patch
from src.schemas.inference import ChurnResponse


class TestInferenceEndpoint:
    """Testes para o endpoint POST /api/v1/inference/predict."""

    def test_predict_returns_200(self, client_with_mock_model, sample_payload):
        response = client_with_mock_model.post(
            "/api/v1/inference/predict", json=sample_payload
        )
        assert response.status_code == 200

    def test_predict_response_schema(self, client_with_mock_model, sample_payload):
        response = client_with_mock_model.post(
            "/api/v1/inference/predict", json=sample_payload
        )
        body = response.json()

        assert "churn_probability" in body
        assert "churn_prediction" in body
        assert "model" in body

    def test_predict_probability_is_float(self, client_with_mock_model, sample_payload):
        response = client_with_mock_model.post(
            "/api/v1/inference/predict", json=sample_payload
        )
        body = response.json()

        assert isinstance(body["churn_probability"], float)
        assert 0.0 <= body["churn_probability"] <= 1.0

    def test_predict_prediction_is_bool(self, client_with_mock_model, sample_payload):
        response = client_with_mock_model.post(
            "/api/v1/inference/predict", json=sample_payload
        )
        body = response.json()

        assert isinstance(body["churn_prediction"], bool)

    def test_predict_missing_required_field_returns_422(self, client_with_mock_model):
        incomplete = {"tenure_months": 2, "monthly_charges": 70.70}
        response = client_with_mock_model.post(
            "/api/v1/inference/predict", json=incomplete
        )
        assert response.status_code == 422

    def test_predict_empty_body_returns_422(self, client_with_mock_model):
        response = client_with_mock_model.post(
            "/api/v1/inference/predict", json={}
        )
        assert response.status_code == 422

    def test_predict_invalid_gender_returns_422(
        self, client_with_mock_model, sample_payload
    ):
        sample_payload["gender"] = "Invalid"
        response = client_with_mock_model.post(
            "/api/v1/inference/predict", json=sample_payload
        )
        assert response.status_code == 422

    def test_predict_invalid_contract_returns_422(
        self, client_with_mock_model, sample_payload
    ):
        sample_payload["contract"] = "Three year"
        response = client_with_mock_model.post(
            "/api/v1/inference/predict", json=sample_payload
        )
        assert response.status_code == 422

    def test_predict_invalid_internet_service_returns_422(
        self, client_with_mock_model, sample_payload
    ):
        sample_payload["internet_service"] = "Satellite"
        response = client_with_mock_model.post(
            "/api/v1/inference/predict", json=sample_payload
        )
        assert response.status_code == 422

    def test_predict_wrong_method_returns_405(self, client_with_mock_model):
        response = client_with_mock_model.get("/api/v1/inference/predict")
        assert response.status_code == 405

    def test_predict_calls_service(
        self, client_with_mock_model, mock_inference_service, sample_payload
    ):
        client_with_mock_model.post(
            "/api/v1/inference/predict", json=sample_payload
        )
        mock_inference_service.predict.assert_called_once()

    def test_predict_no_churn_response(self, sample_payload):
        """Testa resposta quando o modelo prediz não-churn."""
        no_churn_service = MagicMock()
        no_churn_service.predict.return_value = ChurnResponse(
            churn_probability=0.1234,
            churn_prediction=False,
            model="neural_network",
        )

        with patch("src.routers.inference._service", no_churn_service):
            from fastapi import FastAPI
            from fastapi.testclient import TestClient
            from src.api.v1.router import api_router

            app = FastAPI()
            app.state.model_loaded = True
            app.include_router(api_router)

            with TestClient(app) as client:
                response = client.post(
                    "/api/v1/inference/predict", json=sample_payload
                )
                body = response.json()

                assert body["churn_prediction"] is False
                assert body["churn_probability"] == 0.1234
