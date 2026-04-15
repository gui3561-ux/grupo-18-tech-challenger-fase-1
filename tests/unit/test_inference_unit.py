import pytest
from unittest.mock import MagicMock, patch
from src.schemas.inference import ChurnRequest, ChurnResponse


def _build_mock_service(churn_probability: float, churn_prediction: bool) -> MagicMock:
    """Helper para criar um mock do ChurnInferenceService."""
    mock_service = MagicMock()
    mock_service.predict.return_value = ChurnResponse(
        churn_probability=churn_probability,
        churn_prediction=churn_prediction,
        model="neural_network",
    )
    return mock_service


def _build_churn_request(**overrides) -> dict:
    """Helper para criar um payload base com overrides opcionais."""
    base = {
        "tenure_months": 2,
        "monthly_charges": 70.70,
        "total_charges": 151.65,
        "gender": "Female",
        "senior_citizen": "No",
        "partner": "No",
        "dependents": "No",
        "state": "CA",
        "phone_service": "Yes",
        "multiple_lines": "No",
        "internet_service": "Fiber optic",
        "online_security": "No",
        "online_backup": "No",
        "device_protection": "No",
        "tech_support": "No",
        "streaming_tv": "No",
        "streaming_movies": "No",
        "contract": "Month-to-month",
        "paperless_billing": "Yes",
        "payment_method": "Electronic check",
    }
    return {**base, **overrides}


class TestInferenceUnit:
    """Unit tests — testa a lógica de inferência isolada, sem subir a app."""

    def test_service_predict_returns_churn_response(self):
        mock_service = _build_mock_service(
            churn_probability=0.8512, churn_prediction=True
        )
        payload = _build_churn_request()
        result = mock_service.predict(payload)

        assert isinstance(result, ChurnResponse)
        assert result.churn_prediction is True
        assert result.churn_probability == 0.8512
        assert result.model == "neural_network"

    def test_service_predict_returns_no_churn_response(self):
        mock_service = _build_mock_service(
            churn_probability=0.1234, churn_prediction=False
        )
        payload = _build_churn_request()
        result = mock_service.predict(payload)

        assert result.churn_prediction is False
        assert result.churn_probability == 0.1234

    def test_service_predict_is_called_once(self):
        mock_service = _build_mock_service(
            churn_probability=0.7523, churn_prediction=True
        )
        payload = _build_churn_request()
        mock_service.predict(payload)

        mock_service.predict.assert_called_once()

    def test_service_predict_is_called_with_correct_payload(self):
        mock_service = _build_mock_service(
            churn_probability=0.7523, churn_prediction=True
        )
        payload = _build_churn_request()
        mock_service.predict(payload)

        mock_service.predict.assert_called_once_with(payload)

    def test_churn_probability_is_within_valid_range(self):
        mock_service = _build_mock_service(
            churn_probability=0.9999, churn_prediction=True
        )
        result = mock_service.predict(_build_churn_request())

        assert 0.0 <= result.churn_probability <= 1.0

    def test_service_predict_called_multiple_times(self):
        mock_service = _build_mock_service(
            churn_probability=0.5, churn_prediction=True
        )
        payload = _build_churn_request()

        mock_service.predict(payload)
        mock_service.predict(payload)
        mock_service.predict(payload)

        assert mock_service.predict.call_count == 3

    def test_churn_response_model_field_is_string(self):
        mock_service = _build_mock_service(
            churn_probability=0.65, churn_prediction=True
        )
        result = mock_service.predict(_build_churn_request())

        assert isinstance(result.model, str)
        assert len(result.model) > 0