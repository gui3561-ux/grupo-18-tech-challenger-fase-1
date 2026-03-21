import pytest
from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


@pytest.mark.unit
def test_predict_status_200() -> None:
    response = client.post("/predict", json={"features": [5.1, 3.5, 1.4, 0.2]})
    assert response.status_code == 200


@pytest.mark.unit
def test_predict_response_shape() -> None:
    response = client.post("/predict", json={"features": [5.1, 3.5, 1.4, 0.2]})
    body = response.json()
    assert "prediction" in body
    assert "probability" in body


@pytest.mark.unit
def test_predict_missing_features_returns_422() -> None:
    response = client.post("/predict", json={})
    assert response.status_code == 422


@pytest.mark.unit
def test_predict_empty_features_returns_422() -> None:
    response = client.post("/predict", json={"features": []})
    assert response.status_code == 422
