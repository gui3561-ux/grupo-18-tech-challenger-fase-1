import pytest
from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


@pytest.mark.unit
def test_health_status_200() -> None:
    response = client.get("/health")
    assert response.status_code == 200


@pytest.mark.unit
def test_health_returns_ok() -> None:
    response = client.get("/health")
    assert response.json() == {"status": "ok"}
