import pandera as pa
import pandas as pd
from pandera import Column, DataFrameSchema, Check
from fastapi.testclient import TestClient


HEALTH_URL = "/api/v1/health"

health_response_schema = DataFrameSchema(
    {
        "status": Column(
            str,
            checks=Check.isin(["ok", "degraded"]),
            nullable=False,
        ),
        "model_loaded": Column(
            bool,
            nullable=False,
        ),
        "version": Column(
            str,
            checks=Check(lambda x: x.str.match(r"^\d+\.\d+\.\d+$"),
                        error="version must follow semver format"),
            nullable=False,
        ),
    }
)


class TestHealthSchema:

    def test_health_response_schema_when_model_is_loaded(self, client_loaded):
        response = client_loaded.get(HEALTH_URL)
        df = pd.DataFrame([response.json()])
        health_response_schema.validate(df)

    def test_health_response_schema_when_model_is_degraded(self, client_degraded):
        response = client_degraded.get(HEALTH_URL)
        df = pd.DataFrame([response.json()])
        health_response_schema.validate(df)

    def test_health_response_has_no_extra_fields(self, client_loaded):
        response = client_loaded.get(HEALTH_URL)
        body = response.json()
        expected_fields = {"status", "model_loaded", "version"}
        assert set(body.keys()) == expected_fields

    def test_health_response_status_is_string(self, client_loaded):
        response = client_loaded.get(HEALTH_URL)
        assert isinstance(response.json()["status"], str)

    def test_health_response_model_loaded_is_bool(self, client_loaded):
        response = client_loaded.get(HEALTH_URL)
        assert isinstance(response.json()["model_loaded"], bool)