import pandas as pd
from pandera.pandas import Column, DataFrameSchema, Check

INFERENCE_URL = "/api/v1/inference/predict"

inference_response_schema = DataFrameSchema(
    {
        "churn_probability": Column(
            float,
            checks=[
                Check.greater_than_or_equal_to(0.0),
                Check.less_than_or_equal_to(1.0),
            ],
            nullable=False,
        ),
        "churn_prediction": Column(
            bool,
            nullable=False,
        ),
        "model": Column(
            str,
            nullable=False,
        ),
    }
)


class TestInferenceSchema:
    """Schema tests — valida estrutura do response com pandera."""

    def test_inference_response_schema_when_churn(
        self, client_with_mock_model, sample_payload
    ):
        response = client_with_mock_model.post(INFERENCE_URL, json=sample_payload)
        df = pd.DataFrame([response.json()])
        inference_response_schema.validate(df)

    def test_inference_response_has_no_extra_fields(
        self, client_with_mock_model, sample_payload
    ):
        response = client_with_mock_model.post(INFERENCE_URL, json=sample_payload)
        body = response.json()
        expected_fields = {"churn_probability", "churn_prediction", "model"}
        assert set(body.keys()) == expected_fields

    def test_inference_response_probability_is_float(
        self, client_with_mock_model, sample_payload
    ):
        response = client_with_mock_model.post(INFERENCE_URL, json=sample_payload)
        assert isinstance(response.json()["churn_probability"], float)

    def test_inference_response_prediction_is_bool(
        self, client_with_mock_model, sample_payload
    ):
        response = client_with_mock_model.post(INFERENCE_URL, json=sample_payload)
        assert isinstance(response.json()["churn_prediction"], bool)

    def test_inference_response_model_is_string(
        self, client_with_mock_model, sample_payload
    ):
        response = client_with_mock_model.post(INFERENCE_URL, json=sample_payload)
        assert isinstance(response.json()["model"], str)

    def test_inference_probability_within_valid_range(
        self, client_with_mock_model, sample_payload
    ):
        response = client_with_mock_model.post(INFERENCE_URL, json=sample_payload)
        probability = response.json()["churn_probability"]
        assert 0.0 <= probability <= 1.0