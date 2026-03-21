from pydantic import BaseModel
from typing import Literal


class ChurnRequest(BaseModel):
    tenure_months: int
    monthly_charges: float
    total_charges: float

    gender: Literal["Male", "Female"]
    senior_citizen: Literal["Yes", "No"]
    partner: Literal["Yes", "No"]
    dependents: Literal["Yes", "No"]
    state: str = "CA"

    phone_service: Literal["Yes", "No"]
    multiple_lines: Literal["Yes", "No", "No phone service"]

    internet_service: Literal["DSL", "Fiber optic", "No"]
    online_security: Literal["Yes", "No", "No internet service"]
    online_backup: Literal["Yes", "No", "No internet service"]
    device_protection: Literal["Yes", "No", "No internet service"]
    tech_support: Literal["Yes", "No", "No internet service"]
    streaming_tv: Literal["Yes", "No", "No internet service"]
    streaming_movies: Literal["Yes", "No", "No internet service"]

    # Contrato e pagamento
    contract: Literal["Month-to-month", "One year", "Two year"]
    paperless_billing: Literal["Yes", "No"]
    payment_method: Literal[
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ]

    model_config = {
        "json_schema_extra": {
            "example": {
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
                "payment_method": "Electronic check"
            }
        }
    }


class ChurnResponse(BaseModel):
    churn_probability: float
    churn_prediction: bool
    model: str
