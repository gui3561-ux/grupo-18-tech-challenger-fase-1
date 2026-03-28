from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    model_path: str = "models/lightgbm_pipeline.pkl"
    log_level: str = "INFO"
    api_title: str = "Churn Prediction API"
    api_version: str = "1.0.0"
    api_description: str = "API de inferência para o modelo de predição de churn de clientes. Construída com FastAPI seguindo princípios SOLID e pronta para receber um modelo de Machine Learning."

    risk_threshold_low: float = 0.3
    risk_threshold_medium: float = 0.6

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()