from abc import ABC, abstractmethod
import pickle
import pathlib
import time
import pandas as pd
import structlog

from src.schemas.inference import ChurnRequest, ChurnResponse
from src.metrics import (
    churn_predictions_total,
    model_inference_seconds,
    churn_probability_histogram,
)
from pathlib import Path

# Melhor modelo: Neural Network (ROC-AUC: 0.8464 teste, 0.8541 CV)
MODEL_PATH = pathlib.Path("models/neural_network_pipeline.pkl")

logger = structlog.get_logger(__name__)


class ModelNotLoadedError(Exception):
    """Levantada quando o pipeline de ML não está disponível"""

class InferenceServiceInterface(ABC):

    @abstractmethod
    def predict(self) -> ChurnResponse:
        pass


class ChurnInferenceService(InferenceServiceInterface):
    def __init__(self, model_path: str | Path = MODEL_PATH):
        self._model_path = Path(model_path)

        if not self._model_path.exists():
            raise ModelNotLoadedError(
                f"Arquivo de modelo não encontrado: {self._model_path}"
            )

        try:
            with open(self._model_path, "rb") as f:
                self._pipeline = pickle.load(f)
        except Exception as exc:
            raise ModelNotLoadedError(
                f"Falha ao carregar o modelo em '{self._model_path}': {exc}"
            ) from exc

    def predict(self, req: ChurnRequest) -> ChurnResponse:
        start_time = time.time()

        df = self.__prepare_dataframe(req)
        df = self.__feature_engineering(df)

        proba = float(self._pipeline.predict_proba(df)[0, 1])
        prediction = proba >= 0.5

        elapsed = time.time() - start_time

        model_inference_seconds.observe(elapsed)
        churn_predictions_total.labels(
            prediction="churn" if prediction else "nao_churn"
        ).inc()
        churn_probability_histogram.observe(proba)

        logger.info(
            "Prediction done",
            probability=proba,
            prediction=prediction,
            latency_ms=round(elapsed * 1000, 2),
        )

        return ChurnResponse(
            churn_probability=round(proba, 4),
            churn_prediction=prediction,
            model="neural_network",
        )

    def __prepare_dataframe(self, req: ChurnRequest) -> pd.DataFrame:
        logger.info("Preparing DataFrame")
        data = {
            "Tenure Months":     req.tenure_months,
            "Monthly Charges":   req.monthly_charges,
            "Total Charges":     req.total_charges,
            "State":             req.state,
            "Gender":            req.gender,
            "Senior Citizen":    req.senior_citizen,
            "Partner":           req.partner,
            "Dependents":        req.dependents,
            "Phone Service":     req.phone_service,
            "Multiple Lines":    req.multiple_lines,
            "Internet Service":  req.internet_service,
            "Online Security":   req.online_security,
            "Online Backup":     req.online_backup,
            "Device Protection": req.device_protection,
            "Tech Support":      req.tech_support,
            "Streaming TV":      req.streaming_tv,
            "Streaming Movies":  req.streaming_movies,
            "Contract":          req.contract,
            "Paperless Billing": req.paperless_billing,
            "Payment Method":    req.payment_method,
        }
        return pd.DataFrame([data])

    def __feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Started feature engineering")
        df["high_risk_profile"] = (
            (df["Internet Service"] == "Fiber optic") &
            (df["Contract"] == "Month-to-month")
        ).astype(int)

        df["isolated_senior"] = (
            (df["Senior Citizen"] == "Yes") &
            (df["Partner"] == "No") &
            (df["Dependents"] == "No")
        ).astype(int)

        servicos = ["Online Security", "Online Backup", "Device Protection",
                    "Tech Support", "Streaming TV", "Streaming Movies"]
        df["internet_services_count"] = sum(
            (df[c] == "Yes").astype(int) for c in servicos
        )
        df["cost_per_month"] = df["Monthly Charges"] / (df["Tenure Months"] + 1)
        return df