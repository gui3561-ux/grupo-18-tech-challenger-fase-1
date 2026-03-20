# 04 — API de Inferência (FastAPI)

**Arquivos relevantes:** `src/main.py`, `src/routers/inference.py`, `src/schemas/inference.py`, `src/services/inference_service.py`
**Modelo:** `models/neural_network_pipeline.pkl` (ou qualquer outro pkl gerado pelo `comparar_modelos`)

---

## Como funciona

O pipeline salvo em `.pkl` é **self-contained**: contém o `ColumnTransformer` (encoding + normalização), o `SelectKBest` e o classificador. A API precisa apenas:

1. Receber os campos originais do cliente via JSON
2. Calcular as 4 features derivadas (feature engineering)
3. Montar um `pd.DataFrame` e chamar `pipeline.predict_proba()`

---

## Fluxo

```
POST /api/v1/inference/predict
    JSON com 20 campos originais do cliente
        ↓
    ChurnRequest  (validação Pydantic)
        ↓
    Feature engineering  (4 features derivadas)
        ↓
    pipeline.predict_proba(df)   ← pkl self-contained
        ↓
    ChurnResponse { churn_probability, churn_prediction, model }
```

---

## 1. Schema de entrada e saída

**Arquivo:** `src/schemas/inference.py`

```python
from pydantic import BaseModel
from typing import Literal


class ChurnRequest(BaseModel):
    # Numéricas
    tenure_months: int
    monthly_charges: float
    total_charges: float

    # Demográficas
    gender: Literal["Male", "Female"]
    senior_citizen: Literal["Yes", "No"]
    partner: Literal["Yes", "No"]
    dependents: Literal["Yes", "No"]
    state: str = "CA"

    # Serviços de telefone
    phone_service: Literal["Yes", "No"]
    multiple_lines: Literal["Yes", "No", "No phone service"]

    # Serviços de internet
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
    churn_probability: float   # probabilidade de churnar (0.0 – 1.0)
    churn_prediction: bool     # True = vai churnar
    model: str                 # qual modelo foi usado
```

---

## 2. Service — carrega o pkl e executa a inferência

**Arquivo:** `src/services/inference_service.py`

```python
import pickle
import pathlib
import pandas as pd
from src.schemas.inference import ChurnRequest, ChurnResponse

MODEL_PATH = pathlib.Path("models/neural_network_pipeline.pkl")


class ChurnInferenceService:

    def __init__(self):
        with open(MODEL_PATH, "rb") as f:
            self._pipeline = pickle.load(f)

    def predict(self, req: ChurnRequest) -> ChurnResponse:
        # 1. Monta o dict com os nomes EXATOS que o pipeline espera
        #    (com espaços e maiúsculas iguais aos do treino)
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
        df = pd.DataFrame([data])

        # 2. Feature engineering — mesmo código do notebook
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

        # 3. Inferência — pipeline faz encoding + seleção + predição internamente
        proba = float(self._pipeline.predict_proba(df)[0, 1])

        return ChurnResponse(
            churn_probability=round(proba, 4),
            churn_prediction=proba >= 0.5,
            model="neural_network",
        )
```

---

## 3. Router — endpoint POST /predict

**Arquivo:** `src/routers/inference.py`

```python
from fastapi import APIRouter
from src.schemas.inference import ChurnRequest, ChurnResponse
from src.services.inference_service import ChurnInferenceService

router = APIRouter(prefix="/inference", tags=["Inference"])

# Pipeline carregado uma vez ao iniciar a aplicação
_service = ChurnInferenceService()


@router.post("/predict", response_model=ChurnResponse)
def predict_churn(request: ChurnRequest) -> ChurnResponse:
    return _service.predict(request)
```

---

## 4. Exemplo de requisição

```bash
curl -X POST http://localhost:8000/api/v1/inference/predict \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

**Resposta:**
```json
{
  "churn_probability": 0.8743,
  "churn_prediction": true,
  "model": "neural_network"
}
```

---

## 5. Trocar de modelo

Para usar LightGBM, XGBoost ou qualquer outro modelo treinado, basta mudar o `MODEL_PATH` no service — o schema de entrada e o router permanecem iguais:

```python
# Neural Network
MODEL_PATH = pathlib.Path("models/neural_network_pipeline.pkl")

# LightGBM
MODEL_PATH = pathlib.Path("models/lightgbm_pipeline.pkl")

# XGBoost
MODEL_PATH = pathlib.Path("models/xgboost_pipeline.pkl")
```

---

## Observações importantes

| Ponto | Detalhe |
|-------|---------|
| **Nomes das colunas** | Devem ser idênticos aos do treino (com espaços, maiúsculas). O `ColumnTransformer` usa os nomes para rotear cada coluna ao bloco correto. |
| **Colunas extras** | O `ColumnTransformer` com `remainder='drop'` ignora qualquer coluna não declarada — sem risco de erro. |
| **Features derivadas** | `high_risk_profile`, `isolated_senior`, `internet_services_count` e `cost_per_month` são calculadas na API, não enviadas pelo cliente. |
| **Carregamento do pkl** | Feito uma única vez no `__init__` do service, não a cada requisição. |
