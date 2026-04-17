# Churn Prediction API

API REST construída com **FastAPI** para servir um modelo de rede neural de predição de churn de clientes de telecomunicações. O modelo é carregado a partir de um pipeline serializado (`.pkl`) e expõe endpoints para inferência e healthcheck.

---

## Estrutura do projeto

```
src/
├── main.py                        # Cria a instância do FastAPI e registra os routers
├── api/
│   └── v1/
│       └── router.py              # Agrega os routers com prefixo /api/v1
├── routers/
│   ├── health.py                  # GET /api/v1/health
│   └── inference.py               # POST /api/v1/inference/predict
├── schemas/
│   ├── health.py                  # HealthResponse
│   └── inference.py               # ChurnRequest / ChurnResponse
├── services/
│   └── inference_service.py       # ChurnInferenceService — carrega o modelo e executa predições
├── tests/
│   ├── test_health.py
│   └── test_predict.py
├── pyproject.toml                 # Dependências, lint e configuração do pytest
└── README.md
```

O modelo treinado deve estar disponível em:

```
models/neural_network_pipeline.pkl
```

---

## Instalação

```bash
# 1. Clone o repositório
git clone git@github.com:gui3561-ux/grupo-18-tech-challenger-fase-1.git
cd grupo-18-tech-challenger-fase-1/src

# 2. Crie e ative o ambiente virtual
python -m venv .venv
source .venv/bin/activate      # Linux/macOS

# 3. Instale as dependências (produção + dev)
pip install -e ".[dev]"
```

---

## Executando a API

```bash
uvicorn src.main:app --reload
```

A API estará disponível em `http://localhost:8000`.

| URL | Descrição |
|-----|-----------|
| `http://localhost:8000/docs` | Swagger UI interativo |
| `http://localhost:8000/redoc` | Documentação ReDoc |

---

## Endpoints

Todos os endpoints estão sob o prefixo `/api/v1`.

### `GET /api/v1/health`

Verifica se a API está operacional.

**Response `200 OK`:**
```json
{
  "status": "ok",
  "version": "1.0.0"
}
```

---

### `POST /api/v1/inference/predict`

Recebe os dados do cliente e retorna a probabilidade e a predição de churn.

**Request body:**

| Campo | Tipo | Valores aceitos |
|-------|------|-----------------|
| `tenure_months` | `int` | Meses de contrato |
| `monthly_charges` | `float` | Valor mensal |
| `total_charges` | `float` | Total cobrado |
| `gender` | `string` | `"Male"`, `"Female"` |
| `senior_citizen` | `string` | `"Yes"`, `"No"` |
| `partner` | `string` | `"Yes"`, `"No"` |
| `dependents` | `string` | `"Yes"`, `"No"` |
| `state` | `string` | Ex.: `"CA"` (padrão) |
| `phone_service` | `string` | `"Yes"`, `"No"` |
| `multiple_lines` | `string` | `"Yes"`, `"No"`, `"No phone service"` |
| `internet_service` | `string` | `"DSL"`, `"Fiber optic"`, `"No"` |
| `online_security` | `string` | `"Yes"`, `"No"`, `"No internet service"` |
| `online_backup` | `string` | `"Yes"`, `"No"`, `"No internet service"` |
| `device_protection` | `string` | `"Yes"`, `"No"`, `"No internet service"` |
| `tech_support` | `string` | `"Yes"`, `"No"`, `"No internet service"` |
| `streaming_tv` | `string` | `"Yes"`, `"No"`, `"No internet service"` |
| `streaming_movies` | `string` | `"Yes"`, `"No"`, `"No internet service"` |
| `contract` | `string` | `"Month-to-month"`, `"One year"`, `"Two year"` |
| `paperless_billing` | `string` | `"Yes"`, `"No"` |
| `payment_method` | `string` | `"Electronic check"`, `"Mailed check"`, `"Bank transfer (automatic)"`, `"Credit card (automatic)"` |

**Exemplo de request:**
```json
{
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
```

**Response `200 OK`:**
```json
{
  "churn_probability": 0.8731,
  "churn_prediction": true,
  "model": "neural_network"
}
```

| Campo | Tipo | Descrição |
|-------|------|-----------|
| `churn_probability` | `float` | Probabilidade de churn (0.0 a 1.0) |
| `churn_prediction` | `bool` | `true` se probabilidade >= 0.5 |
| `model` | `string` | Identificador do modelo utilizado |

**Response `422 Unprocessable Entity`:** retornado quando algum campo obrigatório está ausente ou com valor inválido.

---

## Feature Engineering

O serviço aplica internamente as seguintes features derivadas antes de enviar os dados ao modelo:

| Feature | Descrição |
|---------|-----------|
| `high_risk_profile` | `1` se `internet_service == "Fiber optic"` e `contract == "Month-to-month"` |
| `isolated_senior` | `1` se `senior_citizen == "Yes"`, `partner == "No"` e `dependents == "No"` |
| `internet_services_count` | Quantidade de serviços de internet ativos (Online Security, Online Backup, Device Protection, Tech Support, Streaming TV, Streaming Movies) |
| `cost_per_month` | `monthly_charges / (tenure_months + 1)` |

---

## Testes

```bash
# Todos os testes
pytest

# Apenas testes unitários
pytest -m unit

# Apenas testes de integração
pytest -m integration
```

---

## Qualidade de código

```bash
ruff check src tests       # lint
ruff format src tests      # formatação
mypy src                   # checagem estática de tipos
```

---

## Dependências principais

| Pacote | Versao minima |
|--------|---------------|
| fastapi | 0.111.0 |
| uvicorn | 0.29.0 |
| pydantic | 2.0.0 |
| scikit-learn | 1.4.0 |
| pandas | 2.0.0 |
| torch | 2.3 |
| numpy | 1.26.0 |
