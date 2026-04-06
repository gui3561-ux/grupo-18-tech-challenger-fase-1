# Tech Challenge Fase 1 — MVP Churn Prediction

> **Pós-graduação FIAP Pós Tech** — Grupo 18

API de predição de churn de clientes de telecomunicações usando **FastAPI** e uma **Neural Network (PyTorch)**. O modelo identifica clientes com alto risco de cancelamento, permitindo ações preventivas de retenção.

---

## Links de Acesso

| Recurso | URL |
|---------|-----|
| **API (produção)** | https://churn-prediction-api.azurewebsites.net |
| **Health Check** | https://churn-prediction-api.azurewebsites.net/api/v1/health |
| **Swagger UI (docs interativas)** | https://churn-prediction-api.azurewebsites.net/docs |
| **ReDoc** | https://churn-prediction-api.azurewebsites.net/redoc |
| **Métricas Prometheus** | https://churn-prediction-api.azurewebsites.net/api/v1/metrics/ |
| **Grafana Dashboards** | https://gui3561.grafana.net/public-dashboards/94cb3fd572b74d69ad10a513388d8d86 |
| **GitHub Actions** | https://github.com/gui3561-ux/grupo-18-tech-challenger-fase-1/actions |

---

## Para que serve

O sistema prevê a **probabilidade de um cliente cancelar seus serviços** (churn) nos próximos meses com base em:

- Dados demográficos (gênero, idade, estado, parceiro, dependentes)
- Serviços contratados (internet, telefone, streaming, suporte técnico)
- Dados de contrato e pagamento (tipo de contrato, método de pagamento, fatura digital)
- Indicadores de risco calculados (perfil de alto risco, idoso isolado, custo por mês)

**Público-alvo:** equipes de retenção, marketing e gestão que precisam priorizar ações preventivas.

---

## Performance do Modelo

| Métrica | Valor |
|---------|-------|
| **ROC-AUC (teste)** | 0.8464 |
| **ROC-AUC (CV 5-fold)** | 0.8541 |
| **Features** | 35 (selecionadas via SelectKBest) |
| **Dataset** | IBM Telco Customer Churn — 7.043 registros |
| **Arquitetura** | MLP 128 → 64 → 32 → 1 com BatchNorm e Dropout |
| **Loss** | Focal Loss (gamma=3.0) + SMOTE |

Para detalhes completos do modelo (dados, vieses, ética, cenários de falha), consulte o [MODEL_CARD.md](MODEL_CARD.md).

---

## Endpoints da API

| Rota | Método | Descrição |
|------|--------|-----------|
| `/api/v1/health` | `GET` | Healthcheck — retorna `{"status": "ok"}` |
| `/api/v1/inference/predict` | `POST` | Predição de churn |
| `/api/v1/metrics/` | `GET` | Métricas Prometheus |
| `/docs` | — | Swagger UI interativo |
| `/redoc` | — | ReDoc (documentação alternativa) |

### Exemplo de requisição

```bash
curl -X POST https://churn-prediction-api.azurewebsites.net/api/v1/inference/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure_months": 2,
    "monthly_charges": 70.70,
    "total_charges": 151.65,
    "state": "California",
    "gender": "Female",
    "senior_citizen": "No",
    "partner": "No",
    "dependents": "No",
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

### Resposta

```json
{
  "churn_probability": 0.8743,
  "churn_prediction": true,
  "model": "neural_network"
}
```

### Categorização de risco

| Probabilidade | Categoria |
|---------------|-----------|
| `< 0.3` | Risco Baixo |
| `0.3 – 0.6` | Risco Médio |
| `>= 0.6` | Risco Alto |

---

## Monitoramento e Observabilidade

### Métricas expostas (Prometheus)

| Métrica | Tipo | Descrição |
|---------|------|-----------|
| `http_requests_total` | Counter | Total de requisições por método, endpoint e status |
| `http_request_duration_seconds` | Histogram | Latência HTTP (p50, p95, p99) |
| `churn_predictions_total` | Counter | Predições por classe (churn / não churn) |
| `model_inference_seconds` | Histogram | Tempo de inferência do modelo |
| `churn_probability_histogram` | Histogram | Distribuição de probabilidades |
| `model_loaded` | Gauge | Status do modelo (0 = não carregado, 1 = carregado) |

### Dashboards no Grafana

O arquivo [docs/grafana_dashboard.json](docs/grafana_dashboard.json) contém o dashboard pré-configurado com os seguintes painéis:

- **Request Rate** — Requisições por segundo
- **HTTP Request Latency** — p50, p95, p99
- **Error Rate (5xx)** — Taxa de erros
- **Model Inference Latency** — p50, p95, p99 da inferência
- **Predictions by Class** — Churn vs Não Churn ao longo do tempo
- **Churn Probability Distribution** — p50, p90, p99
- **Model Status** — Indicador visual de modelo carregado
- **Total Predictions** — Contagem acumulada

#### Como importar o dashboard

1. Acesse seu **Grafana Cloud**
2. Vá em **Dashboards → Import**
3. Faça upload de `docs/grafana_dashboard.json`
4. Selecione o datasource Prometheus do Grafana Cloud

### Coleta de métricas (Grafana Alloy)

O **Grafana Alloy** roda em um Azure Container Instance e faz scrape do endpoint `/api/v1/metrics/` a cada 15 segundos, enviando os dados via *remote write* para o Prometheus gerenciado.

**Deploy do Alloy:**

```bash
bash monitoring/deploy-alloy.sh <PROM_URL> <PROM_USER> <PROM_PASSWORD>
```

---

## Como usar localmente

### Pré-requisitos

- **Python 3.11+**
- **UV** — [Instalação](https://docs.astral.sh/uv/)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Instalação

```bash
uv venv
source .venv/bin/activate
uv sync
```

### Executar a API

```bash
uv run uvicorn src.main:app --reload
```

A API estará disponível em `http://localhost:8000`. Acesse `http://localhost:8000/docs` para a interface Swagger.

### Executar testes

```bash
uv run pytest
```

### Linting e verificação de tipos

```bash
uv run ruff check .
uv run mypy .
```

### Teste rápido com curl

```bash
curl -X POST http://localhost:8000/api/v1/inference/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure_months": 12,
    "monthly_charges": 79.85,
    "total_charges": 958.20,
    "state": "California",
    "gender": "Male",
    "senior_citizen": "No",
    "partner": "Yes",
    "dependents": "No",
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

---

## Deploy no Azure via GitHub Actions

O workflow `.github/workflows/deploy.yml` é executado automaticamente a cada push na branch `main`.

### Secrets necessários no GitHub

Acesse: **GitHub → Settings → Secrets and variables → Actions → New repository secret**

#### AZURE_CREDENTIALS

```bash
az ad sp create-for-rbac \
  --name "github-actions-churn-prediction-api" \
  --role contributor \
  --scopes "/subscriptions/$(az account show --query id -o tsv)/resourceGroups/rg-churn-api" \
  --sdk-auth
```

Copie o JSON retornado e cole como secret.

#### AZURE_WEBAPP_PUBLISH_PROFILE

```bash
az webapp deployment list-publishing-profiles \
  --name churn-prediction-api \
  --resource-group rg-churn-api \
  --query '[?publishMethod=="ZipDeploy"]' -o json
```

Copie o JSON retornado e cole como secret.

#### APPLICATIONINSIGHTS_CONNECTION_STRING (opcional)

```bash
az monitor app-insights component show \
  --app churn-prediction-api-insights \
  --resource-group rg-churn-api \
  --query connectionString -o tsv
```

### Fazer deploy

```bash
git add .
git commit -m "feat: Azure deploy"
git push origin main
```

### Verificar o deploy

- **GitHub Actions**: https://github.com/gui3561-ux/grupo-18-tech-challenger-fase-1/actions
- **Health check**: https://churn-prediction-api.azurewebsites.net/api/v1/health

---

## Arquitetura

```
Desenvolvimento → GitHub → GitHub Actions → GHCR (registry)
                                              ↓
                              Azure App Service (Linux, B1)
                              Web App: churn-prediction-api
                                              ↓
                              Grafana Alloy (ACI) → Grafana Cloud
```

Para justificativas detalhadas das escolhas técnicas, consulte [docs/arquitetura-deploy.md](docs/arquitetura-deploy.md).

---

## Estrutura do Projeto

```
├── src/                          # Código fonte da API
│   ├── main.py                   # FastAPI app factory
│   ├── metrics.py                # Métricas Prometheus
│   ├── middleware.py             # Middleware de latência
│   ├── api/v1/                   # Endpoints
│   ├── core/                     # Config, logging
│   ├── schemas/                  # Pydantic models
│   ├── services/                 # Serviço de inferência
│   └── tests/                    # Testes (unit, integration, smoke)
├── models/                       # Artefatos de modelo (.pkl)
├── utils/                        # Utilitários de ML
│   ├── neural_net.py             # Arquitetura ChurnNet
│   ├── metrics.py                # Funções de avaliação
│   └── feature_selection.py      # Seleção de features
├── notebooks/                    # EDA, feature engineering, modeling
├── data/                         # Dados (raw/ e processed/)
├── monitoring/                   # Config de observabilidade
│   ├── alloy-config.alloy        # Config do Grafana Alloy
│   └── deploy-alloy.sh           # Script de deploy do ACI
├── docs/                         # Documentação técnica
│   ├── arquitetura-deploy.md     # Arquitetura e justificativas
│   ├── api_inference.md          # Detalhes da API de inferência
│   ├── 01_eda.md                 # Análise exploratória
│   ├── 02_feature_engineering.md # Engenharia de features
│   ├── 03_modeling.md            # Treinamento de modelos
│   └── grafana_dashboard.json    # Dashboard do Grafana
├── mlflow_tracking/              # Logs de experimentos MLflow
├── Dockerfile                    # Imagem de produção
├── requirements.txt              # Dependências de runtime
└── pyproject.toml                # Config do projeto
```

---

## Trocar de modelo

O pipeline é intercambiável. Para usar LightGBM ou XGBoost em vez da Neural Network, basta alterar o `MODEL_PATH` no serviço de inferência:

```python
# Neural Network (padrão)
MODEL_PATH = pathlib.Path("models/neural_network_pipeline.pkl")

# LightGBM
MODEL_PATH = pathlib.Path("models/lightgbm_pipeline.pkl")

# XGBoost
MODEL_PATH = pathlib.Path("models/xgboost_pipeline.pkl")
```

O schema de entrada e o router permanecem os mesmos.

---

## Licença

MIT
