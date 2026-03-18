# Tech Challenge Fase 1 — MVP Churn Prediction

Projeto de análise exploratória e modelagem preditiva de churn utilizando o dataset **Telco Customer Churn**.

## Pré-requisitos

- Python 3.10+
- [uv](https://docs.astral.sh/uv/)

## Setup

```bash
# Criar o ambiente virtual e instalar dependências
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## API em produção

A API está disponível publicamente no Azure App Service:

**Base URL:** `https://churn-prediction-api.azurewebsites.net`

| Rota | Método | URL completa | Descrição |
|---|---|---|---|
| `/api/v1/health` | GET | [/api/v1/health](https://churn-prediction-api.azurewebsites.net/api/v1/health) | Healthcheck da API |
| `/api/v1/inference/hello` | GET | [/api/v1/inference/hello](https://churn-prediction-api.azurewebsites.net/api/v1/inference/hello) | Hello World (stub de inferência) |
| `/docs` | GET | [/docs](https://churn-prediction-api.azurewebsites.net/docs) | Documentação interativa (Swagger UI) |
| `/redoc` | GET | [/redoc](https://churn-prediction-api.azurewebsites.net/redoc) | Documentação alternativa (ReDoc) |

## Executando a API localmente

```bash
uvicorn src.main:app --reload
```

Acesse a documentação interativa em [http://localhost:8000/docs](http://localhost:8000/docs).

## Executando o notebook

```bash
jupyter notebook notebooks/eda.ipynb
```

## Estrutura do projeto

```
data/                          → Dataset Telco Customer Churn
notebooks/                     → Notebooks de análise (EDA, modelagem)
src/
├── api/v1/router.py           → Agrega todos os routers na versão v1
├── routers/
│   ├── health.py              → Rota /health
│   └── inference.py           → Rota /inference/hello (futura inferência ML)
├── schemas/
│   ├── health.py              → Schema de resposta do healthcheck
│   └── inference.py           → Schema de resposta da inferência
├── services/
│   └── inference_service.py  → Interface + implementação stub do serviço de inferência
└── main.py                    → Factory da aplicação FastAPI
models/                        → Modelos treinados
tests/                         → Testes
docs/                          → Documentação
```

## Licença

MIT
