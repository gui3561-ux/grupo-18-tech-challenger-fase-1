# Tech Challenge Fase 1 — MVP Churn Prediction

API de predição de churn usando **FastAPI** + **Neural Network (PyTorch)**.

Para a **arquitetura de deploy** (Azure, GitHub Actions, contentor, observabilidade) e as **razões das escolhas técnicas**, ver [docs/arquitetura-deploy.md](docs/arquitetura-deploy.md).

## Deploy no Azure via GitHub Actions

### 1. Configurar 3 Secrets no GitHub

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
Copie a string retornada e cole como secret.

### 2. Fazer Deploy
```bash
git add .
git commit -m "feat: Azure deploy"
git push origin main
```

O workflow `.github/workflows/deploy.yml` executa automaticamente.

### 3. Verificar
- **GitHub Actions**: https://github.com/gui3561-ux/grupo-18-tech-challenger-fase-1/actions
- **API**: https://churn-prediction-api.azurewebsites.net/api/v1/health

## Endpoints

| Rota | Descrição |
|------|-----------|
| `/api/v1/health` | Healthcheck |
| `/api/v1/inference/predict` | Predição de churn (POST) |
| `/api/v1/metrics/` | Métricas Prometheus |
| `/docs` | Swagger UI |

## Modelo Neural Network

- **ROC-AUC**: 0.8464 (teste), 0.8541 (CV)
- **Framework**: PyTorch MLP (128→64→32→1)
- **Features**: 35 (selecionadas via SelectKBest)

## Pré-requisitos 

[UV](https://docs.astral.sh/uv/)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Instalação

```bash
uv venv
source .venv/bin/activate
uv sync
```

## Setup Local

```bash
uv run uvicorn src.main:app --reload
```

## Testes

```bash
uv run pytest
```

## Licença

MIT