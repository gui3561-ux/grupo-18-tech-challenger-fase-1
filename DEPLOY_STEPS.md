# Deploy no Azure - Recursos já existem em brazilsouth

## Passos:

### 1. Verificar se App Service existe
```bash
az webapp show --name churn-prediction-api --resource-group rg-churn-api
```

Se não existir, criar:
```bash
az appservice plan create --name asp-churn-api --resource-group rg-churn-api --sku B1 --is-linux --location brazilsouth
az webapp create --name churn-prediction-api --resource-group rg-churn-api --plan asp-churn-api --runtime "PYTHON|3.12" --deployment-local-git
az webapp config set --name churn-prediction-api --resource-group rg-churn-api --linux-fx-version "PYTHON|3.12"
az webapp config set --name churn-prediction-api --resource-group rg-churn-api --startup-file "gunicorn src.main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 --timeout 300"
```

### 2. Criar Application Insights (se não existir)
```bash
az monitor app-insights component create \
  --app churn-prediction-api-insights \
  --location brazilsouth \
  --resource-group rg-churn-api \
  --application-type web
```

### 3. Obter AZURE_CREDENTIALS
```bash
az ad sp create-for-rbac \
  --name "github-actions-churn-prediction-api" \
  --role contributor \
  --scopes "/subscriptions/$(az account show --query id -o tsv)/resourceGroups/rg-churn-api" \
  --sdk-auth
```
Resultado: JSON → GitHub Secret `AZURE_CREDENTIALS`

### 4. Obter AZURE_WEBAPP_PUBLISH_PROFILE
```bash
az webapp deployment list-publishing-profiles \
  --name churn-prediction-api \
  --resource-group rg-churn-api \
  --query '[?publishMethod=="ZipDeploy"]' -o json
```
Resultado: JSON → GitHub Secret `AZURE_WEBAPP_PUBLISH_PROFILE`

### 5. Obter APPLICATIONINSIGHTS_CONNECTION_STRING
```bash
az monitor app-insights component show \
  --app churn-prediction-api-insights \
  --resource-group rg-churn-api \
  --query connectionString -o tsv
```
Resultado: String → GitHub Secret `APPLICATIONINSIGHTS_CONNECTION_STRING` (opcional)

### 6. Deploy
```bash
git add .
git commit -m "feat: Azure deploy"
git push origin main
```