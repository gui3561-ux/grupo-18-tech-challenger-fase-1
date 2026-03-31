# Monitoramento da API de Churn Prediction

Este documento descreve a stack de monitoramento implementada usando **Azure Monitor + Prometheus + Grafana Cloud** (free tier).

## Arquitetura

```mermaid
flowchart LR
    A[App Service] -->|Métricas<br/>nativas| B[Azure Monitor]
    A -->|Logs| B
    A -->|/metrics<br/>Prometheus| C[Prometheus]
    B --> D[Application Insights]
    C --> E[Grafana Cloud]
```

## Componentes Implementados

### 1. Métricas Prometheus

**Endpoint:** `GET /api/v1/metrics/`

**Métricas expostas:**

| Métrica | Tipo | Descrição |
|---------|------|-----------|
| `churn_predictions_total` | Counter | Total de predições (labels: `churn`, `nao_churn`) |
| `model_inference_seconds` | Histogram | Latência de inferência do modelo |
| `churn_probability_histogram` | Histogram | Distribuição de probabilidades |
| `http_requests_total` | Counter | Total de requisições HTTP |
| `http_request_duration_seconds` | Histogram | Duração de requisições HTTP |
| `model_loaded` | Gauge | Status do modelo (0=erro, 1=ok) |

### 2. Logs Estruturados

O serviço já está instrumentado com `structlog` para logs estruturados em JSON.

## Configuração no Azure

### Opção 1: Application Insights (Recomendado - Grátis)

1. **Criar Application Insights:**
   ```bash
   az monitor app-insights create \
     --app churn-prediction-insights \
     --location eastus
   ```

2. **Obter Connection String:**
   ```bash
   az monitor app-insights show \
     --app churn-prediction-insights \
     --query connectionString
   ```

3. **Adicionar secrets no GitHub:**
   - `APPLICATIONINSIGHTS_CONNECTION_STRING`

4. **O Azure coletará automaticamente:**
   - CPU, memória, requisições HTTP
   - Exceções não tratadas
   - Logs (se configurado)

### Opção 2: Prometheus Scraper (Preview Grátis)

O Azure Managed Prometheus pode coletar do endpoint `/metrics`:

```bash
# App Settings
WEBSITE_PROMETHEUS_SCRAPE_ENDPOINT=/api/v1/metrics/
WEBSITE_PROMETHEUS_PORT=8001
PROMETHEUS_ENABLED=true
```

**Nota:** Azure Managed Prometheus está em preview gratuito.

## Configuração do Grafana Cloud (Free Tier)

### 1. Criar Conta

1. Acesse [grafana.com](https://grafana.com/)
2. Cadastre-se no free tier (1k séries, 14 dias retention)

### 2. Adicionar Data Source

**Opção A: Prometheus**
1. Go to Connections > Data Sources
2. Add Prometheus
3. URL: `https://prometheus-us-central1.grafana.net/api/prom`
4. Token: Grafana Cloud > My Account > API Keys

**Opção B: Azure Monitor**
1. Add Azure Monitor data source
2. Configure Tenant ID, Client ID, Secret

### 3. Importar Dashboard

1. Go to Dashboards > Import
2. Upload `docs/grafana_dashboard.json`
3. Selecione o data source

## Dashboard - Painéis

O dashboard inclui:

| Painel | Descrição | Alerta Sugerido |
|--------|-----------|-----------------|
| Request Rate | Requisições/segundo | > 100 rps |
| HTTP Latency | p50, p95, p99 | p99 > 2s |
| Error Rate | Taxa 5xx | > 1% |
| Model Latency | Latência inferência | p99 > 1s |
| Predictions | Distribuição churn/nao_churn | - |
| Churn Probability | Percentis de probabilidade | - |
| Model Status | 0=não carregado, 1=ok | = 0 |
| Total Predictions | Contador acumulado | - |

## Custo Estimado

| Componente | Custo Mensal |
|------------|---------------|
| Azure Monitor (platform metrics) | R$0 |
| Application Insights (500MB) | R$0 |
| Azure Managed Prometheus | R$0 (preview) |
| Grafana Cloud (free tier) | R$0 |
| **TOTAL** | **R$0** |

## Verificação

### Testar Endpoint de Métricas

```bash
# Localmente
curl http://localhost:8000/api/v1/metrics/

# Produção
curl https://churn-prediction-api.azurewebsites.net/api/v1/metrics/
```

### Exemplo de Saída

```
# HELP churn_predictions_total Total de predições de churn
# TYPE churn_predictions_total counter
churn_predictions_total{prediction="nao_churn"} 1523.0
churn_predictions_total{prediction="churn"} 847.0

# HELP model_inference_seconds Latência de inferência do modelo
# TYPE model_inference_seconds histogram
model_inference_seconds_bucket{le="0.01"} 0.0
model_inference_seconds_bucket{le="0.025"} 45.0
...
model_inference_seconds_sum 234.5
model_inference_seconds_count 2370.0

# HELP churn_probability_histogram Distribuição de probabilidades de churn
# TYPE churn_probability_histogram histogram
churn_probability_histogram_bucket{le="0.1"} 890.0
...
```

## Troubleshooting

### Métricas não aparecem

1. Verifique se o endpoint `/api/v1/metrics/` responde
2. Verifique logs do App Service
3. Confirme que `prometheus-client` está instalado

### Application Insights não funciona

1. Verifique `APPLICATIONINSIGHTS_CONNECTION_STRING`
2. Confirme que `APPINSIGHTS_ENABLED=true`

### Grafana não conecta

1. Verifique data source URL
2. Confirme API key válida
3. Teste connectivity

## Próximos Passos

1. Configurar Application Insights no Azure
2. Criar conta no Grafana Cloud
3. Adicionar secrets no GitHub
4. Fazer deploy e verificar métricas
5. Importar dashboard JSON

## Referências

- [Azure Monitor Pricing](https://azure.microsoft.com/pricing/details/monitor/)
- [Grafana Cloud Free Tier](https://grafana.com/cloud/)
- [Prometheus Client Python](https://prometheus-client.readthedocs.io/)
