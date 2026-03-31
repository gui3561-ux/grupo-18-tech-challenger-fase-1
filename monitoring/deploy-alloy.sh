#!/usr/bin/env bash
# Deploy do Grafana Alloy como Azure Container Instance
# Uso: bash monitoring/deploy-alloy.sh <PROM_URL> <PROM_USER> <PROM_PASSWORD>
set -euo pipefail

GRAFANA_CLOUD_PROM_URL="${1:?Informe a URL do Prometheus Grafana Cloud (arg 1)}"
GRAFANA_CLOUD_PROM_USER="${2:?Informe o User ID do Prometheus Grafana Cloud (arg 2)}"
GRAFANA_CLOUD_PROM_PASSWORD="${3:?Informe o API Token do Grafana Cloud (arg 3)}"

RESOURCE_GROUP="rg-churn-api"
CONTAINER_NAME="alloy-churn-scraper"
LOCATION="brazilsouth"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ALLOY_CONFIG="$(cat "${SCRIPT_DIR}/alloy-config.alloy")"

echo "Fazendo deploy do Grafana Alloy como ACI..."

az container create \
  --name            "$CONTAINER_NAME" \
  --resource-group  "$RESOURCE_GROUP" \
  --location        "$LOCATION" \
  --image           "grafana/alloy:latest" \
  --cpu             0.5 \
  --memory          0.5 \
  --restart-policy  Always \
  --environment-variables \
    GRAFANA_CLOUD_PROM_URL="$GRAFANA_CLOUD_PROM_URL" \
    GRAFANA_CLOUD_PROM_USER="$GRAFANA_CLOUD_PROM_USER" \
  --secure-environment-variables \
    GRAFANA_CLOUD_PROM_PASSWORD="$GRAFANA_CLOUD_PROM_PASSWORD" \
  --command-line "alloy run --server.http.listen-addr=0.0.0.0:12345 /etc/alloy/config.alloy" \
  --azure-file-volume-account-name "" \
  2>&1 || true

# Estratégia: passar config como variável de ambiente e usar ConfigMap embutido
# O Alloy aceita config via stdin ou arquivo; usamos um script de init
az container delete \
  --name           "$CONTAINER_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --yes 2>/dev/null || true

echo "Criando storage account para a config do Alloy..."
STORAGE_ACCOUNT="churnalloycfg$(date +%s | tail -c 6)"
az storage account create \
  --name              "$STORAGE_ACCOUNT" \
  --resource-group    "$RESOURCE_GROUP" \
  --location          "$LOCATION" \
  --sku               Standard_LRS \
  --kind              StorageV2 2>&1 | tail -3

STORAGE_KEY=$(az storage account keys list \
  --account-name  "$STORAGE_ACCOUNT" \
  --resource-group "$RESOURCE_GROUP" \
  --query "[0].value" -o tsv)

az storage share create \
  --name         "alloy-config" \
  --account-name "$STORAGE_ACCOUNT" \
  --account-key  "$STORAGE_KEY" 2>&1 | tail -3

echo "$ALLOY_CONFIG" | az storage file upload \
  --source        /dev/stdin \
  --share-name    "alloy-config" \
  --path          "config.alloy" \
  --account-name  "$STORAGE_ACCOUNT" \
  --account-key   "$STORAGE_KEY" 2>&1 | tail -3

echo "Criando container ACI com Grafana Alloy..."
az container create \
  --name            "$CONTAINER_NAME" \
  --resource-group  "$RESOURCE_GROUP" \
  --location        "$LOCATION" \
  --image           "grafana/alloy:latest" \
  --cpu             0.5 \
  --memory          0.5 \
  --restart-policy  Always \
  --environment-variables \
    GRAFANA_CLOUD_PROM_URL="$GRAFANA_CLOUD_PROM_URL" \
    GRAFANA_CLOUD_PROM_USER="$GRAFANA_CLOUD_PROM_USER" \
  --secure-environment-variables \
    GRAFANA_CLOUD_PROM_PASSWORD="$GRAFANA_CLOUD_PROM_PASSWORD" \
  --azure-file-volume-account-name  "$STORAGE_ACCOUNT" \
  --azure-file-volume-account-key   "$STORAGE_KEY" \
  --azure-file-volume-share-name    "alloy-config" \
  --azure-file-volume-mount-path    "/etc/alloy" \
  --command-line "alloy run --server.http.listen-addr=0.0.0.0:12345 /etc/alloy/config.alloy"

echo ""
echo "Deploy concluído!"
echo "Verifique os logs com:"
echo "  az container logs --name $CONTAINER_NAME --resource-group $RESOURCE_GROUP"
