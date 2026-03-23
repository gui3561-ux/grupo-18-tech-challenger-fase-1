#!/bin/bash
# Script de pós-deploy para limpeza de espaço no Azure App Service
# Execute após o deploy para liberar espaço em disco

echo "=== Iniciando limpeza de pós-deploy ==="

# Limpar cache do pip
echo "Limpando cache do pip..."
pip cache purge 2>/dev/null || pip cache dir -q | xargs rm -rf 2>/dev/null || true

# Remover diretórios de cache do Python
echo "Removendo caches do Python..."
rm -rf ~/.cache/pip 2>/dev/null || true
rm -rf ~/.cache/uv 2>/dev/null || true
rm -rf /tmp/pip* 2>/dev/null || true
rm -rf /tmp/uv* 2>/dev/null || true
rm -rf /tmp/oryx* 2>/dev/null || true
rm -rf /tmp/build* 2>/dev/null || true
rm -rf /tmp/zipdeploy* 2>/dev/null || true

# Remover logs antigos
echo "Removendo logs antigos..."
find /home/site -name "*.log" -mtime +7 -delete 2>/dev/null || true

# Remover __pycache__
echo "Removendo __pycache__..."
find /home/site/wwwroot -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Mostrar espaço liberado
echo "=== Espaço em disco após limpeza ==="
df -h /home

echo "=== Limpeza concluída ==="
