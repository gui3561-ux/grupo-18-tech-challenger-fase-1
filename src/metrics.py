"""Definições de métricas Prometheus para monitoramento do modelo de churn."""

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

registry = CollectorRegistry()

# Métricas de negócio - Contadores
churn_predictions_total = Counter(
    "churn_predictions_total",
    "Total de predições de churn",
    ["prediction"],
    registry=registry,
)

# Métricas de latência do modelo
model_inference_seconds = Histogram(
    "model_inference_seconds",
    "Latência de inferência do modelo em segundos",
    buckets=[0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5],
    registry=registry,
)

# Métricas de distribuição de probabilidade
churn_probability_histogram = Histogram(
    "churn_probability_histogram",
    "Distribuição de probabilidades de churn",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    registry=registry,
)

# Métricas de aplicação
http_requests_total = Counter(
    "http_requests_total",
    "Total de requisições HTTP",
    ["method", "endpoint", "status_code"],
    registry=registry,
)

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "Duração de requisições HTTP em segundos",
    ["method", "endpoint"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
    registry=registry,
)

# Métricas de saúde
model_loaded = Gauge(
    "model_loaded",
    "Indica se o modelo está carregado (1) ou não (0)",
    registry=registry,
)

predictions_pending = Gauge(
    "predictions_pending",
    "Número de predições pendentes",
    registry=registry,
)
