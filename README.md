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

## Executando o notebook

```bash
jupyter notebook notebooks/eda.ipynb
```

## Estrutura do projeto

```
data/        → Dataset Telco Customer Churn
notebooks/   → Notebooks de análise (EDA, modelagem)
src/         → Módulos Python para API
models/      → Modelos treinados
tests/       → Testes
docs/        → Documentação
```

## Licença

MIT
