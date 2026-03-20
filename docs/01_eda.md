# 01 — Análise Exploratória de Dados (EDA)

**Notebook:** `notebooks/01_eda.ipynb`
**Dataset:** `data/Telco_customer_churn.csv`

---

## Objetivo

Entender a estrutura, distribuições e padrões do dataset Telco Customer Churn antes de qualquer modelagem. A EDA orienta decisões de feature engineering, pré-processamento e seleção de variáveis.

---

## Dataset

| Atributo | Valor |
|----------|-------|
| Linhas | 7.043 clientes |
| Colunas originais | 33 |
| Período | Telco California (EUA) |
| Target | `Churn Value` (0 = retido, 1 = cancelou) |

### Colunas removidas na leitura

| Coluna | Motivo |
|--------|--------|
| `CustomerID` | Identificador único sem valor preditivo |
| `Count` | Constante (sempre 1) |
| `Churn Reason` | 73% de valores ausentes; informação pós-churn |

Após remoção: **7.043 linhas × 30 colunas**.

---

## Tipos de Variáveis

**Numéricas (6):**
- `Latitude`, `Longitude` — geolocalização
- `Tenure Months` — tempo de permanência em meses
- `Monthly Charges` — valor mensal da fatura
- `Total Charges` — total pago (armazenado como `object`, requer conversão)
- `Churn Score`, `CLTV` — scores proprietários da operadora *(removidos na modelagem por data leakage)*

**Categóricas (24):** dados demográficos, serviços contratados, tipo de contrato, método de pagamento e variáveis geográficas.

---

## Função `analisar_dataframe()`

Função utilitária definida no próprio notebook que executa **6 fases** de análise automaticamente:

| Fase | O que analisa |
|------|--------------|
| 1 — Visão geral | Shape, tipos de dados, estatísticas descritivas |
| 2 — Valores ausentes | Contagem e percentual por coluna; gráfico de barras |
| 3 — Distribuições numéricas | Histogramas com KDE + boxplots para todas as numéricas |
| 4 — Distribuições categóricas | Count plots para todas as categóricas |
| 5 — Correlação | Heatmap de correlação entre variáveis numéricas |
| 6 — Análise do target | Distribuição por valor, pizza de classes e contagem binária |

**Uso:**
```python
analisar_dataframe(
    df_churn,
    col_target='Churn Value',
    nome_dataset='Telco Customer Churn',
    ignorar_colunas=['CustomerID', 'Count', 'Churn Label']
)
```

---

## Principais Achados da EDA

### Desbalanceamento do Target

| Classe | Volume | Percentual |
|--------|--------|-----------|
| 0 — Não Churn | 5.174 | **73.5%** |
| 1 — Churn | 1.869 | **26.5%** |

O dataset é desbalanceado (~3:1), o que exige estratégias como `class_weight='balanced'`, SMOTE ou Focal Loss na modelagem.

### Variáveis Numéricas

- **`Tenure Months`**: distribuição bimodal — picos em 0-5 meses (novos clientes com alto risco) e ~70 meses (clientes fidelizados). Churners se concentram nos primeiros meses.
- **`Monthly Charges`**: churners têm faturas significativamente maiores. Mediana ~$80 vs ~$65 para retidos.
- **`Total Charges`**: alta correlação com `Tenure Months` (esperado). Armazenado como `object` no CSV — requer `pd.to_numeric(..., errors='coerce')`.
- **`Churn Score` / `CLTV`**: alta correlação com o target (são derivados do churn) — **data leakage**, removidos antes da modelagem.

### Variáveis Categóricas com Maior Poder Discriminativo

| Variável | Observação |
|----------|-----------|
| `Contract` | Month-to-month: ~43% de churn; Two-year: ~3% |
| `Internet Service` | Fiber optic: ~42% de churn; DSL: ~19%; No: ~7% |
| `Payment Method` | Electronic check: ~45% de churn (mais alto) |
| `Online Security` | Sem serviço: ~42% de churn; Com serviço: ~15% |
| `Tech Support` | Mesmo padrão que Online Security |
| `Senior Citizen` | Idosos: ~42% de churn; não idosos: ~24% |

---

## Análise Geográfica de Churn

### Top 20 cidades por taxa de churn (mínimo 10 clientes)

Cidades menores tendem a apresentar taxas extremas (ex: Amador City: 75%, Winters: 50%) por baixo volume. Entre as grandes cidades:

| Cidade | Total Clientes | Total Churners | % Churn |
|--------|---------------|---------------|---------|
| Los Angeles | 305 | 90 | 29.5% |
| San Diego | 150 | 50 | 33.3% |
| San Francisco | 104 | 31 | 29.8% |
| San Jose | 112 | 29 | 25.9% |
| Sacramento | 108 | 26 | 24.1% |

- **833 cidades únicas** identificadas no dataset.
- A análise usa `groupby('City').agg(['mean', 'count'])` com filtro `total >= 10` para remover ruído.

---

## Decisões Tomadas a Partir da EDA

| Decisão | Justificativa |
|---------|--------------|
| Remover `Churn Score`, `CLTV`, `Churn Label` | Data leakage — derivados diretamente do target |
| Remover colunas geográficas (`City`, `Country`, `Lat Long`, `Latitude`, `Longitude`, `Zip Code`) | Alta cardinalidade geográfica com pouco valor preditivo generalista |
| Manter `State` | Uma única categoria (California) — baixo impacto, mas mantida por consistência |
| Tratar `Total Charges` como numérico | Armazenado como string no CSV |
| Usar `class_weight='balanced'` ou técnicas de balanceamento | Desbalanceamento 73.5% / 26.5% |
| Criar features derivadas (Feature Engineering) | Interações identificadas nas distribuições (`Internet Service` + `Contract`) |

---

## Fluxo do Notebook

```
Leitura CSV
    ↓
Drop colunas inúteis (CustomerID, Count, Churn Reason)
    ↓
analisar_dataframe() — 6 fases de análise
    ↓
Análise geográfica por cidade
    ↓
→ Insights alimentam o notebook 02_feature_engineering.ipynb
```
