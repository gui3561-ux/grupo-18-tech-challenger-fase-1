# 02 — Feature Engineering, Pré-processamento e Seleção de Features

**Notebook:** `notebooks/02_feature_engineering.ipynb`
**Dependência:** insights do `01_eda.ipynb`

---

## Objetivo

Construir o conjunto de features definitivo para a modelagem: criação de variáveis derivadas, limpeza dos dados, definição dos pipelines de pré-processamento e análise comparativa de métodos de seleção de features.

---

## 1. Feature Engineering

Quatro novas features criadas a partir de combinações de colunas existentes:

### `high_risk_profile`
```python
df_churn['high_risk_profile'] = (
    (df_churn['Internet Service'] == 'Fiber optic') &
    (df_churn['Contract'] == 'Month-to-month')
).astype(int)
```
- **Lógica:** clientes com internet Fiber optic em contrato mensal são o segmento de maior risco identificado na EDA.
- **Churn rate:** ~54.6% (vs. 26.5% médio geral).

### `isolated_senior`
```python
df_churn['isolated_senior'] = (
    (df_churn['Senior Citizen'] == 'Yes') &
    (df_churn['Partner'] == 'No') &
    (df_churn['Dependents'] == 'No')
).astype(int)
```
- **Lógica:** idosos sem parceiro e sem dependentes têm menor resistência ao cancelamento.
- **Churn rate:** ~48.9%.

### `internet_services_count`
```python
servicos = ['Online Security', 'Online Backup', 'Device Protection',
            'Tech Support', 'Streaming TV', 'Streaming Movies']
df_churn['internet_services_count'] = sum(
    (df_churn[c] == 'Yes').astype(int) for c in servicos
)
```
- **Lógica:** quanto mais serviços contratados, maior o custo de troca (switching cost) e menor o churn.
- **Comportamento:** relação inversamente proporcional com a taxa de churn.

### `cost_per_month`
```python
df_churn['cost_per_month'] = df_churn['Monthly Charges'] / (df_churn['Tenure Months'] + 1)
```
- **Lógica:** normaliza o custo mensal pelo tempo de relacionamento. Novos clientes pagando muito têm custo percebido alto.
- **Comportamento:** churners apresentam valor ~3.2x maior que clientes retidos.

---

## 2. Pré-processamento

### Conversão de `Total Charges`
```python
df_churn["Total Charges"] = pd.to_numeric(df_churn["Total Charges"], errors='coerce')
df_churn['Total Charges'] = df_churn['Total Charges'].fillna(df_churn['Total Charges'].median())
```
- O CSV armazena `Total Charges` como `object` (string), com valores vazios representados como espaços.
- `errors='coerce'` converte inválidos para `NaN`; imputados com a mediana (R$ 1.397,47).

### Remoção de Data Leakage
```python
colunas_vazar = ['Churn Score', 'CLTV', 'Churn Label']
```
- **`Churn Score`**: score proprietário calculado após o evento de churn.
- **`CLTV`** (Customer Lifetime Value): calculado com base no churn histórico.
- **`Churn Label`**: versão textual do target (`Yes`/`No`).

### Remoção de Colunas Geográficas
```python
colunas_geograficas = ['City', 'Country', 'Lat Long', 'Latitude', 'Longitude', 'Zip Code']
```
Alta cardinalidade geográfica sem generalização útil para um modelo de produção.

### Shape Final
| Etapa | Shape |
|-------|-------|
| Após leitura e drops iniciais | 7.043 × 30 |
| Após feature engineering | 7.043 × 34 |
| Após remoção de leakage + geo | 7.043 × 25 |

---

## 3. Divisão Treino / Teste

```python
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.2, random_state=42, stratify=y
)
```

| Conjunto | Linhas | Features |
|----------|--------|---------|
| Treino | 5.634 | 24 |
| Teste | 1.409 | 24 |

- **`stratify=y`** garante que a proporção de churn (~26.5%) seja mantida em ambos os conjuntos.
- **`random_state=42`** assegura reprodutibilidade.
- Distribuição no treino: `{0: 4139, 1: 1495}`.

### Features por Tipo (após remoção de leakage)

**Numéricas (7):**
`Tenure Months`, `Monthly Charges`, `Total Charges`, `high_risk_profile`, `isolated_senior`, `internet_services_count`, `cost_per_month`

**Categóricas (17):**
`State`, `Gender`, `Senior Citizen`, `Partner`, `Dependents`, `Phone Service`, `Multiple Lines`, `Internet Service`, `Online Security`, `Online Backup`, `Device Protection`, `Tech Support`, `Streaming TV`, `Streaming Movies`, `Contract`, `Paperless Billing`, `Payment Method`

---

## 4. Pipelines de Pré-processamento

O encoding e a normalização são feitos **dentro dos Pipelines sklearn**, não antes do split. Isso elimina o data leakage que ocorreria ao aplicar `pd.get_dummies()` ou `StandardScaler` no dataset completo.

### Bloco Numérico — Tree-based (sem escala)
```python
num_passthrough = SKPipeline([
    ('imputer', SimpleImputer(strategy='median')),
])
```
Usado em: Random Forest, Gradient Boosting, XGBoost, LightGBM.

### Bloco Numérico — Modelos Sensíveis à Escala
```python
num_scaled = SKPipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler()),
])
```
Usado em: Logistic Regression, KNN, SVM, Neural Network.

### Bloco Categórico (igual para todos os modelos)
```python
cat_pipe = SKPipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot',  OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
])
```
- `handle_unknown='ignore'`: categorias inéditas no teste (ex: nova cidade) não geram erro.
- `sparse_output=False`: retorna matriz densa, compatível com PyTorch e sklearn.

### ColumnTransformers

```python
# Para modelos tree-based
preprocessor_tree = ColumnTransformer([
    ('num', num_passthrough, num_cols),
    ('cat', cat_pipe,        cat_cols),
], remainder='drop')

# Para modelos sensíveis à escala
preprocessor_scaled = ColumnTransformer([
    ('num', num_scaled, num_cols),
    ('cat', cat_pipe,   cat_cols),
], remainder='drop')
```

- **`remainder='drop'`**: colunas não listadas em `num_cols`/`cat_cols` são descartadas automaticamente — o pipeline aceita o DataFrame bruto sem filtro manual de colunas.

---

## 5. Seleção de Features

### Método Principal: `SelectKBest(f_classif, k=30)`

Aplicado dentro dos pipelines de modelagem após o `ColumnTransformer`. Seleciona as 30 features com maior F-score (ANOVA) em relação ao target binário.

### Análise Comparativa — `analisar_features()`

Função importada de `utils/feature_selection.py` que compara **7 métodos** e gera um ranking agregado:

| # | Método | Tipo | Abordagem |
|---|--------|------|-----------|
| 1 | SelectKBest (ANOVA F-score) | Filter | Estatístico |
| 2 | SelectKBest (Chi-quadrado) | Filter | Estatístico |
| 3 | Informação Mútua | Filter | Não-linear |
| 4 | Correlação de Pearson | Filter | Linear |
| 5 | Random Forest Feature Importance | Embedded | Baseado em árvore |
| 6 | RFE com Regressão Logística | Wrapper | Iterativo |
| 7 | SelectFromModel (RF, limiar=média) | Embedded | Threshold |

O **ranking agregado** normaliza os scores de cada método para uma escala comum e combina as posições, produzindo uma lista estável das top-K features independente do método escolhido.

```python
top_k, ranking_agregado = analisar_features(
    X_tr_enc, y_train,
    k=30,
    rf_importances=rf_imp,
    plot=True,
)
```

### Conclusão da Análise

O `SelectKBest(f_classif, k=30)` dentro dos Pipelines captura automaticamente as features mais relevantes confirmadas pelo ranking agregado. O valor `k=30` (dos ~45-50 features após encoding) foi validado como ponto de equilíbrio entre expressividade e ruído.

---

## Fluxo do Notebook

```
Leitura CSV
    ↓
Drop colunas inúteis
    ↓
Feature Engineering (4 novas features)
    ↓
Limpeza: Total Charges → numérico, fillna(mediana)
    ↓
Remoção de leakage + colunas geográficas
    ↓
Split treino/teste estratificado (80/20)
    ↓
Definição de preprocessor_tree e preprocessor_scaled
    ↓
Análise de features com 7 métodos → ranking agregado
    ↓
→ preprocessors e splits exportados para 03_modeling.ipynb
```
