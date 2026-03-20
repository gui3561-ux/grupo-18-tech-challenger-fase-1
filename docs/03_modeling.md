# 03 — Modelagem, Comparação de Algoritmos e Registro MLflow

**Notebook:** `notebooks/03_modeling.ipynb`
**Dependências:** `utils/metrics.py`, `utils/neural_net.py`

---

## Objetivo

Treinar, otimizar e comparar 8 classificadores para predição de churn. Cada modelo é encapsulado em um Pipeline sklearn completo (pré-processamento + seleção de features + classificador), otimizado via `RandomizedSearchCV` e registrado no MLflow.

---

## Arquitetura dos Pipelines

Cada modelo segue a estrutura:

```
ColumnTransformer         → encoding + normalização (se necessário)
    ↓
SelectKBest(f_classif)    → seleção das top-k features
    ↓
Classificador             → modelo específico
```

O pipeline recebe o DataFrame bruto (`X_raw`) diretamente, sem necessidade de pré-filtrar colunas — o `ColumnTransformer` com `remainder='drop'` descarta automaticamente qualquer coluna não declarada em `num_cols`/`cat_cols`.

### Pipeline salvo em disco (`models/{nome}_pipeline.pkl`)

O arquivo `.pkl` é **self-contained**: inclui o `ColumnTransformer` ajustado (com os parâmetros do `StandardScaler` e os vocabulários do `OneHotEncoder`), o `SelectKBest` e o classificador. A API pode usar diretamente:

```python
import pickle
pipeline = pickle.load(open('models/lightgbm_pipeline.pkl', 'rb'))
proba = pipeline.predict_proba(X_raw)[:, 1]
```

---

## Modelos Treinados

### 1. Random Forest

| Preprocessor | `preprocessor_tree` (sem escala) |
|---|---|
| Classificador | `RandomForestClassifier(class_weight='balanced')` |

**Grid de hiperparâmetros:**
```python
{
    'selector__k':                   [20, 25, 30, 35, 40],
    'classifier__n_estimators':      [100, 200, 300, 500],
    'classifier__max_depth':         [None, 5, 10, 15, 20],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf':  [1, 2, 4],
    'classifier__max_features':      ['sqrt', 'log2'],
}
```

---

### 2. Gradient Boosting

| Preprocessor | `preprocessor_tree` |
|---|---|
| Classificador | `GradientBoostingClassifier` |

**Grid:**
```python
{
    'selector__k':               [20, 25, 30],
    'classifier__n_estimators':  [100, 200, 300, 500],
    'classifier__max_depth':     [3, 4, 5, 6],
    'classifier__learning_rate': [0.01, 0.05, 0.1, 0.15],
    'classifier__subsample':     [0.7, 0.8, 0.9, 1.0],
}
```

---

### 3. Logistic Regression

| Preprocessor | `preprocessor_scaled` (com StandardScaler) |
|---|---|
| Classificador | `LogisticRegression(max_iter=2000, class_weight='balanced')` |

**Grid:**
```python
{
    'selector__k':          [20, 25, 30],
    'classifier__C':        [0.001, 0.01, 0.1, 1, 10, 100],
    'classifier__penalty':  ['l1', 'l2', 'elasticnet'],
    'classifier__solver':   ['saga'],
    'classifier__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
}
```

---

### 4. KNN

| Preprocessor | `preprocessor_scaled` |
|---|---|
| Classificador | `KNeighborsClassifier` |

**Grid:**
```python
{
    'selector__k':             [20, 25, 30],
    'classifier__n_neighbors': [3, 5, 7, 9, 11, 15, 21],
    'classifier__weights':     ['uniform', 'distance'],
    'classifier__metric':      ['euclidean', 'manhattan'],
}
```

---

### 5. SVM

| Preprocessor | `preprocessor_scaled` |
|---|---|
| Classificador | `SVC(probability=True, class_weight='balanced')` |

**Grid:**
```python
{
    'selector__k':        [20, 25, 30],
    'classifier__C':      [0.1, 1, 10, 100],
    'classifier__kernel': ['rbf', 'linear', 'poly'],
    'classifier__gamma':  ['scale', 'auto', 0.01, 0.1],
}
```

---

### 6. XGBoost

| Preprocessor | `preprocessor_tree` |
|---|---|
| Classificador | `XGBClassifier(eval_metric='logloss')` |

**Grid:**
```python
{
    'selector__k':                  [20, 25, 30],
    'classifier__n_estimators':     [100, 200, 300],
    'classifier__max_depth':        [3, 4, 5, 6],
    'classifier__learning_rate':    [0.01, 0.05, 0.1],
    'classifier__subsample':        [0.7, 0.8, 0.9],
    'classifier__colsample_bytree': [0.7, 0.8, 0.9],
    'classifier__scale_pos_weight': [1, 2, 3],
}
```

---

### 7. LightGBM

| Preprocessor | `preprocessor_tree` |
|---|---|
| Classificador | `LGBMClassifier(class_weight='balanced')` |

**Grid:**
```python
{
    'selector__k':                  [20, 25, 30],
    'classifier__n_estimators':     [100, 200, 300],
    'classifier__num_leaves':       [31, 50, 63],
    'classifier__learning_rate':    [0.01, 0.05, 0.1],
    'classifier__subsample':        [0.7, 0.8, 0.9],
    'classifier__colsample_bytree': [0.7, 0.8, 0.9],
}
```

---

### 8. Neural Network (PyTorch)

| Preprocessor | `preprocessor_scaled` |
|---|---|
| Classe | `ChurnNetWrapper` (em `utils/neural_net.py`) |
| `n_jobs` | **1** (obrigatório) |

A classe vive em `utils/neural_net.py` (não no notebook) para que os workers do `joblib` consigam desserializar o Pipeline via pickle — classes definidas em células de notebook existem apenas no `__main__` do processo principal.

#### Arquitetura `ChurnNet`

```
Input (k features)
    ↓
Linear(k, 128) → BatchNorm1d(128) → ReLU → Dropout(p1)
    ↓
Linear(128, 64) → BatchNorm1d(64) → ReLU → Dropout(p2)
    ↓
Linear(64, 32) → ReLU
    ↓
Linear(32, 1)  → sigmoid (na inferência)
```

#### Estratégias de Treinamento

| Técnica | Detalhe |
|---------|---------|
| **Focal Loss** | Substitui BCE + pos_weight; foca nos exemplos difíceis (`gamma` ajustável) |
| **SMOTE** | Over-sampling da classe minoritária antes do treino |
| **Cosine Annealing + Warmup** | LR sobe linearmente nas primeiras `warmup_epochs`, depois decai via cosseno |
| **Gradient Clipping** | `clip_grad_norm_(max_norm=1.0)` — estabiliza o treino com Focal Loss |
| **Early Stopping** | Monitora ROC-AUC no split de validação interno (10%); restaura best state |
| **AdamW** | Otimizador com weight decay desacoplado |

#### Grid da Neural Network

```python
{
    'selector__k':              [25, 30, 35, 40],
    'classifier__lr':           [1e-3, 5e-4, 1e-4],
    'classifier__weight_decay': [1e-4, 1e-3],
    'classifier__dropout':      [[0.3, 0.2], [0.4, 0.3], [0.2, 0.1]],
    'classifier__focal_gamma':  [1.5, 2.0, 3.0],
}
```

**Melhores hiperparâmetros encontrados:**
```python
{
    'selector__k':              35,
    'classifier__lr':           1e-4,
    'classifier__weight_decay': 1e-3,
    'classifier__dropout':      [0.4, 0.3],
    'classifier__focal_gamma':  3.0,
}
```

> **Nota:** `n_jobs=1` é obrigatório para a Neural Network. O `RandomizedSearchCV` usa `multiprocessing` do `joblib` quando `n_jobs > 1`, o que conflita com os threads internos do PyTorch e causa deadlock.

---

## Função `comparar_modelos()`

Definida em `utils/metrics.py`. Orquestra o treinamento, avaliação, persistência e rastreamento MLflow de todos os modelos.

```python
df_comparativo = comparar_modelos(
    X_train_raw, X_test_raw, y_train, y_test,
    modelos=modelos_pipeline,
    n_iter=60,
    cv=5,
    salvar_modelos=True,
    models_dir='../models',
    experiment_name='churn_mvp',
    tracking_uri='../mlflow_tracking',
)
```

### Parâmetros principais

| Parâmetro | Valor | Descrição |
|-----------|-------|-----------|
| `n_iter` | 60 | Combinações testadas por modelo no RandomizedSearchCV |
| `cv` | 5 | Folds do StratifiedKFold |
| `random_state` | 42 | Reprodutibilidade |
| `salvar_modelos` | True | Persiste `.pkl` em `models_dir` |
| `experiment_name` | `churn_mvp` | Nome do experimento no MLflow |
| `tracking_uri` | `../mlflow_tracking` | URI do MLflow local |

### Por modelo o que acontece

1. `RandomizedSearchCV` com scorer customizado (bypassa verificação `is_regressor()` do sklearn 1.8+)
2. Avaliação do `best_estimator_` no conjunto de teste
3. Log no MLflow: métricas + hiperparâmetros + artefato `.pkl` + modelo sklearn
4. Print dos resultados no console

### Scorer customizado

```python
def _roc_auc_scorer(estimator, X, y):
    return roc_auc_score(y, estimator.predict_proba(X)[:, 1])
```

O sklearn 1.8 introduziu uma verificação que chama `is_regressor()` antes de `predict_proba`, quebrando pipelines com estimadores PyTorch. O scorer customizado bypassa essa verificação chamando `predict_proba` diretamente.

---

## Registro MLflow

### Estrutura de artefatos por run

```
mlflow_tracking/
└── churn_mvp/
    └── {run_id}/
        ├── params/          ← hiperparâmetros do best_estimator_
        ├── metrics/         ← accuracy, precision, recall, f1, roc_auc, cv_roc_auc
        ├── models/          ← {nome}_pipeline.pkl (pipeline completo self-contained)
        └── sklearn_model/   ← modelo no formato MLflow sklearn
```

### Métricas registradas

| Métrica | Descrição |
|---------|-----------|
| `accuracy` | Acurácia no conjunto de teste |
| `precision` | Precisão (classe positiva) |
| `recall` | Recall (classe positiva) |
| `f1` | F1-score |
| `roc_auc` | ROC-AUC no conjunto de teste |
| `cv_roc_auc` | Melhor ROC-AUC médio nos folds de validação cruzada |

---

## Baseline Inicial com Random Forest

Antes do `comparar_modelos`, o notebook treina um **RF baseline** sem tuning para:
- Validar o pipeline end-to-end
- Gerar feature importances para referência
- Visualizar a matriz de confusão

```python
pipe_rf_baseline = SKPipeline([
    ('preprocessor', preprocessor_tree),
    ('classifier',   RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
])
```

---

## Script Standalone para Neural Network

O arquivo `train_nn.py` na raiz do projeto permite iterar na Neural Network **isoladamente**, sem executar o loop completo dos 8 modelos:

```bash
python train_nn.py
```

Ou de dentro do notebook:
```python
%run ../train_nn.py
```

Útil para experimentos rápidos de arquitetura e hiperparâmetros antes de integrar ao `comparar_modelos`.

---

## Fluxo do Notebook

```
Importações (sklearn, xgboost, lightgbm, torch)
    ↓
Leitura do CSV + Feature Engineering
    ↓
Remoção de leakage e colunas geográficas
    ↓
Split treino/teste (80/20 estratificado)
    ↓
Definição de preprocessor_tree e preprocessor_scaled
    ↓
RF Baseline (treino rápido sem tuning)
    ↓
Definição dos 8 Pipelines + grids (modelos_pipeline)
    ↓
comparar_modelos() → RandomizedSearchCV por modelo
    ↓
Tabela comparativa (ROC-AUC, F1, Recall, Precision, Accuracy)
    ↓
Artefatos salvos: models/*.pkl + mlflow_tracking/
```
