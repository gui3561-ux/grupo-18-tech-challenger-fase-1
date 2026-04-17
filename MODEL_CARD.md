# Model Card — Predição de Churn (Telecom)

| Campo | Valor |
|-------|--------|
| **Versão do documento** | 1.1 |
| **Última atualização** | Abril/2026 |
| **Equipe** | Grupo 18 — Tech Challenge Fase 1 (FIAP Pós Tech) |
| **Repositório** | https://github.com/gui3561-ux/grupo-18-tech-challenger-fase-1 |
| **API de produção (referência)** | https://churn-prediction-api.azurewebsites.net |
| **Artefato em produção** | `models/neural_network_pipeline.pkl` |

Este *Model Card* segue o espírito de transparência de [Mitchell et al., 2019](https://arxiv.org/abs/1810.03993) adaptado ao escopo acadêmico do desafio: descreve o **modelo**, os **dados**, a **performance**, **limitações**, **equidade**, **riscos** e **operacionalização** (API, monitoramento).

---

## 1. Visão geral

### 1.1 Identificação do modelo

| Atributo | Valor |
|----------|--------|
| **Nome interno** | Neural Network — pipeline `neural_network` |
| **Família** | Multi-Layer Perceptron (classificação binária) |
| **Frameworks** | **PyTorch** (módulo neural), **scikit-learn** (`Pipeline`, pré-processamento, métricas) |
| **Arquitetura (camadas densas)** | 128 → 64 → 32 → 1 logit; **BatchNorm** e **Dropout** nas camadas ocultas |
| **Serialização** | `pickle` (objeto `sklearn.pipeline.Pipeline`) |
| **Caminho padrão** | `models/neural_network_pipeline.pkl` |
| **Rastreio de experimentos** | MLflow (experimento histórico `churn_mvp` / runs sob `mlflow_tracking/`) |
| **Versão da API exposta** | Campo `model` na resposta JSON: ex. `"neural_network"` |

### 1.2 Propósito

Estimar **P(churn)** — probabilidade de o cliente **cancelar** o serviço em um horizonte alinhado ao *label* do dataset original (comportamento histórico observado no *snapshot* IBM Telco). Uso pretendido:

- Priorização para **retenção** e campanhas de **marketing**
- Apoio à **gestão** de risco de base (não substitui política comercial ou jurídica)

### 1.3 Não é destinado a

- Decisões **totalmente automatizadas** sem revisão humana (crédito, rescisão de contrato, preços individualizados sensíveis)
- **Outras indústrias** sem novo treino e validação
- **Conformidade regulatória** sem revisão legal (LGPD/GDPR exigem base legal, DPIA, etc.)

---

## 2. Dados de treinamento e inferência

### 2.1 Dataset de referência (treino)

| Campo | Descrição |
|-------|-----------|
| **Nome comum** | Telco Customer Churn |
| **Origem** | Conjunto público amplamente usado (IBM / Kaggle) |
| **Tamanho típico** | **7.043** linhas |
| **Granularidade** | Um registro por cliente (snapshot) |
| **Target** | Binário — churn sim/não |

**Arquivo no repositório (quando presente):** `data/Telco_customer_churn.csv` (ou equivalente versionado via DVC — ver política do grupo).

### 2.2 Distribuição do target (ordem de grandeza)

| Classe | Proporção aproximada |
|--------|----------------------|
| Não churn (0) | ~81% |
| Churn (1) | ~19% |

*Desbalanceamento tratado no treino com **SMOTE** (pipeline) e **Focal Loss** (γ = 3,0).*

### 2.3 *Leakage* e higiene de dados

Colunas com **vazamento de informação** em relação ao alvo foram removidas no pipeline de treino, por exemplo:

- `Churn Label`, `Churn Score`, `CLTV`, `Churn Reason` (quando existirem no *raw*)

Valores ausentes pontuais (ex.: `Total Charges`) foram tratados (ex.: mediana) no *notebook* / *pipeline* de treino.

### 2.4 Cobertura geográfica e representatividade

O dataset IBM Telco refere-se a clientes de **telecom nos EUA**, com coluna de **estado** (vários estados). **Não** restringir mentalmente o domínio a um único estado: o modelo foi treinado em **mistura de estados**; ainda assim, **não há garantia** de generalização para outros países, regulamentos ou ofertas de produto.

### 2.5 Features utilizadas pelo modelo (após engenharia)

#### Numéricas / derivadas (exemplos)

- `Tenure Months`, `Monthly Charges`, `Total Charges`
- `high_risk_profile` — fibra + contrato *month-to-month*
- `isolated_senior` — idoso sem parceiro/dependentes
- `internet_services_count` — contagem de add-ons ativos
- `cost_per_month` — `Monthly Charges / (Tenure Months + 1)`

#### Categóricas (alto nível)

Estado, gênero, serviços, tipo de internet, contrato, pagamento, etc. — codificadas no *pipeline* (ex.: *One-Hot*).

**Seleção de features:** `SelectKBest` com **k = 35** (hiperparâmetro do melhor run).

### 2.6 Features na API (entrada HTTP)

A API recebe **nomes em snake_case** (`tenure_months`, `payment_method`, …). O serviço monta um `DataFrame` com os **nomes de coluna do dataset** (`Tenure Months`, `Payment Method`, …) e aplica a mesma engenharia que no treino antes de `predict_proba`. Ver [`src/services/inference_service.py`](src/services/inference_service.py).

---

## 3. Metodologia de treino

### 3.1 Pré-processamento (dentro do `Pipeline`)

- **Escalonamento:** `StandardScaler` onde aplicável à rede
- **Encoding:** variáveis categóricas tratadas no *pipeline* (ex.: *one-hot*)
- **Seleção:** `SelectKBest`, **k = 35**
- **Balanceamento:** **SMOTE** (via `imbalanced-learn`, quando habilitado no classificador)

### 3.2 Arquitetura neural (detalhe)

```
Entrada (35 features após seleção)
  → Linear(35→128) + BatchNorm + ReLU + Dropout(0,4)
  → Linear(128→64) + BatchNorm + ReLU + Dropout(0,3)
  → Linear(64→32) + ReLU
  → Linear(32→1)  [logits]
  → sigmoid na inferência (probabilidade classe positiva)
```

### 3.3 Otimização e regularização

| Item | Valor / nota |
|------|----------------|
| **Otimizador** | AdamW |
| **Learning rate** | 0,0001 (ver hiperparâmetros) |
| **Weight decay** | 0,0001 |
| **Loss** | Focal Loss (γ = 3,0) para foco na classe minoritária |
| **Scheduler** | Cosseno com *warmup* linear (~10 épocas) |
| **Early stopping** | Monitorando ROC-AUC em validação (*patience* 50 épocas) |
| **Gradient clipping** | `max_norm = 1,0` |

### 3.4 Hiperparâmetros registrados (referência MLflow)

Valores típicos do melhor modelo (podem variar levemente entre commits de treino):

```json
{
  "selector__k": 35,
  "classifier__lr": 0.0001,
  "classifier__weight_decay": 0.0001,
  "classifier__dropout": [0.4, 0.3],
  "classifier__focal_gamma": 3.0,
  "classifier__epochs": 500,
  "classifier__batch_size": 64,
  "classifier__patience": 50,
  "classifier__warmup_epochs": 10,
  "classifier__val_fraction": 0.1,
  "classifier__use_focal_loss": true,
  "classifier__use_smote": true
}
```

---

## 4. Performance offline

### 4.1 Métricas principais

| Métrica | Valor | Notas |
|---------|-------|--------|
| **ROC-AUC (teste / holdout)** | **0,8464** | Métrica principal de *ranking* |
| **ROC-AUC (5-fold CV)** | **0,8541** | Estabilidade entre *folds* |
| **Accuracy** | ~0,80–0,81 | Depende do threshold 0,5 |
| **Precision (classe churn)** | ~0,66–0,68 | Estimativa |
| **Recall (classe churn)** | ~0,51–0,53 | Estimativa — **metade dos churns** pode escapar |
| **F1** | ~0,58–0,60 | Estimativa |

Valores de precisão/recall/F1 são **ordens de grandeza**; o *threshold* de negócio pode diferir de 0,5.

### 4.2 Matriz de confusão (ilustrativa, threshold 0,5)

Ordem de grandeza no conjunto de teste (não substitui matriz oficial exportada do notebook):

|  | Predito não churn | Predito churn |
|--|-------------------|---------------|
| **Real não churn** | TN alto | FP moderado |
| **Real churn** | **FN alto** (recall ~0,52) | TP moderado |

### 4.3 Thresholds de negócio vs modelo

| Uso | Threshold |
|-----|-----------|
| **Rótulo binário na API** (`churn_prediction`) | **≥ 0,5** sobre `churn_probability` |
| **Faixas de risco** (apenas orientação; configurável via env) | Baixo `< 0,3`; médio `[0,3, 0,6)`; alto `≥ 0,6` |

Constantes em `src/core/config.py`: `risk_threshold_low`, `risk_threshold_medium`.

---

## 5. Serviço online (inferência)

### 5.1 Endpoint

- **Método / URL:** `POST /api/v1/inference/predict`
- **Base produção:** `https://churn-prediction-api.azurewebsites.net`

### 5.2 Contrato de saída

```json
{
  "churn_probability": 0.7234,
  "churn_prediction": true,
  "model": "neural_network"
}
```

- `churn_probability` arredondada a **4 casas** no código de serviço
- `churn_prediction` é **estritamente** derivada do threshold **0,5** na probabilidade

### 5.3 Disponibilidade e degradação

- `GET /api/v1/health` retorna `status: "ok"` se o modelo foi carregado; caso contrário `"degraded"` e `model_loaded: false` (API pode responder, mas **não** use predições para decisões críticas).

### 5.4 Latência e escala

- Histograma Prometheus: `model_inference_seconds`
- **Gunicorn** com 2 *workers* no Azure; latência depende de CPU e carga
- Para *throughput* alto, avaliar mais réplicas no App Service ou filas assíncronas

### 5.5 Dependências de runtime (imagem Docker)

Principais pacotes em [`requirements.txt`](requirements.txt): `torch` (CPU), `scikit-learn`, `pandas`, `numpy`, `fastapi`, `uvicorn`, `gunicorn`, `prometheus-client`, `structlog`, etc.

Desenvolvimento e treino completos: ver [`pyproject.toml`](pyproject.toml) (Jupyter, MLflow, XGBoost, LightGBM, DVC, …).

---

## 6. Monitoramento em produção

### 6.1 Métricas recomendadas

| Sinal | Onde | Ação sugerida |
|-------|------|----------------|
| **Taxa de 5xx** | `http_requests_total` + logs | Alerta se > 1% |
| **Latência p95 inferência** | `model_inference_seconds` | Investigar CPU / cold start |
| **Distribuição de scores** | `churn_probability_histogram` | Drift de entrada |
| **Churn previsto vs base** | `churn_predictions_total` | Comparar com taxa real (requer *feedback* de negócio) |
| **Modelo carregado** | `model_loaded` | Alerta se 0 |

### 6.2 Drift e retreino

- **Data drift:** mudança nas distribuições de *features* (KS, PSI por variável)
- **Concept drift:** relação feature→churn muda (ex.: nova oferta agressiva de concorrente)
- **Gatilhos de retreino sugeridos:** queda sustentada de ROC-AUC em validação periódica; drift forte em variáveis-chave; mudança de produto

Frequência sugerida de revisão: **3–6 meses** ou após eventos de mercado relevantes.

---

## 7. Limitações técnicas

1. **Tamanho e estática da base:** ~7k linhas; *snapshot* único — não captura séries temporais longas.
2. **Generalização geográfica e de produto:** EUA / *telco* histórico; outros mercados exigem novo treino.
3. **Desbalanceamento:** recall moderado — muitos churns **não** detectados com threshold 0,5.
4. **Split aleatório:** se o treino usou divisão i.i.d., performance pode ser otimista vs. validação **temporal** (*walk-forward*).
5. **Pickle:** carregar apenas de fonte confiável (mesmo repositório / artefato assinado); risco teórico de execução arbitrária se o arquivo for adulterado.
6. **Custo computacional:** PyTorch + dependências são pesados vs. modelos só árvores; justificado pela ROC-AUC obtida no desafio.

---

## 8. Limitações de negócio

- Sem **NPS**, **tickets** de suporte ou **uso real** de rede (GB/minutos) — variáveis fortes em churn real.
- **Ofertas e preços** futuros não estão no histórico.
- Uso indevido para **discriminar** grupos protegidos viola políticas éticas e legais.

---

## 9. Equidade e vieses

### 9.1 Variáveis sensíveis presentes

Gênero, idade (*senior*), local (*state*) aparecem como *features*. O modelo pode **correlacionar** esses campos com churn de forma **espúria** ou **socialmente sensível**.

### 9.2 Boas práticas

- Monitorar **taxas de predição positiva** e **taxa de erro** por segmento (quando houver *ground truth* operacional)
- Não usar a saída como único critério para **tratamento desigual** de clientes
- Combinar com **valor do cliente** (receita, margem) em processos de decisão — *não* implementado neste MVP

### 9.3 EDA (resumo)

- Contratos *month-to-month* e fibra costumam concentrar churn no dataset original
- Idosos isolados são *feature* explícita (`isolated_senior`) — revisar impacto em políticas de retenção

---

## 10. Riscos éticos e conformidade

| Risco | Mitigação |
|-------|-----------|
| **Privacidade** | Não enviar PII desnecessária à API; em produção real, pseudonimização e base legal (LGPD) |
| **Transparência** | *Model card* + documentação de API; para explicabilidade por cliente, considerar SHAP em modelo substituto ou *surrogate* |
| **Automação** | Manter **humano no loop** para ofertas financeiras relevantes |

---

## 11. Cenários de falha

| Cenário | Efeito |
|---------|--------|
| **Cliente novo** (*tenure* muito baixo) | `cost_per_month` instável; maior incerteza |
| **Categoria não vista** no treino | *One-hot* pode gerar vetor inesperado — validação Pydantic reduz, mas combinações raras existem |
| **Mudança macro** (crise, fusão de operadoras) | Performance degrada até retreino |
| **Falso positivo** | Custo de campanha / irritação do cliente |
| **Falso negativo** | Perda de receita por churn não evitado |

---

## 12. Manutenção e governança

| Artefato | Responsabilidade |
|----------|-------------------|
| Código e `Dockerfile` | Repositório Git + PRs |
| Modelo `.pkl` | Versionar por tag/commit; idealmente **MLflow Model Registry** em cenário corporativo |
| Dados | DVC / *lake* (fora do escopo mínimo deste MVP) |
| **Model Card** | Atualizar a cada retreino material ou mudança de métricas alvo |

---

## 13. Como carregar o modelo offline (Python)

```python
import pickle
from pathlib import Path

path = Path("models/neural_network_pipeline.pkl")
with path.open("rb") as f:
    pipeline = pickle.load(f)

# DataFrame com colunas do dataset + features derivadas
# (replicar feature engineering do serviço ou usar o mesmo código)
proba = pipeline.predict_proba(X)[0, 1]
```

Para *feature engineering* idêntico ao serviço, reutilize `ChurnInferenceService.__feature_engineering` ou os notebooks em `notebooks/`.

---

## 14. Referências

1. **Dataset:** IBM / Kaggle — [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
2. **Focal Loss:** Lin et al., 2017 — *Focal Loss for Dense Object Detection*  
3. **SMOTE:** Chawla et al., 2002  
4. **Model Cards:** Mitchell et al., 2019 — *Model Cards for Model Reporting*  
5. **Código da rede:** [`utils/neural_net.py`](utils/neural_net.py)  
6. **Documentação da API (usuário):** [README.md](README.md)  

---

## 15. Histórico de versões do documento

| Versão | Data | Alterações |
|--------|------|------------|
| 1.0 | 2026-03 | Versão inicial (Neural Network, ROC-AUC teste 0,8464 / CV 0,8541) |
| 1.1 | 2026-04 | Ampliação operacional (API, health, deploy, métricas, correções de URL e escopo geográfico); alinhamento ao repositório `gui3561-ux` |

---

*Este Model Card deve ser revisado após cada retreino significativo, mudança de dados ou alteração de métricas de negócio.*
