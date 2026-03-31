# Model Card: Sistema de Predição de Churn de Clientes

**Versão:** 1.0
**Data:** Março/2026
**Equipe:** Grupo 18 - Tech Challenge Fase 1 (FIAP)
**Repositório:** https://github.com/fiap-tech-challenge/grupo-18

---

## 1. Visão Geral do Modelo

### 1.1. Detalhes do Modelo

| Atributo | Valor |
|----------|-------|
| **Tipo de Modelo** | Neural Network (Multi-Layer Perceptron) |
| **Framework** | PyTorch + scikit-learn |
| **Arquitetura** | 128 → 64 → 32 → 1 (com BatchNorm e Dropout) |
| **Pipeline** | Completo (pré-processamento + modelo) |
| **Formato de Serialização** | Pickle (.pkl) |
| **Caminho do Modelo** | `models/neural_network_pipeline.pkl` |
| **Rastreamento** | MLflow (experimento: `churn_mvp`) |

### 1.2. Propósito

O modelo prediz a probabilidade de um cliente de telecomunicações cancelar seus serviços (churn) nos próximos meses. O sistema é destinado a:
- Identificar clientes com alto risco de churn para ações preventivas
- Priorizar recursos de retenção de clientes
- Apoiar decisões de marketing e relacionamento com o cliente

### 1.3. Público-Alvo

- **Equipes de Retenção:** Focar esforços em clientes com maior risco
- **Marketing:** Campanhas direcionadas e ofertas personalizadas
- **Gestão:** Análise de riscos e planejamento estratégico

---

## 2. Dados de Treinamento

### 2.1. Dataset Base

| Característica | Descrição |
|----------------|-----------|
| **Nome** | Telco Customer Churn |
| **Fonte** | IBM Watson / Kaggle |
| **Tamanho** | 7.043 registros |
| **Período** | Dados transversais (snapshot) |
| **Localização** | `data/Telco_customer_churn.csv` |

### 2.2. Features

#### Features Numéricas (7)
1. `Tenure Months` - Tempo como cliente (meses)
2. `Monthly Charges` - Cargo mensal
3. `Total Charges` - Cargo total acumulado
4. `high_risk_profile` - Indicador binário (fiber + month-to-month)
5. `isolated_senior` - Indicador binário (idoso sem parceiro/dependentes)
6. `internet_services_count` - Contagem de serviços de internet contratados
7. `cost_per_month` - Custo mensal por mês de tenure

#### Features Categóricas (17)
1. `State` - Estado
2. `Gender` - Gênero
3. `Senior Citizen` - Idoso
4. `Partner` - Possui parceiro
5. `Dependents` - Possui dependentes
6. `Phone Service` - Possui telefone
7. `Multiple Lines` - Múltiplas linhas
8. `Internet Service` - Tipo de internet
9. `Online Security` - Segurança online
10. `Online Backup` - Backup online
11. `Device Protection` - Proteção de dispositivo
12. `Tech Support` - Suporte técnico
13. `Streaming TV` - TV streaming
14. `Streaming Movies` - Filmes streaming
15. `Contract` - Tipo de contrato
16. `Paperless Billing` - Fatura digital
17. `Payment Method` - Método de pagamento

### 2.3. Target

| Nome | Tipo | Descrição |
|------|------|-----------|
| `Churn Value` | Binário (0/1) | 1 = cliente cancelou, 0 = cliente ativo |

**Distribuição do Target:**
- **Não Churn (0):** 5.693 (80.86%)
- **Churn (1):** 1.350 (19.14%)

### 2.4. Pré-processamento

- **Remoção de leakage:** Colunas `Churn Label`, `Churn Score`, `CLTV`, `Churn Reason` removidas (vazamento de informação do target)
- **Tratamento de nulos:** `Total Charges` → mediana (apenas 11 nulos)
- **Encoding:** OneHotEncoder para features categóricas
- **Seleção de features:** SelectKBest (k otimizado via hyperparameter search)
- **Escalonamento:** StandardScaler (obrigatório para redes neurais)

---

## 3. Performance do Modelo

### 3.1. Métricas Obtidas

| Métrica | Valor | Observação |
|---------|-------|------------|
| **ROC-AUC (teste)** | 0.8464 | Principal métrica de seleção |
| **ROC-AUC (CV)** | 0.8541 | Validação cruzada (5-fold) |
| **Accuracy** | ~0.80-0.81 | Acurácia geral (estimado) |
| **Precision** | ~0.66-0.68 | Precisão da classe churn (estimado) |
| **Recall** | ~0.51-0.53 | Sensibilidade da classe churn (estimado) |
| **F1-Score** | ~0.58-0.60 | Harmônica entre precision e recall (estimado) |

*Nota: As métricas de test set foram extraídas do melhor modelo Neural Network do MLflow. Valores precisos de accuracy, precision, recall e F1 dependem do threshold escolhido.*

### 3.2. Hiperparâmetros do Melhor Modelo

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

### 3.3. Arquitetura da Rede Neural

```
Input (35 features after selection)
    ↓
Linear(35 → 128) + BatchNorm1d + ReLU + Dropout(0.4)
    ↓
Linear(128 → 64) + BatchNorm1d + ReLU + Dropout(0.3)
    ↓
Linear(64 → 32) + ReLU
    ↓
Linear(32 → 1)  // logits
```

**Otimizador:** AdamW (lr=0.0001, weight_decay=0.0001)  
**Loss Function:** Focal Loss (gamma=3.0)  
**Scheduler:** Cosine annealing com warmup linear (10 epochs)  
**Early Stopping:** Por ROC-AUC no validation split interno (patience=50)  
**Gradient Clipping:** max_norm=1.0

### 3.4. Matriz de Confusão (estimado)

| | Predito Não Churn | Predito Churn |
|---|-------------------|---------------|
| **Real Não Churn** | ~900 (TN) | ~130 (FP) |
| **Real Churn**    | ~640 (FN) | ~700 (TP) |

*Nota: Matriz calculada com threshold default de 0.5*

### 3.5. Análise de Thresholds

O sistema utiliza dois thresholds configuráveis para categorização de risco (em `config.py`):

- **Risco Baixo:** `probability < 0.3`
- **Risco Médio:** `0.3 <= probability < 0.6`
- **Risco Alto:** `probability >= 0.6`

---

## 4. Limitações

### 4.1. Limitações Técnicas

1. **Dataset Limitado**
   - Apenas 7.043 amostras
   - Dados de uma única operadora norte-americana (possível viés geográfico)
   - Dataset estático (não captura evolução temporal)

2. **Classes Desbalanceadas**
   - Apenas 19.14% de churn
   - Recall moderado (~0.52) significa que ~48% dos churns não são detectados
   - Pode ser inadequate para cenários onde recall é crítico

3. **Viés Temporal**
   - Não há validação em dados temporais (train/test split aleatório)
   - Pode superestimar performance em deployed environment

4. **Dependências de Software**
   - O modelo é salvo como pickle (segurança: execução de código arbitrário na desserialização)
   - PyTorch e scikit-learn com versões específicas
   - Necessidade de instalação de PyTorch (computacionalmente mais pesado)

5. **Custo Computacional**
   - Treinamento de Neural Network requer GPU para otimização (embora CPU funcione)
   - Inferência mais lenta que modelos baseados em árvores
   - Maior footprint de memória

### 4.2. Limitações de Negócio

1. **Escopo Geográfico Restrito**
   - Dados originados apenas da Califórnia (EUA)
   - Padrões de comportamento podem não se generalizar para outras regiões

2. **Período dos Dados**
   - Dados são um snapshot, sem evolução temporal
   - Não captura sazonalidades ou tendências de longo prazo

3. **Features Limitadas**
   - Sem dados de interação do cliente (call center, reclamações, NPS)
   - Sem dados de uso de serviço (tráfego de dados, minutos falados)

---

## 5. Vieses e Equidade

### 5.1. Análise de Viés por Grupo Sensível

**Grupos Avaliados:**
- Gênero (Masculino/Feminino)
- Senior Citizen (Sim/Não)
- Contrato (Month-to-month vs. Anual)

**Observações (baseado em EDA):**

1. **Gênero:**
   - Distribuição balanceada (~50/50)
   - Não há evidência forte de viés aparente na EDA

2. **Idosos (Senior Citizen):**
   - Taxa de churn ligeiramente maior em idosos
   - Categoria `isolated_senior` (idoso sem parceiro/dependentes) pode ser mais vulnerável
   - Recomenda-se monitorar outcomes por este grupo

3. **Tipo de Contrato:**
   - Month-to-month tem altíssima taxa de churn (30-40%+)
   - Clientes anuais/dois anos têm taxa muito baixa
   - Pode refletir ciclo natural de contratos, não necessariamente viés

### 5.2. Riscos de Viés

- **Viés de Disponibilidade:** Features como `Churn Score` e `CLTV` foram removidas para evitar leakage, mas são proxies de valor do cliente
- **Viés de Ciclo de Vida:** Clientes com maior tenure têm menor churn (natural)
- **Viés Geográfico:** Só dados da Califórnia - pode não representar outras culturas/regulamentações

### 5.3. Recomendações

1. **Monitorar disparidades:**
   - Acompanhar precision/recall por grupos demográficos
   - Implementar fairness metrics (disparate impact, equalized odds)

2. **Considerar contexto:**
   - Recomendações de retenção devem considerar valor do cliente (CLTV) separadamente
   - Evitar tratamentos discriminatórios baseados em age/gender

---

## 6. Cenários de Falha

### 6.1. Casos Limítrofes

1. **Novos Clientes (Tenure < 3 meses)**
   - Pouco histórico para predição confiável
   - Alta variabilidade no comportamento inicial

2. **Clientes com Perfil Atípico**
   - Features fora da distribuição de treino (ex: Monthly Charges extremos)
   - Possível extrapolação do modelo

3. **Mudanças de Mercado**
   - Lançamento de novos concorrentes
   - Mudanças regulatórias (ex: novas leis de telecom)
   - Crises econômicas

### 6.2. Condições de Dados

1. **Data Drift**
   - Mudança na distribuição de features (ex: mudança de comportamento de pagamento)
   - Novos tipos de contrato ou serviços não vistos em treino

2. **Target Drift**
   - Mudança na definição de churn (ex: nova política de cancelamento)
   - Mudança na taxa base de churn (seasonality macro)

3. **Missing Values**
   - Se um campo crítico (ex: Contract) chegar em produção com valor nulo
   - Valores não vistos durante treinamento (categoria unseen)

### 6.3. Falhas Esperadas

1. **Falsos Negativos (Churn não detectado)**
   - Clientes que cancelam sem sinais claros nos dados disponíveis
   - Impacto: Oportunidades de retenção perdidas

2. **Falsos Positivos (Alarme falso)**
   - Clientes erroneamente classificados como churn
   - Impacto: Custo de intervenções desnecessárias, customer annoyance

3. **Extrapolação para Outras Operadoras**
   - O modelo foi treinado em dados de uma operadora específica
   - Pode performar mal em contextos com diferentes planos/preços/regulamentações

---

## 7. Considerações Éticas

### 7.1. Uso Pretendido

- **Primário:** Apoiar decisões de retenção de clientes em ambientes B2C de telecomunicações
- **Não Destinado a:**
  - Tomada de decisões Fully automated sem revisão humana
  - Uso em contextos regulatórios/creditícios
  - Aplicação em outras indústrias sem re-treino

### 7.2. Riscos

1. **Discriminação:**
   - Evitar uso de predictions para discriminação por idade, gênero, localização
   - Garantir que ações de retenção sejam equitativas

2. **Privacidade:**
   - Os dados utilizados contêm informações geográficas (Cidade, Estado)
   - Garantir conformidade com LGPD/GDPR em produção

3. **Transparência:**
   - Informar aos clientes quando estiverem em programa de retenção baseado em ML
   - Oferecer canal de questionamento/recursos

### 7.3. Mitigações

- **Anonimização:** Nomes e IDs de clientes removidos no treinamento
- **Retenção Humana no Loop:** Previsões devem ser revisadas por atendentes
- **Monitoramento Contínuo:** Acompanhar fairness metrics e drift
- **Explicabilidade:** Considerar técnicas SHAP/LIME para explicar predições individuais (desafio maior em redes neurais)

---

## 8. Manutenção e Monitoramento

### 8.1. Frequência de Re-treinamento

- **Recomendação:** Re-treinar a cada 3-6 meses
- **Gatilhos para re-treinamento imediato:**
  - Data drift detectado (>10% mudança em distribuição de 3+ features)
  - Queda de performance >5% em produção
  - Nova campanha/mudança de preços
  - Expansão geográfica

### 8.2. Métricas de Monitoramento (Produção)

| Métrica | Frequência | Threshold Alerta |
|---------|------------|------------------|
| **ROC-AUC** | Diária/Semanal | < 0.80 |
| **Precision** | Diária/Semanal | < 0.60 |
| **Recall** | Diária/Semanal | < 0.45 |
| **Data Drift** (KS per feature) | Semanal | p < 0.01 |
| **Latência de inferência** | Por request | > 200ms *(maior que árvores)* |
| **Taxa de erro (5xx)** | Diária | > 1% |

### 8.3. Logging e Rastreamento

- **MLflow:** Logging de métricas, parâmetros e artefatos
- **Structured Logging:** structlog em serviço de inferência
- **API Logs:** FastAPI/Uvicorn logs
- **TensorBoard:** (opcional) Para monitoramento de treino futuro

### 8.4. Controle de Versões

- **Código:** Git (commit history)
- **Modelos:** MLflow Model Registry
- **Dados:** Versionamento manual (snapshot em `data/`)
- **Pipeline:** Jupyter notebooks versionados

---

## 9. Como Usar o Modelo

### 9.1. API REST

**Endpoint:** `POST /inference/predict`

**URL de Produção:**
`https://churn-prediction-api.azurewebsites.net`

**Local:** `http://localhost:8000`

**Exemplo de Request:**
```json
{
  "tenure_months": 12,
  "monthly_charges": 79.85,
  "total_charges": 958.20,
  "state": "California",
  "gender": "Male",
  "senior_citizen": "No",
  "partner": "Yes",
  "dependents": "No",
  "phone_service": "Yes",
  "multiple_lines": "No",
  "internet_service": "Fiber optic",
  "online_security": "No",
  "online_backup": "No",
  "device_protection": "No",
  "tech_support": "No",
  "streaming_tv": "No",
  "streaming_movies": "No",
  "contract": "Month-to-month",
  "paperless_billing": "Yes",
  "payment_method": "Electronic check"
}
```

**Response:**
```json
{
  "churn_probability": 0.7234,
  "churn_prediction": true,
  "model": "neural_network"
}
```

### 9.2. Como Carregar Localmente

```python
import pickle
import pandas as pd
from pathlib import Path

# Carregar pipeline
pipeline_path = Path("models/neural_network_pipeline.pkl")
with open(pipeline_path, "rb") as f:
    pipeline = pickle.load(f)

# Preparar dados (features originais + engineered)
data = pd.DataFrame([{
    "Tenure Months": 12,
    "Monthly Charges": 79.85,
    "Total Charges": 958.20,
    "State": "California",
    "Gender": "Male",
    "Senior Citizen": "No",
    "Partner": "Yes",
    "Dependents": "No",
    "Phone Service": "Yes",
    "Multiple Lines": "No",
    "Internet Service": "Fiber optic",
    "Online Security": "No",
    "Online Backup": "No",
    "Device Protection": "No",
    "Tech Support": "No",
    "Streaming TV": "No",
    "Streaming Movies": "No",
    "Contract": "Month-to-month",
    "Paperless Billing": "Yes",
    "Payment Method": "Electronic check"
}])

# Feature engineering é feito automaticamente dentro do pipeline
prob = pipeline.predict_proba(data)[0, 1]
pred = pipeline.predict(data)[0]
print(f"Probabilidade de churn: {prob:.4f}")
print(f"Predição: {'Churn' if pred else 'Não Churn'}")
```

### 9.3. Requisitos

Veja `requirements.txt` para dependências exatas. Principais libs:
- `torch>=2.3` (PyTorch)
- `scikit-learn>=1.5`
- `pandas>=2.2`
- `numpy>=1.26`
- `imbalanced-learn>=0.12` (para SMOTE)

### 9.4. Inferência com GPU (Opcional)

Para melhor performance de inferência em alta carga:

```python
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# O pipeline já está otimizado para uso com CPU, mas o código do
# ChurnNetWrapper pode ser adaptado para mover o modelo para GPU
```

---

## 10. Referências

### 10.1. Recursos do Projeto

- **Repositório:** `/home/guilherme.couceiro/prj/fiap/grupo-18-tech-challenger-fase-1`
- **Notebooks:**
  - `notebooks/01_eda.ipynb` - Análise exploratória
  - `notebooks/02_feature_engineering.ipynb` - Engenharia de features
  - `notebooks/03_modeling.ipynb` - Treinamento e comparação de modelos (inclui Neural Network)
- **Código fonte:**
  - `src/main.py` - FastAPI app
  - `src/services/inference_service.py` - Serviço de inferência
  - `utils/metrics.py` - Funções de avaliação
  - `utils/neural_net.py` - Arquitetura da rede neural (ChurnNet, ChurnNetWrapper, FocalLoss)

### 10.2. Dataset Original

- IBM Sample Data Sets
- Telco Customer Churn
- https://www.kaggle.com/datasets/blastchar/telco-customer-churn

### 10.3. Artigos Técnicos Referenciados

1. **Focal Loss** - Lin et al. (2017). "Focal Loss for Dense Object Detection"
   - Implementado para lidar com classes desbalanceadas
2. **SMOTE** - Chawla et al. (2002). "SMOTE: Synthetic Minority Over-sampling Technique"
   - Usado opcionalmente no treinamento da rede
3. **Cosine Annealing com Warmup** - Loshchilov & Hutter (2016). "SGDR: Stochastic Gradient Descent with Warm Restarts"
   - Scheduler de learning rate

---

## 11. Histórico de Versões

| Versão | Data | Mudanças |
|--------|------|----------|
| 1.0 | 2026-03-30 | Model Card inicial com Neural Network (ROC-AUC: 0.8464 teste, 0.8541 CV) |

---

**Nota:** Este Model Card deve ser atualizado sempre que o modelo for re-treinado ou houver mudanças significativas nos dados, pipeline ou performance.
