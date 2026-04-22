# ML Canvas — Telco Customer Churn Prediction

> **Dataset:** IBM Telco Customer Churn  
> **Tipo de problema:** Classificação binária supervisionada  
> **Setor:** Telecomunicações  

---

## 1. Problema de Negócio

Taxa de churn de aproximadamente 26% ao mês, com perda significativa de receita mensal. 
Objetivo: Identificar com antecedência quais clientes podem está próximo ao cancelamento, afim de que equipes de retenção atuem de forma proativa com suporte, ofertas ou melhoria de plano, antes que o cliente cancele.

Meta a ser alcançada: reduzir a taxa de churn mensal de ~26% para menos de 20%. 
Periodo de 6 meses, via campanhas de retenção orientadas pelo grau de risco.

--- 

## 2. Stakeholders

| Papel | Responsabilidade |
|---|---|
| Diretoria Financeira | Aprova o projeto e define metas de retenção |
| Equipe de CRM / Retenção | Recebe lista de clientes em risco e aciona campanhas |
| Time de Data Science | Responsável técnico pela EDA, modelagem e validação |
| Jurídico / Compliance | Garante conformidade com LGPD |

---

## 3. Variável Alvo

- **Variável:** `Churn` (Yes / No)
- **Tipo:** Binária
- **Output esperado:** Score de probabilidade de churn e a predição de churn, se vai realizar o churn ou não.

---

## 4. Métricas de Negócio (KPIs)

| KPI | Descrição |
|---|---|
| Taxa de churn mensal | Indicador principal — meta: reduzir de ~26% para <20% |
| Taxa de retenção pós-campanha | Percentual de clientes contatados que não cancelaram |
| ROI da campanha de retenção | Custo por intervenção versus LTV do cliente |
| Cobertura de clientes de alto risco | Percentual dos clientes com alto risco de churn |

---

## 5. SLOs — Service Level Objectives

### Performance do modelo

| SLO | Target | Justificativa |
|---|---|---|
| Recall mínimo aceitável | ≥ 75% | Capturar a maioria dos clientes que realmente estão em churn da base mensal |
| Precision mínima | ≥ 60% | Evitar que o time de CRM receba volume excessivo de falsos positivos |

---

## 6. Restrições e Premissas

- Dataset público IBM Telco (7.043 clientes) — escopo acadêmico
