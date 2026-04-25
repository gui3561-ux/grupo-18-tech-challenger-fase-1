# Roteiro de vídeo — Tech Challenge Fase 1 (Grupo 18)

## Metadados

| Campo | Valor |
|--------|--------|
| **Título sugerido** | Do dataset IBM Telco à API em produção: churn preditivo com observabilidade |
| **Duração alvo** | ~4 minutos |
| **Projeto** | MVP de predição de churn (telecom) — API FastAPI, modelo serializado, CI/CD Azure |
| **Fonte dos fatos** | `README.md`, `MODEL_CARD.md`, `docs/03_modeling.md` |

**Confirmação (coleta):** Problema = priorizar clientes com alto risco de cancelamento; dados = IBM Telco Customer Churn (~7.043 clientes); modelo em produção = pipeline neural (`neural_network_pipeline.pkl`); métrica principal = ROC-AUC **0,8464** (holdout) e **0,8541** (validação cruzada 5 folds); entrega = API + métricas Prometheus + deploy GitHub Actions → GHCR → Azure App Service.

---

## Roteiro por blocos de tempo (texto falado)

### 0:00–0:22 — Abertura (*hook*)

**Fala sugerida:**

> Em telecom, cada cliente que cancela sem aviso é receita que some.  
> A pergunta que guiou este trabalho foi: como passar de um modelo no notebook para um **serviço** que time de retenção pode consumir — com **probabilidade de churn**, documentação e **monitoramento**?  
> Neste vídeo mostro o MVP que o Grupo 18 entregou na FIAP Pós Tech: predição de churn ponta a ponta.

---

### 0:22–1:05 — Situação (S) + início da Tarefa (T)

**Fala sugerida:**

> **Situação:** churn é uma das principais fontes de perda de receita. Agir tarde encarece campanhas e reduz a chance de reter o cliente.  
> O cenário típico é ter cadastro, contrato, serviços e fatura — mas não ter um score único e operável para priorizar quem contatar primeiro.  
> **Tarefa:** no Tech Challenge, o foco não era só acurar um algoritmo. Era entregar **engenharia de ML**: pipeline versionado, **API REST** com contrato validado, *health check*, documentação OpenAPI e **observabilidade** — além de **CI/CD** até ambiente em nuvem.

---

### 1:05–2:35 — Ação (A) — dados, modelagem, API e operações

**Fala sugerida:**

> **Dados:** usamos o conjunto público **IBM Telco Customer Churn**, com cerca de **sete mil** clientes em *snapshot*. O alvo é binário: cancelou ou não. Há desbalanceamento — por volta de **vinte e seis por cento** de churn —, tratado no treino com **SMOTE** e com função de custo **Focal Loss** na rede neural.  
> Removemos colunas com **vazamento** em relação ao alvo, como *churn score* ou motivo de cancelamento, quando presentes no *raw*.  
> **Features:** além das variáveis de contrato e serviço, derivamos sinais de negócio — por exemplo, perfil de alto risco quando internet é fibra **e** contrato é mês a mês; idoso isolado; custo médio por mês de permanência.  
> **Modelagem:** no notebook de modelagem comparamos **vários classificadores** em *pipelines* scikit-learn completos — *encoding*, escalonamento quando necessário, **SelectKBest** com **trinta e cinco** *features* no melhor *run*, e busca de hiperparâmetros com validação cruzada estratificada.  
> O artefato que **subimos em produção** é uma **rede MLP** em PyTorch dentro do *pipeline*: camadas **cento e vinte e oito, sessenta e quatro, trinta e dois** até a saída, com **BatchNorm** e **Dropout**, *early stopping* e agendador de taxa de aprendizado.  
> **Métrica principal:** **ROC-AUC** no *holdout* de **0,8464**; na validação cruzada de cinco partes, **0,8541** — ou seja, bom poder de **ordenação** de risco. No *model card* deixamos explícito que, com limiar **0,5**, há **trade-off**: o *recall* de churn real não é perfeito; em produção real o limiar costuma ser ajustado ao custo de falhar em detectar churn versus incomodar quem não cancelaria.  
> **Serviço:** a inferência está na **FastAPI**. O cliente envia JSON com demografia, serviços e contrato; a API devolve **probabilidade**, decisão binária com limiar **0,5** e identificador do modelo. O mesmo serviço expõe **métricas Prometheus** — latência, contagem de predições, histograma de probabilidade — integradas a **Grafana** para acompanhamento.  
> **Deploy:** *pipeline* no GitHub Actions: testes com Ruff, Mypy e Pytest; na *main*, build da imagem, push para o **GHCR** e deploy no **Azure App Service**, com *health gate* após o deploy.

---

### 2:35–3:35 — Resultado (R) + limitações e fechamento

**Fala sugerida:**

> **Resultado:** entregamos um **MVP acadêmico** que fecha o ciclo: dados tabulares → modelo com ROC-AUC forte → **artefato *pickle*** servido por API pública documentada, com observabilidade e *pipeline* de entrega reprodutível.  
> **Limitações:** o *dataset* é um *snapshot* de referência, com viés geográfico e de oferta; **não** generaliza automaticamente para outro país ou operadora. Não há autenticação nem CRM acoplado — KPIs como churn mensal observado ou ROI de campanha dependem de **eventos de negócio** que ainda não estão no escopo.  
> **Próximos passos naturais:** ingestão de outcomes de campanha, *feedback loop* de modelo e *fairness* contínua sob política de dados.  
> **Fechamento:** o código e o *model card* estão no repositório do grupo; na descrição do vídeo deixo os links para **Swagger**, **health** e **dashboard** — convido a abrir o `/docs` e testar um `predict` ao vivo.

---

## O que mostrar na tela (por segmento)

| Tempo | Sugestão de tela |
|--------|-------------------|
| 0:00–0:22 | README no GitHub — título do projeto e uma linha do problema; ou logo FIAP + título em slide simples. |
| 0:22–1:05 | `MODEL_CARD.md` seção de propósito / “não é destinado a”; ou diagrama mental: “notebook → produto”. |
| 1:05–1:40 | Notebook `notebooks/eda.ipynb` ou `docs/01_eda.md` — distribuição do *target*; opcional: `docs/02_feature_engineering.md`. |
| 1:40–2:10 | `docs/03_modeling.md` ou trecho do `03_modeling.ipynb` — esquema *ColumnTransformer* → *SelectKBest* → classificador; `utils/neural_net.py` se quiser mostrar arquitetura. |
| 2:10–2:35 | **Swagger** (`/docs`) em produção ou `localhost`; **POST** `/api/v1/inference/predict` com o exemplo do README; aba com **Prometheus** (`/api/v1/metrics/`) ou **Grafana** (link público do README). |
| 2:35–3:35 | Tabela de métricas do README (ROC-AUC); `.github/workflows/ci.yml` ou *Actions* verdes; slide com “limitações” em 3 *bullets*; fechar com QR ou links na descrição. |

---

## Checklist pré-gravação

- [ ] Definir se a demo será **produção** (`churn-prediction-api.azurewebsites.net`) ou **local** (`uv run uvicorn …` na porta 8000).
- [ ] Testar antes: `GET /api/v1/health` retorna `ok` e `model_loaded: true`.
- [ ] Ter o **JSON de exemplo** do README copiado em arquivo ou ferramenta (*Insomnia* / *curl*) para colar no *predict* sem erro de digitação.
- [ ] Abas já abertas: **Swagger**, **métricas** (primeiras linhas), **Grafana** ou *GitHub Actions* — evitar *login* ou *2FA* no meio da gravação.
- [ ] Áudio: fones ou microfone testado; ambiente sem eco forte.
- [ ] Não exibir **secrets**, tokens do Grafana Alloy nem `.env`; se mostrar terminal, máscara para variáveis sensíveis.
- [ ] Se citar **F1** ou *recall* além do ROC-AUC, alinhar com o **MODEL_CARD** (valores são ordens de grandeza / cenários — não arredondar como se fossem medidos no vídeo sem consultar o artefato).
- [ ] Resolução e escala da fonte do navegador legíveis em **1080p**; fechar notificações.

---

## Referência rápida (não ler em voz alta)

- **ROC-AUC (*teste*):** 0,8464  
- **ROC-AUC (CV 5-*fold*):** 0,8541  
- **Registros:** ~7.043  
- **Features após seleção:** 35  
- **Artefato:** `models/neural_network_pipeline.pkl`  
- **Links:** ver tabela “Links de Acesso” no `README.md`
