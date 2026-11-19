# Registro de Prompts de IA — Fila Invisível (GS 2026.2 · CitizenTech)

> Conforme combinado da GS (Slide 21 — "IA como copiloto"), este arquivo registra os prompts utilizados no apoio ao desenvolvimento, revisão arquitetural e validação estatística do módulo de Data Science.

## 1. Alinhamento Arquitetural e Revisão da Proposta ("Grill Me")
- **Objetivo:** Validar se a ideia principal (*Fila Invisível*) e os modelos propostos atendem aos critérios da disciplina de Data Science & Statistical Computing (Regressão Linear + Regressão Logística, sem vazamento de dados) e à arquitetura integrada com Spring Boot / Oracle.
- **Resumo do prompt:** Solicitação de revisão crítica da proposta (modelos de lotação e especialidade, uso de FastAPI, alinhamento com o briefing CitizenTech e definição passo a passo da infraestrutura e arquitetura).
- **Decisões tomadas após revisão:**
  - Manter **Regressão Linear** (previsão de `tempo_espera_min`) e **Regressão Logística** (probabilidade `prob_lotacao` de `ubs_lotada`, alimentando o semáforo Verde/Amarelo/Vermelho), atendendo integralmente à rubrica.
  - Usar `especialidade` como feature categórica de entrada nos modelos (e não como modelo separado de ML, já que a disponibilidade cadastral de especialidades por UBS é uma consulta determinística no Oracle/Spring Boot).
  - Priorizar primeiro 100% do Notebook `.ipynb` + CSV + recomendações/limitações, e construir a API FastAPI na sequência consumindo os pipelines `.joblib` exportados.

## 2. Etapa 2 — Geração da Base Sintética Calibrada (`scripts/gerar_dataset.py`)
- **Objetivo:** Criar base sintética realista e reprodutível (`SEED = 42`) representando 12 meses de dias úteis (seg–sex, 07h–18h) na UBS Cambuci (São Paulo/SP), sem dados pessoais reais e com imperfeições operacionais controladas para o Data Wrangling.
- **Resultado:** `data/registros_ubs.csv` gerado com 14.898 linhas × 19 colunas (`1,37 MB`), 48,1% lotada ($\ge 45$ min) e 51,9% não lotada.

## 3. Etapa 3 — Notebook Parte A (Seções 0 a 4)
- **Objetivo:** Construir `notebooks/fila_invisivel_ds.ipynb` seguindo o padrão acadêmico e visual do projeto de referência (tipografia serifada, protocolo experimental, diagnóstico de qualidade, EDA com interpretações, análise de data leakage, split cronológico 80/20 e `ColumnTransformer` sem vazamento).

## 4. Etapas 4 e 5 — Notebook Completo (Seções 5 a 9 + Exportação de Modelos)
- **Objetivo:** Implementar Regressão Linear (`MAE`, `RMSE`, `R²`, resíduos, coeficientes), Regressão Logística (limiar temporal apenas no treino, Matriz de Confusão, Precisão, Recall, F1, ROC/PR-AUC, auditoria do Semáforo e Odds Ratios), Teste de Consistência, exportação dos pipelines `.joblib` + `metadados.json` + teste de paridade e seção de Recomendações, Impactos e Limitações.

## 5. Etapa 6 — API FastAPI, Paridade HTTP e README
- **Objetivo:** Criar `api/schemas.py` e `api/main.py` (`/health`, `/metadados`, `/predict`, `/melhor-horario`, `/previsoes-spring`), validar paridade 100% HTTP vs. Notebook via `scripts/testar_api.py` e documentar o projeto em `README.md` com a Ficha da CitizenTech.



