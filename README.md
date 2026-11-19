# Fila Invisível — Data Science & API de Previsão (FIAP GS 2026.2 · CitizenTech)

## Ficha da CitizenTech

| Campo | Descrição |
|---|---|
| **Nome da CitizenTech** | Fila Invisível |
| **Cidade e bairro** | São Paulo – SP, bairro Cambuci *(referência: UBS Cambuci — parametrizável em `scripts/gerar_dataset.py`)* |
| **Cidadão beneficiado** | Idosos (ex.: Dona Marlene, 68 anos, hipertensa, celular simples) e trabalhadores informais que dependem da UBS e não podem perder horas em filas evitáveis |
| **Problema, em uma frase** | Moradores não sabem quando a UBS está cheia, concentram-se na abertura e perdem horas em filas evitáveis |
| **Como funciona** | Modelos de **Regressão Linear** (tempo estimado de espera em minutos) e **Regressão Logística** (probabilidade de lotação $\ge 45$ min) preveem a lotação por dia, horário e especialidade e exibem um **semáforo simples** (Verde, Amarelo e Vermelho) indicando o melhor momento para ir |
| **O que cada parte faz** | **Data Science** (este repositório): base calibrada, notebook estatístico completo, pipelines `.joblib` e API FastAPI de inferência; **Database**: modelagem e scripts Oracle; **DDD Java**: API Spring Boot + persistência; **Dynamic Programming**: melhor rota/horário com grafo; **Network**: rede segmentada da UBS; **Agile**: persona, backlog e protótipo; **AR-VR**: simulação 3D da sala de espera |
| **Link do repositório** | https://github.com/PedroH-Batista/GS_Data_Science |
| **API em produção (Render)** | https://gs-data-science.onrender.com/docs |
| **Integrantes** | Bernardo Moreira (RM: 564103) · Pedro Batista (RM: 563220) · Larissa Shiba (RM: 560462) · Renan Jordão (RM: 560618) |
| **Indicador de impacto** | Redução estimada do tempo médio de espera por visita (**−31,1 min / −52,4%** no conjunto de teste ao migrar do pico `07h–09h` para o Semáforo Verde) |
| **ODS conectados** | **ODS 3** (Saúde e Bem-Estar), **ODS 10** (Redução das Desigualdades) e **ODS 11** (Cidades e Comunidades Sustentáveis) |

---

## Arquitetura do Módulo de Data Science

```text
scripts/gerar_dataset.py  --->  data/registros_ubs.csv (base sintética calibrada, sem dados pessoais)
                                       │
                                       ▼
notebooks/fila_invisivel_ds.ipynb  (Entregável principal — 10,0 pts)
  ├── Diagnóstico de qualidade, limpeza e EDA completa
  ├── Análise de data leakage (exclusão de pessoas_na_fila) e split cronológico 80/20
  ├── Pipeline Regressão Linear    --> prevê tempo_espera_min (MAE, RMSE, R²)
  ├── Pipeline Regressão Logística --> prevê prob_lotacao / ubs_lotada (Matriz Confusão, P, R, F1)
  ├── Recomendações práticas, indicador de impacto e limitações
  └── Exporta models/*.joblib, models/metadados.json e models/exemplos_consistencia.csv
                                       │
                                       ▼
api/main.py  (FastAPI — inferência sob demanda e integração com Spring Boot)
  ├── GET  /health            --> status do serviço e métricas dos modelos
  ├── GET  /metadados         --> metadados completos do experimento
  ├── POST /predict           --> previsão de espera, probabilidade e semáforo
  ├── POST /melhor-horario    --> simula 07h–18h e recomenda a melhor janela do dia
  └── POST /previsoes-spring  --> lote compatível com POST /previsoes do Spring Boot (DDD)
```

---

## Estrutura de Pastas

```text
GS_Data_Science/
├── README.md                              # Ficha da CitizenTech e guia de execução
├── render.yaml                            # Blueprint de deploy gratuito no Render.com
├── requirements.txt                       # Dependências enxutas de produção (API FastAPI / Linux-safe)
├── requirements-dev.txt                   # Dependências completas para executar o Notebook localmente
├── data/
│   ├── registros_ubs.csv                  # Base bruta sintética calibrada (14.898 linhas)
│   └── registros_ubs_tratada.csv          # Base limpa e ordenada cronologicamente (14.874 linhas)
├── notebooks/
│   └── fila_invisivel_ds.ipynb            # Notebook completo e executado (Seções 0 a 9)
├── models/
│   ├── modelo_tempo_espera.joblib         # Pipeline Regressão Linear
│   ├── modelo_lotacao.joblib              # Pipeline Regressão Logística
│   ├── metadados.json                     # Metadados, limiares e métricas
│   └── exemplos_consistencia.csv          # Casos do teste para validação de paridade
├── api/
│   ├── schemas.py                         # Contratos Pydantic v2
│   └── main.py                            # Aplicação FastAPI (Swagger em /docs)
└── scripts/
    ├── gerar_dataset.py                   # Gerador reprodutível (SEED=42, UBS_CONFIG)
    └── testar_api.py                      # Teste automatizado de paridade HTTP
```

---

## Resultados no Conjunto de Teste (20% futuros nunca vistos)

### Regressão Linear (`tempo_espera_min`)

| Métrica | Baseline Ingênuo (Média) | Regressão Linear (Teste) |
|---|---:|---:|
| **MAE (min)** | `14,94` | **`7,36`** *(−50,8%)* |
| **RMSE (min)** | `18,32` | **`9,14`** |
| **$R^2$** | `-0,0079` | **`0,7492`** |

### Regressão Logística (`ubs_lotada >= 45 min`, limiar calibrado no treino `= 0,4224`)

| Métrica | Regressão Logística (Teste) |
|---|---:|
| **Precisão** | `0,7680` |
| **Recall** | `0,9284` |
| **F1-Score** | `0,8406` |
| **ROC-AUC** | `0,9414` |
| **PR-AUC** | `0,9346` |

### Auditoria do Semáforo no Conjunto de Teste

| Cor do Semáforo | Regra | Tempo médio real | Taxa real de lotação |
|---|---|---:|---:|
| **Verde** | $\hat{p} < 0{,}40$ | **`28,3 min`** | **`6,8%`** |
| **Amarelo** | $0{,}40 \le \hat{p} < 0{,}70$ | **`43,2 min`** | **`41,3%`** |
| **Vermelho** | $\hat{p} \ge 0{,}70$ | **`58,7 min`** | **`87,7%`** |

---

## Como Executar Localmente

### 1. Criar e ativar o ambiente virtual (Windows / PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# Para rodar apenas a API FastAPI:
pip install -r requirements.txt
# Para executar também o Notebook e os gráficos localmente:
pip install -r requirements-dev.txt
```

### 2. (Opcional) Regerar a base de dados ou trocar a UBS de referência
Para mudar a UBS ou o bairro, edite `UBS_CONFIG` em `scripts/gerar_dataset.py` e execute:
```powershell
.\.venv\Scripts\python.exe scripts\gerar_dataset.py
```

### 3. Executar o Notebook de Data Science
Abra `notebooks/fila_invisivel_ds.ipynb` no VS Code / Jupyter (selecionando o kernel `.venv`) ou execute via terminal:
```powershell
.\.venv\Scripts\jupyter.exe nbconvert --to notebook --execute --inplace notebooks\fila_invisivel_ds.ipynb
```

### 4. Subir a API FastAPI e abrir o Swagger
```powershell
.\.venv\Scripts\uvicorn.exe api.main:app --reload
```
- **Swagger interativo:** http://127.0.0.1:8000/docs
- **Health check:** http://127.0.0.1:8000/health

### 5. Rodar o teste automatizado de paridade HTTP
```powershell
.\.venv\Scripts\python.exe scripts\testar_api.py
```

---

## Exemplo de Chamada (`POST /predict`)

```json
{
  "especialidade": "Clinica Medica",
  "dia_semana": 0,
  "hora": 8,
  "mes": 10,
  "eh_pos_feriado": 0,
  "campanha_vacinacao": 1,
  "chuva": 0,
  "temperatura_c": 21.5,
  "medicos_escala": 2
}
```

---

## Observação sobre Privacidade e Dados
Nenhum dado pessoal identificável de pacientes ou profissionais de saúde foi utilizado. A base `data/registros_ubs.csv` é sintética e reprodutível (`SEED = 42`), calibrada a partir de parâmetros públicos de operação da Atenção Primária em São Paulo.
