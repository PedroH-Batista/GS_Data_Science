"""
API REST FastAPI — Fila Invisível (FIAP Global Solution 2026.2 · CitizenTech).

Carrega exclusivamente os pipelines treinados e validados no notebook:
- models/modelo_tempo_espera.joblib (Regressão Linear)
- models/modelo_lotacao.joblib      (Regressão Logística)
- models/metadados.json             (Configuração da UBS, limiares e métricas)

Execução local:
    uvicorn api.main:app --reload
Documentação interativa (Swagger):
    http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import json
import math
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from api.schemas import (
    ItemPrevisaoSpring,
    MelhorHorarioRequest,
    MelhorHorarioResponse,
    PrevisaoRequest,
    PrevisaoResponse,
    PrevisaoSpringRequest,
    SlotHorario,
    UbsInfo,
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

STATE: dict[str, Any] = {}


def carregar_artefatos() -> None:
    """Carrega os pipelines .joblib e metadados.json gerados pelo notebook."""
    caminho_rlin = MODELS_DIR / "modelo_tempo_espera.joblib"
    caminho_rlog = MODELS_DIR / "modelo_lotacao.joblib"
    caminho_meta = MODELS_DIR / "metadados.json"

    for p in (caminho_rlin, caminho_rlog, caminho_meta):
        if not p.exists():
            raise RuntimeError(
                f"Artefato ausente: {p}. Execute notebooks/fila_invisivel_ds.ipynb antes de subir a API."
            )

    STATE["modelo_rlin"] = joblib.load(caminho_rlin)
    STATE["modelo_rlog"] = joblib.load(caminho_rlog)
    STATE["metadados"] = json.loads(caminho_meta.read_text(encoding="utf-8"))


@asynccontextmanager
async def lifespan(_: FastAPI):
    carregar_artefatos()
    yield
    STATE.clear()


app = FastAPI(
    title="Fila Invisível — API de Previsão de Lotação em UBS",
    description=(
        "Serviço de inferência em Python (FastAPI) para a CitizenTech Fila Invisível "
        "(FIAP Global Solution 2026.2). Serve os pipelines de Regressão Linear "
        "(tempo estimado de espera em minutos) e Regressão Logística (probabilidade "
        "de lotação e semáforo Verde/Amarelo/Vermelho)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def classificar_semaforo(prob: float, verde_max: float, amarelo_max: float) -> str:
    if prob < verde_max:
        return "Verde"
    if prob < amarelo_max:
        return "Amarelo"
    return "Vermelho"


def gerar_mensagem_cidadao(semaforo: str, tempo_min: float, hora: int) -> str:
    if semaforo == "Verde":
        return f"Bom momento para ir! Às {hora:02d}h a UBS tende a estar tranquila (espera estimada de ~{tempo_min:.0f} min)."
    if semaforo == "Amarelo":
        return f"Movimento moderado às {hora:02d}h (espera estimada de ~{tempo_min:.0f} min). Se puder, consulte os horários Verdes."
    return f"Alta chance de fila longa às {hora:02d}h (espera estimada de ~{tempo_min:.0f} min). Recomendamos escolher outro horário."


def montar_dataframe_entrada(
    *,
    especialidade: str,
    dia_semana: int,
    hora: int,
    mes: int,
    eh_pos_feriado: int,
    campanha_vacinacao: int,
    chuva: int,
    temperatura_c: float | None,
    medicos_escala: int | None,
    colunas_x: list[str],
) -> pd.DataFrame:
    """Constrói exatamente o DataFrame esperado pelos pipelines salvos (mesma ordem de colunas)."""
    mes_sin = math.sin(2.0 * math.pi * mes / 12.0)
    mes_cos = math.cos(2.0 * math.pi * mes / 12.0)

    linha = {
        "temperatura_c": np.nan if temperatura_c is None else float(temperatura_c),
        "medicos_escala": np.nan if medicos_escala is None else float(medicos_escala),
        "mes_sin": mes_sin,
        "mes_cos": mes_cos,
        "eh_pos_feriado": int(eh_pos_feriado),
        "campanha_vacinacao": int(campanha_vacinacao),
        "chuva": int(chuva),
        "especialidade": especialidade,
        "dia_semana": int(dia_semana),
        "hora": int(hora),
    }
    return pd.DataFrame([linha], columns=colunas_x)


def executar_inferencia(entrada: pd.DataFrame) -> tuple[float, float, bool, str]:
    meta = STATE["metadados"]
    modelo_rlin = STATE["modelo_rlin"]
    modelo_rlog = STATE["modelo_rlog"]

    tempo_min = float(np.clip(modelo_rlin.predict(entrada)[0], 0.0, None))
    prob = float(modelo_rlog.predict_proba(entrada)[0, 1])
    limiar = float(meta["limiar_classificacao_logistica"])
    lotada = bool(prob >= limiar)
    semaforo = classificar_semaforo(
        prob,
        verde_max=float(meta["semaforo"]["verde_max"]),
        amarelo_max=float(meta["semaforo"]["amarelo_max"]),
    )
    return tempo_min, prob, lotada, semaforo


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Redireciona a raiz para a documentação interativa Swagger (/docs)."""
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["Operação"])
def health_check() -> dict[str, Any]:
    meta = STATE["metadados"]
    return {
        "status": "ok",
        "ubs": meta["ubs"],
        "especialidades_validas": meta["especialidades_validas"],
        "modelos_carregados": ["modelo_tempo_espera.joblib", "modelo_lotacao.joblib"],
        "metricas_teste": {
            "regressao_linear": meta["metricas_regressao_linear"]["teste"],
            "regressao_logistica": meta["metricas_regressao_logistica"]["teste"],
        },
    }


@app.get("/metadados", tags=["Operação"])
def obter_metadados() -> dict[str, Any]:
    return STATE["metadados"]


@app.post("/predict", response_model=PrevisaoResponse, tags=["Previsão"])
def prever_horario(req: PrevisaoRequest) -> PrevisaoResponse:
    meta = STATE["metadados"]
    entrada = montar_dataframe_entrada(
        especialidade=req.especialidade,
        dia_semana=req.dia_semana,
        hora=req.hora,
        mes=req.mes,
        eh_pos_feriado=req.eh_pos_feriado,
        campanha_vacinacao=req.campanha_vacinacao,
        chuva=req.chuva,
        temperatura_c=req.temperatura_c,
        medicos_escala=req.medicos_escala,
        colunas_x=meta["features_modelo"],
    )
    tempo_min, prob, lotada, semaforo = executar_inferencia(entrada)
    return PrevisaoResponse(
        ubs=UbsInfo(**meta["ubs"]),
        especialidade=req.especialidade,
        dia_semana=req.dia_semana,
        hora=req.hora,
        tempo_espera_min=round(tempo_min, 1),
        prob_lotacao=round(prob, 4),
        ubs_lotada=lotada,
        semaforo=semaforo,
        mensagem_cidadao=gerar_mensagem_cidadao(semaforo, tempo_min, req.hora),
    )


@app.post("/melhor-horario", response_model=MelhorHorarioResponse, tags=["Previsão"])
def recomendar_melhor_horario(req: MelhorHorarioRequest) -> MelhorHorarioResponse:
    meta = STATE["metadados"]
    slots: list[SlotHorario] = []

    for hora in range(7, 19):
        entrada = montar_dataframe_entrada(
            especialidade=req.especialidade,
            dia_semana=req.dia_semana,
            hora=hora,
            mes=req.mes,
            eh_pos_feriado=req.eh_pos_feriado,
            campanha_vacinacao=req.campanha_vacinacao,
            chuva=req.chuva,
            temperatura_c=req.temperatura_c,
            medicos_escala=req.medicos_escala,
            colunas_x=meta["features_modelo"],
        )
        tempo_min, prob, _, semaforo = executar_inferencia(entrada)
        slots.append(
            SlotHorario(
                hora=hora,
                tempo_espera_min=round(tempo_min, 1),
                prob_lotacao=round(prob, 4),
                semaforo=semaforo,
            )
        )

    melhor = min(slots, key=lambda s: (s.prob_lotacao, s.tempo_espera_min))
    pior = max(slots, key=lambda s: (s.prob_lotacao, s.tempo_espera_min))
    economia = round(max(0.0, pior.tempo_espera_min - melhor.tempo_espera_min), 1)

    recomendacao = (
        f"Prefira ir às {melhor.hora:02d}h ({melhor.semaforo}, ~{melhor.tempo_espera_min:.0f} min) "
        f"e evite as {pior.hora:02d}h ({pior.semaforo}, ~{pior.tempo_espera_min:.0f} min) — "
        f"economia estimada de {economia:.0f} min."
    )

    return MelhorHorarioResponse(
        ubs=UbsInfo(**meta["ubs"]),
        especialidade=req.especialidade,
        dia_semana=req.dia_semana,
        melhor_horario=melhor,
        pior_horario=pior,
        economia_estimada_min=economia,
        recomendacao=recomendacao,
        grade_horarios=slots,
    )


@app.post("/previsoes-spring", response_model=list[ItemPrevisaoSpring], tags=["Integração Spring Boot"])
def gerar_lote_spring(req: PrevisaoSpringRequest) -> list[ItemPrevisaoSpring]:
    """Gera a grade de previsões do dia no contrato Previsao(idUbs, dataHora, probLotacao) do Spring Boot."""
    try:
        dia_obj = date.fromisoformat(req.data)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="Data inválida. Use YYYY-MM-DD.") from exc

    dia_semana = dia_obj.weekday()
    if dia_semana > 4:
        raise HTTPException(
            status_code=400,
            detail="A UBS funciona apenas em dias úteis (segunda a sexta-feira).",
        )

    meta = STATE["metadados"]
    id_ubs = int(meta["ubs"]["id_ubs"])
    itens: list[ItemPrevisaoSpring] = []

    for hora in range(7, 19):
        entrada = montar_dataframe_entrada(
            especialidade=req.especialidade,
            dia_semana=dia_semana,
            hora=hora,
            mes=dia_obj.month,
            eh_pos_feriado=req.eh_pos_feriado,
            campanha_vacinacao=req.campanha_vacinacao,
            chuva=req.chuva,
            temperatura_c=req.temperatura_c,
            medicos_escala=req.medicos_escala,
            colunas_x=meta["features_modelo"],
        )
        tempo_min, prob, _, semaforo = executar_inferencia(entrada)
        itens.append(
            ItemPrevisaoSpring(
                idUbs=id_ubs,
                dataHora=f"{req.data}T{hora:02d}:00:00",
                especialidade=req.especialidade,
                probLotacao=round(prob, 4),
                tempoEsperaMin=round(tempo_min, 1),
                semaforo=semaforo,
            )
        )

    return itens
