"""Contratos de entrada e saída (Pydantic v2) da API FastAPI — Fila Invisível."""

from __future__ import annotations

import unicodedata
from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator

EspecialidadeLiteral = Literal[
    "Clinica Medica",
    "Pediatria",
    "Vacinacao",
    "Odontologia",
    "Enfermagem",
]

SemaforoLiteral = Literal["Verde", "Amarelo", "Vermelho"]

_MAPA_ESPECIALIDADES = {
    "clinica medica": "Clinica Medica",
    "pediatria": "Pediatria",
    "vacinacao": "Vacinacao",
    "odontologia": "Odontologia",
    "enfermagem": "Enfermagem",
}


def _normalizar_especialidade(valor: str) -> str:
    if not isinstance(valor, str):
        return valor
    sem_acento = "".join(
        ch for ch in unicodedata.normalize("NFKD", valor.strip())
        if not unicodedata.combining(ch)
    )
    chave = " ".join(sem_acento.lower().split())
    return _MAPA_ESPECIALIDADES.get(chave, valor)


class PrevisaoRequest(BaseModel):
    """Atributos conhecidos ANTES de o cidadão sair de casa (zero data leakage)."""

    especialidade: EspecialidadeLiteral = Field(
        ...,
        description="Especialidade desejada na UBS.",
        examples=["Clinica Medica"],
    )
    dia_semana: int = Field(
        ...,
        ge=0,
        le=4,
        description="Dia útil da semana: 0=Segunda, 1=Terça, 2=Quarta, 3=Quinta, 4=Sexta.",
        examples=[0],
    )
    hora: int = Field(
        ...,
        ge=7,
        le=18,
        description="Hora cheia de funcionamento da UBS (07h às 18h).",
        examples=[8],
    )
    mes: int = Field(
        ...,
        ge=1,
        le=12,
        description="Mês do ano (1 a 12) para sazonalidade cíclica.",
        examples=[10],
    )
    eh_pos_feriado: int = Field(
        0,
        ge=0,
        le=1,
        description="1 se for o primeiro dia útil após feriado em SP; 0 caso contrário.",
    )
    campanha_vacinacao: int = Field(
        0,
        ge=0,
        le=1,
        description="1 se houver campanha ativa de vacinação; 0 caso contrário.",
    )
    chuva: int = Field(
        0,
        ge=0,
        le=1,
        description="1 se houver previsão de chuva moderada/forte; 0 caso contrário.",
    )
    temperatura_c: Optional[float] = Field(
        None,
        ge=0.0,
        le=45.0,
        description="Temperatura prevista em °C. Se omitida, o Pipeline aplica SimpleImputer (mediana do treino).",
        examples=[21.5],
    )
    medicos_escala: Optional[int] = Field(
        None,
        ge=1,
        le=10,
        description="Quantidade de profissionais escalados no turno. Se omitida, o Pipeline aplica SimpleImputer.",
        examples=[2],
    )

    @field_validator("especialidade", mode="before")
    @classmethod
    def validar_especialidade(cls, v: str) -> str:
        return _normalizar_especialidade(v)


class UbsInfo(BaseModel):
    id_ubs: int
    nome_ubs: str
    bairro: str
    cidade: str
    capacidade_ubs: int


class PrevisaoResponse(BaseModel):
    ubs: UbsInfo
    especialidade: EspecialidadeLiteral
    dia_semana: int
    hora: int
    tempo_espera_min: float = Field(..., description="Tempo estimado de espera em minutos (Regressão Linear).")
    prob_lotacao: float = Field(..., ge=0.0, le=1.0, description="Probabilidade estimada de lotação >= 45 min (Regressão Logística).")
    ubs_lotada: bool = Field(..., description="Classificação binária segundo o limiar calibrado no treino.")
    semaforo: SemaforoLiteral = Field(..., description="Semáforo cidadão: Verde (<0.40), Amarelo (0.40-0.69), Vermelho (>=0.70).")
    mensagem_cidadao: str = Field(..., description="Orientação em linguagem simples para o cidadão.")


class MelhorHorarioRequest(BaseModel):
    """Parâmetros do dia para simular toda a grade horária (07h–18h) e encontrar a melhor janela."""

    especialidade: EspecialidadeLiteral = Field(..., examples=["Clinica Medica"])
    dia_semana: int = Field(..., ge=0, le=4, examples=[3])
    mes: int = Field(..., ge=1, le=12, examples=[10])
    eh_pos_feriado: int = Field(0, ge=0, le=1)
    campanha_vacinacao: int = Field(0, ge=0, le=1)
    chuva: int = Field(0, ge=0, le=1)
    temperatura_c: Optional[float] = Field(None, ge=0.0, le=45.0, examples=[22.0])
    medicos_escala: Optional[int] = Field(None, ge=1, le=10, examples=[3])

    @field_validator("especialidade", mode="before")
    @classmethod
    def validar_especialidade(cls, v: str) -> str:
        return _normalizar_especialidade(v)


class SlotHorario(BaseModel):
    hora: int
    tempo_espera_min: float
    prob_lotacao: float
    semaforo: SemaforoLiteral


class MelhorHorarioResponse(BaseModel):
    ubs: UbsInfo
    especialidade: EspecialidadeLiteral
    dia_semana: int
    melhor_horario: SlotHorario
    pior_horario: SlotHorario
    economia_estimada_min: float
    recomendacao: str
    grade_horarios: list[SlotHorario]


class PrevisaoSpringRequest(BaseModel):
    """Gera payload em lote compatível com POST /previsoes do backend Spring Boot (DDD)."""

    data: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$", description="Data útil no formato YYYY-MM-DD.", examples=["2026-11-23"])
    especialidade: EspecialidadeLiteral = Field(..., examples=["Clinica Medica"])
    eh_pos_feriado: int = Field(0, ge=0, le=1)
    campanha_vacinacao: int = Field(0, ge=0, le=1)
    chuva: int = Field(0, ge=0, le=1)
    temperatura_c: Optional[float] = Field(None, ge=0.0, le=45.0, examples=[22.0])
    medicos_escala: Optional[int] = Field(None, ge=1, le=10, examples=[3])

    @field_validator("especialidade", mode="before")
    @classmethod
    def validar_especialidade(cls, v: str) -> str:
        return _normalizar_especialidade(v)


class ItemPrevisaoSpring(BaseModel):
    """Espelho do modelo de domínio Previsao(idUbs, dataHora, probLotacao) do AGENTS.md."""

    idUbs: int
    dataHora: str
    especialidade: EspecialidadeLiteral
    probLotacao: float
    tempoEsperaMin: float
    semaforo: SemaforoLiteral
