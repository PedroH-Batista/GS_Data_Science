"""
Gerador da base sintética calibrada — Projeto Fila Invisível (FIAP GS 2026.2 · CitizenTech).

Gera 12 meses de registros operacionais simulados de uma Unidade Básica de Saúde (UBS)
de São Paulo (padrão: UBS Cambuci), sem nenhum dado pessoal identificável, calibrados
em padrões públicos da Atenção Primária do SUS (horário de funcionamento seg–sex 07h–18h,
sazonalidade semanal, picos pós-feriado, sazonalidade respiratória de inverno e campanhas
de vacinação).

Para trocar a UBS de referência, basta alterar o dicionário UBS_CONFIG abaixo e executar
novamente este script.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

# ==============================================================================
# CONFIGURAÇÃO CENTRALIZADA DA UBS (troque aqui caso o grupo mude a UBS/bairro)
# ==============================================================================
UBS_CONFIG = {
    "id_ubs": 1,
    "nome_ubs": "UBS Cambuci",
    "bairro": "Cambuci",
    "cidade": "São Paulo - SP",
    "capacidade_ubs": 40,
}

SEED = 42
LIMIAR_LOTACAO_MIN = 45.0
DATA_INICIO = "2025-11-01"
DATA_FIM = "2026-10-31"
CAMINHO_SAIDA = Path(__file__).resolve().parent.parent / "data" / "registros_ubs.csv"

# Feriados nacionais e municipais de São Paulo no período 2025-11 a 2026-10
FERIADOS_SP = {
    date(2025, 11, 2),   # Finados
    date(2025, 11, 15),  # Proclamação da República
    date(2025, 11, 20),  # Consciência Negra
    date(2025, 12, 25),  # Natal
    date(2026, 1, 1),    # Confraternização Universal
    date(2026, 1, 25),   # Aniversário de São Paulo
    date(2026, 2, 16),   # Carnaval (segunda)
    date(2026, 2, 17),   # Carnaval (terça)
    date(2026, 4, 3),    # Sexta-Feira Santa
    date(2026, 4, 21),   # Tiradentes
    date(2026, 5, 1),    # Dia do Trabalho
    date(2026, 6, 4),    # Corpus Christi
    date(2026, 7, 9),    # Revolução Constitucionalista (SP)
    date(2026, 9, 7),    # Independência do Brasil
    date(2026, 10, 12),  # Nossa Senhora Aparecida
}

ESPECIALIDADES = [
    "Clinica Medica",
    "Pediatria",
    "Vacinacao",
    "Odontologia",
    "Enfermagem",
]

# Efeitos médios de espera (em minutos) calibrados por padrão operacional de UBS
EFEITO_ESPECIALIDADE = {
    "Clinica Medica": 16.0,
    "Pediatria": 10.0,
    "Odontologia": 6.0,
    "Enfermagem": 1.0,
    "Vacinacao": -6.0,
}

EFEITO_HORA = {
    7: 18.0,
    8: 24.0,
    9: 22.0,
    10: 14.0,
    11: 6.0,
    12: -4.0,
    13: 2.0,
    14: 10.0,
    15: 4.0,
    16: -6.0,
    17: -10.0,
    18: -12.0,
}

EFEITO_DIA_SEMANA = {
    0: 12.0,  # Segunda-feira (demanda represada do fim de semana)
    1: 5.0,   # Terça-feira
    2: 1.0,   # Quarta-feira
    3: -3.0,  # Quinta-feira
    4: -6.0,  # Sexta-feira
}

# Temperatura média mensal (°C) e probabilidade diária de chuva em São Paulo
TEMP_MEDIA_MES_SP = {
    1: 24.5, 2: 24.8, 3: 23.8, 4: 21.5, 5: 18.5, 6: 16.8,
    7: 16.5, 8: 18.0, 9: 19.8, 10: 21.2, 11: 22.4, 12: 23.6,
}
PROB_CHUVA_MES_SP = {
    1: 0.48, 2: 0.45, 3: 0.38, 4: 0.22, 5: 0.18, 6: 0.14,
    7: 0.12, 8: 0.14, 9: 0.24, 10: 0.34, 11: 0.38, 12: 0.46,
}

# Escala típica de profissionais por especialidade (mín, máx)
ESCALA_ESPECIALIDADE = {
    "Clinica Medica": (2, 4),
    "Pediatria": (1, 3),
    "Odontologia": (1, 3),
    "Enfermagem": (2, 4),
    "Vacinacao": (1, 3),
}


def calcular_dias_pos_feriado(datas_uteis: list[date]) -> set[date]:
    """Identifica o primeiro dia útil imediatamente após qualquer feriado de SP."""
    pos_feriado: set[date] = set()
    feriados_ordenados = sorted(FERIADOS_SP)
    for fer in feriados_ordenados:
        for d in datas_uteis:
            if d > fer:
                pos_feriado.add(d)
                break
    return pos_feriado


def gerar_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)

    todas_datas = pd.date_range(DATA_INICIO, DATA_FIM, freq="D")
    datas_uteis = [
        d.date() for d in todas_datas
        if d.weekday() < 5 and d.date() not in FERIADOS_SP
    ]
    dias_pos_feriado = calcular_dias_pos_feriado(datas_uteis)

    registros = []
    id_registro = 1

    for dia in datas_uteis:
        mes = dia.month
        dia_semana = dia.weekday()
        eh_segunda = int(dia_semana == 0)
        eh_pos_feriado = int(dia in dias_pos_feriado)

        # Campanhas sazonais típicas (Influenza abr-jun; multivacinação out)
        campanha_vacinacao = int(mes in (4, 5, 6, 10))
        efeito_inverno = 6.0 if mes in (5, 6, 7, 8) else 0.0

        # Clima diário base da cidade de São Paulo
        temp_dia = float(rng.normal(TEMP_MEDIA_MES_SP[mes], 2.6))
        chuva_dia = int(rng.random() < PROB_CHUVA_MES_SP[mes])

        # Escala do turno da manhã (7–12h) e da tarde (13–18h) por especialidade
        escala_manha = {
            esp: int(rng.integers(lo, hi + 1))
            for esp, (lo, hi) in ESCALA_ESPECIALIDADE.items()
        }
        escala_tarde = {
            esp: int(rng.integers(lo, hi + 1))
            for esp, (lo, hi) in ESCALA_ESPECIALIDADE.items()
        }

        for hora in range(7, 19):
            # Variação térmica intradiária realista (mais frio às 7h, mais quente às 14h)
            ajuste_hora_temp = -2.5 + 4.5 * np.sin(np.pi * (hora - 7) / 11.0)
            temperatura_c = round(temp_dia + ajuste_hora_temp + float(rng.normal(0, 0.5)), 1)
            # Chuva tem leve chance de mudar entre manhã e tarde
            chuva = chuva_dia if hora < 13 else int(
                chuva_dia if rng.random() < 0.75 else (1 - chuva_dia)
            )
            escala_atual = escala_manha if hora < 13 else escala_tarde

            for especialidade in ESPECIALIDADES:
                medicos_escala = escala_atual[especialidade]

                # Efeito climático (frio aumenta demanda respiratória; calor extremo eleva leve procura)
                efeito_frio = max(0.0, 20.0 - temperatura_c) * 0.85
                efeito_calor = max(0.0, temperatura_c - 29.0) * 0.90
                efeito_chuva = -6.0 if chuva == 1 else 0.0

                # Efeito de campanha (mais forte em Vacinação e Enfermagem)
                efeito_campanha = 0.0
                if campanha_vacinacao == 1:
                    if especialidade == "Vacinacao":
                        efeito_campanha = 16.0
                    elif especialidade == "Enfermagem":
                        efeito_campanha = 7.0
                    else:
                        efeito_campanha = 3.0

                # Tempo esperado linear + ruído operacional realista
                tempo_base = (
                    42.0
                    + EFEITO_ESPECIALIDADE[especialidade]
                    + EFEITO_HORA[hora]
                    + EFEITO_DIA_SEMANA[dia_semana]
                    + (14.0 * eh_pos_feriado)
                    + efeito_inverno
                    + efeito_campanha
                    + efeito_frio
                    + efeito_calor
                    + efeito_chuva
                    - (6.5 * medicos_escala)
                )

                # Ruído normal + ocasionais picos aleatórios de urgência (4% dos casos)
                ruido = float(rng.normal(0.0, 7.5))
                if rng.random() < 0.04:
                    ruido += float(rng.uniform(10.0, 22.0))

                tempo_espera_min = round(max(4.0, tempo_base + ruido), 1)
                ubs_lotada = int(tempo_espera_min >= LIMIAR_LOTACAO_MIN)

                # Variável observacional (NÃO entra em X no modelo; alinhada a RegistroFila)
                pessoas_estimadas = (tempo_espera_min * medicos_escala) / 11.5 + float(rng.normal(0.0, 1.8))
                pessoas_na_fila = int(np.clip(round(pessoas_estimadas), 0, UBS_CONFIG["capacidade_ubs"] + 15))

                registros.append({
                    "id_registro": id_registro,
                    "id_ubs": UBS_CONFIG["id_ubs"],
                    "nome_ubs": UBS_CONFIG["nome_ubs"],
                    "bairro": UBS_CONFIG["bairro"],
                    "data_hora": f"{dia.isoformat()} {hora:02d}:00:00",
                    "dia_semana": dia_semana,
                    "hora": hora,
                    "mes": mes,
                    "eh_segunda": eh_segunda,
                    "eh_pos_feriado": eh_pos_feriado,
                    "especialidade": especialidade,
                    "campanha_vacinacao": campanha_vacinacao,
                    "chuva": chuva,
                    "temperatura_c": temperatura_c,
                    "medicos_escala": float(medicos_escala),
                    "capacidade_ubs": UBS_CONFIG["capacidade_ubs"],
                    "pessoas_na_fila": pessoas_na_fila,
                    "tempo_espera_min": tempo_espera_min,
                    "ubs_lotada": ubs_lotada,
                })
                id_registro += 1

    df = pd.DataFrame(registros)

    # ==========================================================================
    # Imperfeições operacionais realistas (para demonstração de Data Wrangling)
    # ==========================================================================
    n = len(df)

    # 1) ~1,8% de valores ausentes em temperatura_c (falha de coleta meteorológica)
    idx_nulos_temp = rng.choice(n, size=int(n * 0.018), replace=False)
    df.loc[idx_nulos_temp, "temperatura_c"] = np.nan

    # 2) ~1,5% de valores ausentes em medicos_escala (escala não registrada no turno)
    idx_nulos_med = rng.choice(n, size=int(n * 0.015), replace=False)
    df.loc[idx_nulos_med, "medicos_escala"] = np.nan

    # 3) 6 registros impossíveis de tempo_espera_min negativo (erro de relógio na recepção)
    idx_negativos = rng.choice(n, size=6, replace=False)
    df.loc[idx_negativos, "tempo_espera_min"] = -rng.uniform(2.0, 15.0, size=6).round(1)

    # 4) 18 duplicatas exatas (duplo envio do terminal de recepção)
    idx_duplicatas = rng.choice(n, size=18, replace=False)
    df = pd.concat([df, df.iloc[idx_duplicatas]], ignore_index=True)

    return df


def main() -> None:
    CAMINHO_SAIDA.parent.mkdir(parents=True, exist_ok=True)
    df = gerar_dataset()
    df.to_csv(CAMINHO_SAIDA, index=False, encoding="utf-8")

    print(f"Dataset salvo em: {CAMINHO_SAIDA}")
    print(f"UBS configurada:  {UBS_CONFIG['nome_ubs']} ({UBS_CONFIG['bairro']}, {UBS_CONFIG['cidade']})")
    print(f"Dimensões:        {df.shape[0]:,} linhas x {df.shape[1]} colunas")
    print(f"Período:          {df['data_hora'].min()} até {df['data_hora'].max()}")
    print(f"Duplicatas exatas:{df.duplicated().sum()}")
    print(f"Nulos por coluna:\n{df.isnull().sum()[df.isnull().sum() > 0].to_string()}")
    print(f"Tempos negativos: {(df['tempo_espera_min'] < 0).sum()}")

    validos = df[df["tempo_espera_min"] >= 0].drop_duplicates()
    print(f"\nResumo da base válida ({len(validos):,} registros):")
    print(f"  tempo_espera_min -> média: {validos['tempo_espera_min'].mean():.1f} min | mediana: {validos['tempo_espera_min'].median():.1f} min | dp: {validos['tempo_espera_min'].std():.1f} | min: {validos['tempo_espera_min'].min():.1f} | max: {validos['tempo_espera_min'].max():.1f}")
    print(f"  ubs_lotada (>= {LIMIAR_LOTACAO_MIN:.0f} min) -> {validos['ubs_lotada'].mean() * 100:.1f}% lotada | {(1 - validos['ubs_lotada'].mean()) * 100:.1f}% não lotada")
    print("\nTempo médio por especialidade (min):")
    print(validos.groupby("especialidade")["tempo_espera_min"].mean().sort_values(ascending=False).round(1).to_string())
    print("\nTempo médio por faixa de hora (min):")
    print(validos.groupby("hora")["tempo_espera_min"].mean().round(1).to_string())


if __name__ == "__main__":
    main()
