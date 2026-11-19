r"""
Teste automatizado de paridade HTTP (Notebook <-> FastAPI) e validação dos endpoints.
Execução:
    .\.venv\Scripts\python.exe scripts\testar_api.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from api.main import app


def main() -> None:
    exemplos = pd.read_csv(BASE_DIR / "models" / "exemplos_consistencia.csv")

    with TestClient(app) as client:
        # 1) Health check
        r_health = client.get("/health")
        assert r_health.status_code == 200, f"Health falhou: {r_health.text}"
        health = r_health.json()
        print(f"[OK] GET /health -> status={health['status']} | UBS={health['ubs']['nome_ubs']}")

        # 2) Paridade HTTP contra os 6 casos do notebook
        print("\nTeste de Paridade HTTP (POST /predict vs. Notebook):")
        print(f"{'Caso':<44}{'Tempo NB':>10}{'Tempo HTTP':>11}{'Prob NB':>10}{'Prob HTTP':>10}{'Semáforo':>10}")
        for _, row in exemplos.iterrows():
            payload = {
                "especialidade": str(row["especialidade"]),
                "dia_semana": int(row["dia_semana"]),
                "hora": int(row["hora"]),
                "mes": int(row["mes"]),
                "eh_pos_feriado": int(row["eh_pos_feriado"]),
                "campanha_vacinacao": int(row["campanha_vacinacao"]),
                "chuva": int(row["chuva"]),
                "temperatura_c": None if pd.isna(row["temperatura_c"]) else float(row["temperatura_c"]),
                "medicos_escala": None if pd.isna(row["medicos_escala"]) else int(row["medicos_escala"]),
            }
            resp = client.post("/predict", json=payload)
            assert resp.status_code == 200, f"Erro em /predict: {resp.text}"
            body = resp.json()

            dif_t = abs(body["tempo_espera_min"] - round(float(row["tempo_previsto"]), 1))
            dif_p = abs(body["prob_lotacao"] - round(float(row["prob_lotacao"]), 4))
            assert dif_t <= 0.11, f"Divergência de tempo: {dif_t}"
            assert dif_p <= 1e-4, f"Divergência de prob: {dif_p}"
            assert body["semaforo"] == row["semaforo"], f"Semáforo divergente: {body['semaforo']} vs {row['semaforo']}"

            print(
                f"{row['caso']:<44}{row['tempo_previsto']:>10.1f}{body['tempo_espera_min']:>11.1f}"
                f"{row['prob_lotacao']:>10.4f}{body['prob_lotacao']:>10.4f}{body['semaforo']:>10}"
            )

        # 3) Teste de robustez com valores ausentes (SimpleImputer dentro do Pipeline)
        r_nulo = client.post(
            "/predict",
            json={
                "especialidade": "Clinica Medica",
                "dia_semana": 0,
                "hora": 8,
                "mes": 6,
                "eh_pos_feriado": 1,
                "campanha_vacinacao": 1,
                "chuva": 0,
                "temperatura_c": None,
                "medicos_escala": None,
            },
        )
        assert r_nulo.status_code == 200, f"Erro com nulos: {r_nulo.text}"
        b_nulo = r_nulo.json()
        print(
            f"\n[OK] POST /predict (com nulos imputados pelo Pipeline) -> "
            f"{b_nulo['tempo_espera_min']} min | prob={b_nulo['prob_lotacao']} | {b_nulo['semaforo']}"
        )

        # 4) Melhor horário do dia (cenário Dona Marlene)
        r_melhor = client.post(
            "/melhor-horario",
            json={
                "especialidade": "Clinica Medica",
                "dia_semana": 3,
                "mes": 10,
                "eh_pos_feriado": 0,
                "campanha_vacinacao": 0,
                "chuva": 0,
                "temperatura_c": 22.0,
                "medicos_escala": 3,
            },
        )
        assert r_melhor.status_code == 200, f"Erro em /melhor-horario: {r_melhor.text}"
        b_melhor = r_melhor.json()
        print(f"[OK] POST /melhor-horario -> {b_melhor['recomendacao']}")

        # 5) Lote no formato Spring Boot (POST /previsoes)
        r_spring = client.post(
            "/previsoes-spring",
            json={
                "data": "2026-11-23",
                "especialidade": "Clinica Medica",
                "eh_pos_feriado": 0,
                "campanha_vacinacao": 0,
                "chuva": 0,
                "temperatura_c": 22.0,
                "medicos_escala": 3,
            },
        )
        assert r_spring.status_code == 200, f"Erro em /previsoes-spring: {r_spring.text}"
        lote = r_spring.json()
        assert len(lote) == 12
        print(f"[OK] POST /previsoes-spring -> {len(lote)} slots gerados (ex.: {lote[0]})")

        # 6) Normalização de acentos em especialidade (integração resiliente com Java/Front)
        r_acento = client.post(
            "/predict",
            json={
                "especialidade": "Clínica Médica",
                "dia_semana": 0,
                "hora": 8,
                "mes": 10,
                "eh_pos_feriado": 0,
                "campanha_vacinacao": 0,
                "chuva": 0,
                "temperatura_c": 22.0,
                "medicos_escala": 2,
            },
        )
        assert r_acento.status_code == 200, f"Falha na normalização de acentos: {r_acento.text}"
        assert r_acento.json()["especialidade"] == "Clinica Medica"
        print(f"[OK] Normalização de acento ('Clínica Médica' -> '{r_acento.json()['especialidade']}')")

        # 7) Redirecionamento da raiz / -> /docs
        r_root = client.get("/", follow_redirects=False)
        assert r_root.status_code in (302, 307) and r_root.headers.get("location") == "/docs"
        print(f"[OK] GET / -> redireciona para {r_root.headers.get('location')}")

    print("\nTodos os testes da API FastAPI passaram com 100% de paridade!")


if __name__ == "__main__":
    main()
