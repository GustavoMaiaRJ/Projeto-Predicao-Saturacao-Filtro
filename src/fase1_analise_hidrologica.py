"""
FASE 1: Cruzamento de Dados e Analise Sazonal
COC351 - Matematica Computacional - UFRJ Poli 2026.1
Autores: Gustavo Maia de Araujo 
         Gilson Batista Machado Martins 

Objetivo: cruzar dados de precipitacao (INMET A249) com leituras reais de
turbidez da ETA Mazagao, aplicando lag time de 24h, e calibrar a equacao
de potencia TRB = k * P^m por estacao hidrologica via curve_fit.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path

# --- Parametros ---
LAG_DIAS   = 1        # Lag time: chuva do dia D afeta turbidez do dia D+1
CHUVA_MIN  = 0.5      # mm minimos para incluir no curve_fit
ALPHA      = 0.85     # (mg/L)/NTU -- Abreu et al. 2022
OUT        = Path("outputs")
OUT.mkdir(parents=True, exist_ok=True)

ESTACOES_MESES = {
    "Enchente": [1, 2, 3],
    "Cheia":    [4, 5, 6],
    "Vazante":  [7, 8, 9],
    "Seca":     [10, 11, 12]
}


# --- Carregar dados ---
def carregar_precipitacao(caminho):
    """
    Le o CSV horario do INMET (encoding ISO-8859-1, separador ';',
    8 linhas de cabecalho), agrega para precipitacao diaria acumulada
    e aplica o shift de LAG_DIAS para simular o Lag Time hidrologico.
    """
    df = pd.read_csv(caminho, encoding="ISO-8859-1",
                     sep=";", skiprows=8, decimal=",")
    df["Data"] = pd.to_datetime(df["Data"],
                                format="%Y/%m/%d", errors="coerce")
    df = df.dropna(subset=["Data"])
    col_p = "PRECIPITACAO TOTAL, HORARIO (mm)"
    df[col_p] = pd.to_numeric(df[col_p], errors="coerce").fillna(0)
    chuva = df.groupby("Data")[col_p].sum().reset_index()
    chuva.columns = ["Data", "Chuva_mm"]
    # Aplicar Lag Time: chuva do dia D -> turbidez do dia D+1
    chuva["Data_Turbidez"] = chuva["Data"] + pd.Timedelta(days=LAG_DIAS)
    return chuva[["Data_Turbidez", "Chuva_mm"]]


def carregar_turbidez(caminho):
    """
    Le o CSV consolidado de turbidez real e calcula a media diaria
    para compatibilidade com a base de precipitacao.
    """
    df = pd.read_csv(caminho)
    df["Data"] = pd.to_datetime(df["Data"])
    df["Turbidez"] = pd.to_numeric(df["Turbidez"], errors="coerce")
    df = df.dropna(subset=["Turbidez"])
    df = df[df["Turbidez"] > 0]
    return (df.groupby("Data")["Turbidez"]
              .mean()
              .reset_index()
              .rename(columns={"Turbidez": "Turbidez_media"}))


# --- Cruzar dados ---
def cruzar(df_chuva, df_turb):
    """
    Realiza o inner join entre precipitacao (com lag aplicado) e turbidez
    diaria, classificando cada dia por estacao hidrologica.
    """
    df = df_turb.merge(
        df_chuva.rename(columns={"Data_Turbidez": "Data"}),
        on="Data", how="inner"
    )
    df["Mes"]     = df["Data"].dt.month
    df["Estacao"] = df["Mes"].map(
        lambda m: next(
            (e for e, ms in ESTACOES_MESES.items() if m in ms), "?"
        )
    )
    return df.sort_values("Data").reset_index(drop=True)


# --- Modelo de potencia ---
def pot(P, k, m):
    """Modelo de potencia: TRB = k * P^m  (P > 0)."""
    return k * np.power(np.maximum(P, 0.1), m)


def calibrar(df_est, nome):
    """
    Calibra TRB = k * P^m via curve_fit para dias com chuva >= CHUVA_MIN.
    Retorna dict com k, m, R2, N amostras ou None em caso de falha.
    """
    sub = df_est[df_est["Chuva_mm"] >= CHUVA_MIN]
    if len(sub) < 5:
        print(f"  [AVISO] {nome}: pontos insuficientes ({len(sub)}).")
        return None
    try:
        popt, _ = curve_fit(pot, sub["Chuva_mm"], sub["Turbidez_media"],
                            p0=[50, 0.3],
                            bounds=([0.001, -5], [10000, 5]),
                            maxfev=5000)
        T_pred = pot(sub["Chuva_mm"].values, *popt)
        ss_res = np.sum((sub["Turbidez_media"].values - T_pred) ** 2)
        ss_tot = np.sum((sub["Turbidez_media"].values
                         - sub["Turbidez_media"].mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        return {"k": popt[0], "m": popt[1], "R2": r2, "N": len(sub)}
    except Exception as e:
        print(f"  [AVISO] {nome}: curve_fit falhou ({e}).")
        return None


# --- Visualizacao ---
def plotar_dispersao_sazonal(df, parametros, caminho_saida):
    """
    Gera figura com 4 subplots (2x2), um por estacao hidrologica,
    mostrando a dispersao Chuva x Turbidez e a curva de tendencia ajustada.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ordem = list(ESTACOES_MESES.keys())

    for ax, est in zip(axes.flatten(), ordem):
        sub = df[df["Estacao"] == est]
        ax.scatter(sub["Chuva_mm"], sub["Turbidez_media"],
                   alpha=0.6, s=40, label=f"Dados reais (n={len(sub)})")

        params = parametros.get(est)
        if params:
            P_range = np.linspace(0.1, max(sub["Chuva_mm"].max(), 1), 200)
            T_fit = pot(P_range, params["k"], params["m"])
            ax.plot(P_range, T_fit, "--", lw=2,
                    label=f"TRB={params['k']:.1f}*P^{params['m']:.3f} "
                          f"(R2={params['R2']:.3f})")

        ax.set_title(f"{est}  ({ESTACOES_MESES[est][0]}-{ESTACOES_MESES[est][-1]})")
        ax.set_xlabel("Precipitacao acumulada D-1 (mm)")
        ax.set_ylabel("Turbidez media diaria (NTU)")
        ax.legend(fontsize=8)
        ax.grid(True, ls="--", alpha=0.5)

    fig.suptitle("Relacao Precipitacao-Turbidez por Estacao Hidrologica",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(caminho_saida, dpi=150, bbox_inches="tight")
    plt.close()


# --- Exportar CSV ---
def exportar(df, caminho):
    """Exporta a tabela final de cruzamento com metadados de estacao."""
    out = df[["Data", "Mes", "Estacao",
              "Turbidez_media", "Chuva_mm"]].copy()
    out["Data"] = out["Data"].dt.strftime("%Y-%m-%d")
    out.to_csv(caminho, index=False, sep=";",
               decimal=",", encoding="utf-8-sig")


# --- Pipeline principal ---
if __name__ == "__main__":
    print("=" * 60)
    print("FASE 1 -- Cruzamento de Dados e Analise Sazonal")
    print("=" * 60)

    df_chuva = carregar_precipitacao(
        "INMET_N_AP_A249_MACAPA_01-01-2025_A_31-12-2025.CSV")
    df_turb  = carregar_turbidez("consolidado_turbidez_real.csv")
    df       = cruzar(df_chuva, df_turb)

    print(f"\nDias cruzados: {len(df)}")

    parametros = {}
    for est in ["Enchente", "Cheia", "Vazante", "Seca"]:
        sub    = df[df["Estacao"] == est]
        params = calibrar(sub, est)
        parametros[est] = params
        if params:
            print(f"  {est}: k={params['k']:.2f}, m={params['m']:.4f}, "
                  f"R2={params['R2']:.3f}, N={params['N']}")

    plotar_dispersao_sazonal(df, parametros, OUT / "fig1_dispersao_sazonal.png")
    exportar(df, OUT / "dados_cruzados_reais_mazagao.csv")

    print(f"\nArquivos salvos em: {OUT.resolve()}")
    print("  - fig1_dispersao_sazonal.png")
    print("  - dados_cruzados_reais_mazagao.csv")
