"""
FASE 2: Simulador de Saturacao do Filtro Rapido de Areia
COC351 - Matematica Computacional - UFRJ Poli 2026.1
Autores: Gustavo Maia de Araujo 
         Gilson Batista Machado Martins 

Metodos numericos aplicados:
    1. Busca de Raizes -- Metodo da Secante
    2. Integracao Numerica -- Regra de Simpson 1/3 Composta

Objetivo: encontrar o instante de colapso (t*) do filtro rapido de areia
da ETA Mazagao e o volume de agua tratada ate esse instante, considerando
vazao oscilante da bomba e calibracao sazonal do fator de bloqueio de
poros (beta).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

OUT = Path("outputs")
OUT.mkdir(parents=True, exist_ok=True)

# --- Parametros fisicos ---
EPS_0    = 0.387        # porosidade inicial
MU       = 1.002e-3     # viscosidade dinamica [Pa*s]
L_LEITO  = 0.60         # espessura do leito [m]
D_P      = 1.0e-3       # diametro do grao [m]
RHO      = 998.0        # massa especifica da agua [kg/m3]
G        = 9.81         # aceleracao gravitacional [m/s2]
A_FILTRO = 17.28        # area do filtro [m2]
DP0_MCA  = 0.30         # offset inicial de pressao [MCA]
DP_MAX   = 3.0          # gatilho de colmatacao / retrolavagem [MCA]

# Estacoes hidrologicas calibradas (k, Q_base em L/s)
# k [1/h]: taxa de reducao de porosidade, calibrada offline via
#          k = beta * SST * Q / (A * L * rho_lodo), SST = 0.85 * NTU
# Ver Secao 3.2 do relatorio para a justificativa completa da calibracao.
ESTACOES = {
    "Enchente": {"k": 0.0175, "Q_base": 21.5, "beta": 38},
    "Cheia":    {"k": 0.0115, "Q_base": 20.0, "beta": 25},
    "Vazante":  {"k": 0.0075, "Q_base": 19.5, "beta": 16},
    "Seca":     {"k": 0.0058, "Q_base": 18.0, "beta": 13},
}


# ============================================================================
# MODELO FISICO
# ============================================================================

def Q_din(t, Q_base):
    """
    Vazao dinamica com oscilacao senoidal, representando a variacao
    real do inversor de frequencia da bomba:
        Q(t) = Q_base + 1.5*sin(t)   [L/s]
    Retorna em m3/s.
    """
    return (Q_base + 1.5 * np.sin(t)) * 1e-3  # L/s -> m3/s


def ergun(eps, v):
    """
    Equacao de Ergun-Botari para perda de carga em leito granular [Pa].

        dP = [150*mu*L*(1-eps)^2 / (dp^2*eps^3)] * v      <- viscoso (Darcy)
           + [1.75*rho*L*(1-eps) / (dp*eps^3)] * v^2       <- inercial (Forchheimer)

    Parametros
    ----------
    eps : porosidade atual do leito [adimensional]
    v   : velocidade superficial [m/s]
    """
    if eps <= 0.005:
        return np.inf
    t_vis = 150 * MU * L_LEITO * (1 - eps) ** 2 / (D_P ** 2 * eps ** 3)
    t_ine = 1.75 * RHO * L_LEITO * (1 - eps) / (D_P * eps ** 3) * v
    return (t_vis + t_ine) * v


def f_raiz(t, k, Q_base):
    """
    Funcao-alvo para a busca de raiz:
        f(t) = DeltaP_total(t) - DeltaP_max

    f(t*) = 0  <=>  o filtro atingiu o limite de 3.0 MCA (colapso).
    """
    eps = max(EPS_0 - k * t, 0.005)
    v   = Q_din(t, Q_base) / A_FILTRO
    dP  = ergun(eps, v) / (RHO * G) + DP0_MCA  # Pa -> MCA
    return dP - DP_MAX


# ============================================================================
# METODO 1 -- BUSCA DE RAIZES: METODO DA SECANTE
# ============================================================================

def metodo_secante(f, t0=5.0, t1=6.0, tol=1e-8, max_iter=200):
    """
    Metodo da Secante com pre-condicionador de bracketing adaptativo.

    Iteracao central:
        t_{n+1} = t_n - f(t_n) * (t_n - t_{n-1}) / (f(t_n) - f(t_{n-1}))

    Justificativa: a derivada numerica de f(t) amplificaria o ruido
    senoidal de Q(t), fazendo o metodo de Newton-Raphson divergir.
    A Secante usa apenas avaliacoes de f, sem depender de derivada,
    garantindo estabilidade sob a perturbacao oscilatoria.

    Pre-condicionador de bracketing:
        Os chutes fixos t0=5h e t1=6h partem antes da raiz (f<0 em
        ambos). A secante pura divergiria nesse caso. O pre-condicionador
        avanca a janela com passo adaptativo (x1.5) ate encontrar
        f(t0)*f(t1) < 0 (mudanca de sinal), seguido de 3 passos de
        bissecao para refinar a janela antes de aplicar a Secante.

    Parametros
    ----------
    f        : funcao-alvo f(t*) = 0
    t0, t1   : chutes iniciais [h] (conforme especificacao: 5.0 e 6.0)
    tol      : tolerancia de convergencia |f(t)| < tol [MCA]
    max_iter : limite total de iteracoes

    Retorna
    -------
    dict com: t_star, iteracoes (apenas fase Secante), residuo, convergiu
    """
    f0, f1 = f(t0), f(t1)
    iters = 2

    # Fase 1: bracketing adaptativo
    passo = (t1 - t0)
    while f0 * f1 > 0 and iters < max_iter:
        passo *= 1.5
        t1, f1 = t0 + passo, f(t0 + passo)
        iters += 1

    if f0 * f1 > 0:
        return {"t_star": None, "iteracoes": iters,
                "residuo": None, "convergiu": False,
                "mensagem": "Bracketing falhou."}

    # Fase 1b: refinamento por bissecao (3 passos)
    for _ in range(3):
        t_m, f_m = (t0 + t1) / 2, f((t0 + t1) / 2)
        iters += 1
        if f0 * f_m < 0:
            t1, f1 = t_m, f_m
        else:
            t0, f0 = t_m, f_m

    # Fase 2: Metodo da Secante propriamente dito
    iter_sec = 0
    for _ in range(max_iter - iters):
        den = f1 - f0
        if abs(den) < 1e-20:
            break
        t2 = t1 - f1 * (t1 - t0) / den
        f2 = f(max(t2, 0.01))
        iters += 1
        iter_sec += 1
        if abs(f2) < tol:
            return {"t_star": t2, "iteracoes": iter_sec,
                    "residuo": abs(f2), "convergiu": True}
        t0, f0 = t1, f1
        t1, f1 = t2, f2

    return {"t_star": t1, "iteracoes": iter_sec,
            "residuo": abs(f1), "convergiu": False}


# ============================================================================
# METODO 2 -- INTEGRACAO NUMERICA: REGRA DE SIMPSON 1/3 COMPOSTA
# ============================================================================

def simpson_volume(t_star, Q_base, n=None):
    """
    Calcula o volume de agua tratada pela Regra de Simpson 1/3 Composta:

        V = integral de 0 a t* de Q(t) dt
          ~= (h/3) * [Q0 + 4*Q1 + 2*Q2 + 4*Q3 + ... + Qn]

    com passo uniforme h = t*/n (n par). Erro de truncamento: O(h^4).

    Parametros
    ----------
    t_star  : instante de colapso [h]
    Q_base  : vazao base [L/s]
    n       : numero de subintervalos (par); default: ceil(t*/0.25)

    Retorna
    -------
    Volume tratado [m3]
    """
    if n is None:
        n = int(np.ceil(t_star / 0.25))
    if n % 2 != 0:
        n += 1
    n = max(n, 4)

    t_vec = np.linspace(0.0, t_star, n + 1)
    h     = t_vec[1] - t_vec[0]
    Q_vec = np.array([Q_din(t, Q_base) * 3600 for t in t_vec])  # m3/h

    # Coeficientes de Simpson: 1, 4, 2, 4, 2, ..., 4, 1
    coef = np.ones(n + 1)
    coef[1:-1:2] = 4.0
    coef[2:-2:2] = 2.0

    return (h / 3.0) * np.dot(coef, Q_vec)


# ============================================================================
# VISUALIZACAO
# ============================================================================

def gerar_figura_resultados(resultados, caminho_saida):
    """
    Gera figura com 3 paineis:
      (A) Evolucao de DeltaP(t) por estacao
      (B) Decaimento da porosidade eps(t)
      (C) Eficiencia computacional -- iteracoes da Secante
    """
    fig = plt.figure(figsize=(20, 11))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1.5, 1])
    ax_dp  = fig.add_subplot(gs[0, 0])
    ax_eps = fig.add_subplot(gs[1, 0], sharex=ax_dp)
    ax_bar = fig.add_subplot(gs[:, 1])

    cores = {"Enchente": "#002d72", "Cheia": "#1f77b4",
             "Vazante": "#ff7f0e", "Seca": "#2ca02c"}

    for nome, res in resultados.items():
        cor = cores[nome]
        t_vec = np.linspace(0, res["t_star"], 300)
        dP_vec  = [f_raiz(t, res["k"], res["Q_base"]) + DP_MAX for t in t_vec]
        eps_vec = [max(EPS_0 - res["k"] * t, 0.005) for t in t_vec]

        ax_dp.plot(t_vec, dP_vec, color=cor, lw=2.2,
                  label=f"{nome} (t*={res['t_star']:.1f}h)")
        ax_dp.plot(res["t_star"], DP_MAX, "o", color=cor, ms=10)

        ax_eps.plot(t_vec, eps_vec, color=cor, lw=2.2)

    ax_dp.axhline(DP_MAX, color="red", ls="--", label="Limite: 3.0 MCA")
    ax_dp.set_ylabel("Perda de carga (MCA)")
    ax_dp.set_title("(A) Evolucao da Perda de Carga", fontweight="bold")
    ax_dp.legend(fontsize=9)
    ax_dp.grid(True, ls="--", alpha=0.5)

    ax_eps.set_xlabel("Tempo (horas)")
    ax_eps.set_ylabel("Porosidade")
    ax_eps.set_title("(B) Decaimento da Porosidade", fontweight="bold")
    ax_eps.grid(True, ls="--", alpha=0.5)

    nomes  = list(resultados.keys())
    iters  = [resultados[n]["iteracoes"] for n in nomes]
    cores_b = [cores[n] for n in nomes]
    bars = ax_bar.bar(nomes, iters, color=cores_b, alpha=0.85)
    for bar, it in zip(bars, iters):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, it + 0.1,
                    str(it), ha="center", fontweight="bold", fontsize=14)
    ax_bar.set_ylabel("Iteracoes da Secante")
    ax_bar.set_title("(C) Eficiencia Computacional", fontweight="bold")
    ax_bar.grid(True, ls="--", alpha=0.5, axis="y")

    plt.tight_layout()
    fig.savefig(caminho_saida, dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================================
# SIMULACAO PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("FASE 2 -- Simulador de Saturacao do Filtro")
    print("=" * 60)

    resultados = {}

    for nome, p in ESTACOES.items():
        k, Qb = p["k"], p["Q_base"]
        res = metodo_secante(lambda t: f_raiz(t, k, Qb))
        vol = simpson_volume(res["t_star"], Qb) if res["convergiu"] else 0

        resultados[nome] = {
            "t_star": res["t_star"], "k": k, "Q_base": Qb,
            "iteracoes": res["iteracoes"], "residuo": res["residuo"],
            "volume": vol,
        }

        print(f"\n[{nome}]")
        print(f"  t*      = {res['t_star']:.2f} h")
        print(f"  Volume  = {vol:.0f} m3 ({vol*1000:.0f} L)")
        print(f"  Secante = {res['iteracoes']} iteracoes | "
              f"residuo = {res['residuo']:.2e} MCA")

    gerar_figura_resultados(resultados, OUT / "fig_simulador_saturacao.png")

    total_litros = sum(r["volume"] for r in resultados.values()) * 1000
    print(f"\nTotal (soma das 4 carreiras): {total_litros:.0f} L")
    print(f"\nFigura salva em: {(OUT / 'fig_simulador_saturacao.png').resolve()}")
