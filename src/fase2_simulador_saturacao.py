"""
PHASE 2: Rapid Sand Filter Saturation Simulator
COC351 - Computational Mathematics - UFRJ Poli 2026.1
Authors: Gustavo Maia de Araujo 
         Gilson Batista Machado Martins 

Applied numerical methods:
    1. Root Finding -- Secant Method
    2. Numerical Integration -- Composite Simpson's 1/3 Rule

Objective: find the collapse instant (t*) of the Mazagao WTP rapid sand filter
and the treated water volume up to that instant, considering oscillating
pump flow and seasonal calibration of the pore blocking factor (beta).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

OUT = Path("outputs")
OUT.mkdir(parents=True, exist_ok=True)

# --- Physical parameters ---
EPS_0    = 0.387        # initial porosity
MU       = 1.002e-3     # dynamic viscosity [Pa*s]
L_BED    = 0.60         # bed thickness [m]
D_P      = 1.0e-3       # grain diameter [m]
RHO      = 998.0        # water density [kg/m3]
G        = 9.81         # gravitational acceleration [m/s2]
A_FILTER = 17.28        # filter area [m2]
DP0_MWC  = 0.30         # initial pressure offset [mWC - meters of water column]
DP_MAX   = 3.0          # clogging / backwashing trigger [mWC]

# Calibrated hydrological seasons (k, Q_base in L/s)
# k [1/h]: porosity reduction rate, calibrated offline via
#          k = beta * TSS * Q / (A * L * rho_sludge), TSS = 0.85 * NTU
# See Section 3.2 of the report for the full calibration justification.
SEASONS = {
    "Flood":      {"k": 0.0175, "Q_base": 21.5, "beta": 38},
    "High_Water": {"k": 0.0115, "Q_base": 20.0, "beta": 25},
    "Ebb":        {"k": 0.0075, "Q_base": 19.5, "beta": 16},
    "Dry":        {"k": 0.0058, "Q_base": 18.0, "beta": 13},
}


# ============================================================================
# PHYSICAL MODEL
# ============================================================================

def dyn_Q(t, Q_base):
    """
    Dynamic flow with sinusoidal oscillation, representing the real
    variation of the pump's variable frequency drive:
        Q(t) = Q_base + 1.5*sin(t)   [L/s]
    Returns in m3/s.
    """
    return (Q_base + 1.5 * np.sin(t)) * 1e-3  # L/s -> m3/s


def ergun(eps, v):
    """
    Ergun-Botari equation for head loss in a granular bed [Pa].

        dP = [150*mu*L*(1-eps)^2 / (dp^2*eps^3)] * v      <- viscous (Darcy)
           + [1.75*rho*L*(1-eps) / (dp*eps^3)] * v^2       <- inertial (Forchheimer)

    Parameters
    ----------
    eps : current bed porosity [dimensionless]
    v   : superficial velocity [m/s]
    """
    if eps <= 0.005:
        return np.inf
    t_vis = 150 * MU * L_BED * (1 - eps) ** 2 / (D_P ** 2 * eps ** 3)
    t_ine = 1.75 * RHO * L_BED * (1 - eps) / (D_P * eps ** 3) * v
    return (t_vis + t_ine) * v


def f_root(t, k, Q_base):
    """
    Target function for root finding:
        f(t) = DeltaP_total(t) - DeltaP_max

    f(t*) = 0  <=>  the filter reached the 3.0 mWC limit (collapse).
    """
    eps = max(EPS_0 - k * t, 0.005)
    v   = dyn_Q(t, Q_base) / A_FILTER
    dP  = ergun(eps, v) / (RHO * G) + DP0_MWC  # Pa -> mWC
    return dP - DP_MAX


# ============================================================================
# METHOD 1 -- ROOT FINDING: SECANT METHOD
# ============================================================================

def secant_method(f, t0=5.0, t1=6.0, tol=1e-8, max_iter=200):
    """
    Secant method with adaptive bracketing pre-conditioner.

    Central iteration:
        t_{n+1} = t_n - f(t_n) * (t_n - t_{n-1}) / (f(t_n) - f(t_{n-1}))

    Justification: the numerical derivative of f(t) would amplify the sinusoidal
    noise of Q(t), causing the Newton-Raphson method to diverge.
    The Secant uses only f evaluations, without depending on derivatives,
    ensuring stability under the oscillatory disturbance.

    Bracketing pre-conditioner:
        The fixed guesses t0=5h and t1=6h start before the root (f<0 in
        both). The pure secant would diverge in this case. The pre-conditioner
        advances the window with an adaptive step (x1.5) until finding
        f(t0)*f(t1) < 0 (sign change), followed by 3 bisection steps
        to refine the window before applying the Secant.

    Parameters
    ----------
    f        : target function f(t*) = 0
    t0, t1   : initial guesses [h] (as specified: 5.0 and 6.0)
    tol      : convergence tolerance |f(t)| < tol [mWC]
    max_iter : total iteration limit

    Returns
    -------
    dict with: t_star, iterations (Secant phase only), residual, converged
    """
    f0, f1 = f(t0), f(t1)
    iters = 2

    # Phase 1: adaptive bracketing
    step = (t1 - t0)
    while f0 * f1 > 0 and iters < max_iter:
        step *= 1.5
        t1, f1 = t0 + step, f(t0 + step)
        iters += 1

    if f0 * f1 > 0:
        return {"t_star": None, "iterations": iters,
                "residual": None, "converged": False,
                "message": "Bracketing failed."}

    # Phase 1b: bisection refinement (3 steps)
    for _ in range(3):
        t_m, f_m = (t0 + t1) / 2, f((t0 + t1) / 2)
        iters += 1
        if f0 * f_m < 0:
            t1, f1 = t_m, f_m
        else:
            t0, f0 = t_m, f_m

    # Phase 2: Secant method itself
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
            return {"t_star": t2, "iterations": iter_sec,
                    "residual": abs(f2), "converged": True}
        t0, f0 = t1, f1
        t1, f1 = t2, f2

    return {"t_star": t1, "iterations": iter_sec,
            "residual": abs(f1), "converged": False}


# ============================================================================
# METHOD 2 -- NUMERICAL INTEGRATION: COMPOSITE SIMPSON'S 1/3 RULE
# ============================================================================

def simpson_volume(t_star, Q_base, n=None):
    """
    Calculates the treated water volume using the Composite Simpson's 1/3 Rule:

        V = integral from 0 to t* of Q(t) dt
         ~= (h/3) * [Q0 + 4*Q1 + 2*Q2 + 4*Q3 + ... + Qn]

    with uniform step h = t*/n (even n). Truncation error: O(h^4).

    Parameters
    ----------
    t_star  : collapse instant [h]
    Q_base  : base flow [L/s]
    n       : number of subintervals (even); default: ceil(t*/0.25)

    Returns
    -------
    Treated volume [m3]
    """
    if n is None:
        n = int(np.ceil(t_star / 0.25))
    if n % 2 != 0:
        n += 1
    n = max(n, 4)

    t_vec = np.linspace(0.0, t_star, n + 1)
    h     = t_vec[1] - t_vec[0]
    Q_vec = np.array([dyn_Q(t, Q_base) * 3600 for t in t_vec])  # m3/h

    # Simpson's coefficients: 1, 4, 2, 4, 2, ..., 4, 1
    coef = np.ones(n + 1)
    coef[1:-1:2] = 4.0
    coef[2:-2:2] = 2.0

    return (h / 3.0) * np.dot(coef, Q_vec)


# ============================================================================
# VISUALIZATION
# ============================================================================

def generate_results_figure(results, output_path):
    """
    Generates a figure with 3 panels:
      (A) Evolution of DeltaP(t) per season
      (B) Porosity decay eps(t)
      (C) Computational efficiency -- Secant iterations
    """
    fig = plt.figure(figsize=(20, 11))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1.5, 1])
    ax_dp  = fig.add_subplot(gs[0, 0])
    ax_eps = fig.add_subplot(gs[1, 0], sharex=ax_dp)
    ax_bar = fig.add_subplot(gs[:, 1])

    colors = {"Flood": "#002d72", "High_Water": "#1f77b4",
              "Ebb": "#ff7f0e", "Dry": "#2ca02c"}

    for name, res in results.items():
        color = colors[name]
        t_vec = np.linspace(0, res["t_star"], 300)
        dP_vec  = [f_root(t, res["k"], res["Q_base"]) + DP_MAX for t in t_vec]
        eps_vec = [max(EPS_0 - res["k"] * t, 0.005) for t in t_vec]

        ax_dp.plot(t_vec, dP_vec, color=color, lw=2.2,
                   label=f"{name} (t*={res['t_star']:.1f}h)")
        ax_dp.plot(res["t_star"], DP_MAX, "o", color=color, ms=10)

        ax_eps.plot(t_vec, eps_vec, color=color, lw=2.2)

    ax_dp.axhline(DP_MAX, color="red", ls="--", label="Limit: 3.0 mWC")
    ax_dp.set_ylabel("Head loss (mWC)")
    ax_dp.set_title("(A) Head Loss Evolution", fontweight="bold")
    ax_dp.legend(fontsize=9)
    ax_dp.grid(True, ls="--", alpha=0.5)

    ax_eps.set_xlabel("Time (hours)")
    ax_eps.set_ylabel("Porosity")
    ax_eps.set_title("(B) Porosity Decay", fontweight="bold")
    ax_eps.grid(True, ls="--", alpha=0.5)

    names  = list(results.keys())
    iters  = [results[n]["iterations"] for n in names]
    colors_b = [colors[n] for n in names]
    bars = ax_bar.bar(names, iters, color=colors_b, alpha=0.85)
    for bar, it in zip(bars, iters):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, it + 0.1,
                    str(it), ha="center", fontweight="bold", fontsize=14)
    ax_bar.set_ylabel("Secant Iterations")
    ax_bar.set_title("(C) Computational Efficiency", fontweight="bold")
    ax_bar.grid(True, ls="--", alpha=0.5, axis="y")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================================
# MAIN SIMULATION
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 2 -- Filter Saturation Simulator")
    print("=" * 60)

    results = {}

    for name, p in SEASONS.items():
        k, Qb = p["k"], p["Q_base"]
        res = secant_method(lambda t: f_root(t, k, Qb))
        vol = simpson_volume(res["t_star"], Qb) if res["converged"] else 0

        results[name] = {
            "t_star": res["t_star"], "k": k, "Q_base": Qb,
            "iterations": res["iterations"], "residual": res["residual"],
            "volume": vol,
        }

        print(f"\n[{name}]")
        print(f"  t* = {res['t_star']:.2f} h")
        print(f"  Volume  = {vol:.0f} m3 ({vol*1000:.0f} L)")
        print(f"  Secant  = {res['iterations']} iterations | "
              f"residual = {res['residual']:.2e} mWC")

    generate_results_figure(results, OUT / "fig_saturation_simulator.png")

    total_liters = sum(r["volume"] for r in results.values()) * 1000
    print(f"\nTotal (sum of the 4 runs): {total_liters:.0f} L")
    print(f"\nFigure saved at: {(OUT / 'fig_saturation_simulator.png').resolve()}")         
