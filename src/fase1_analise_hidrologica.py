"""
PHASE 1: Data Merging and Seasonal Analysis
COC351 - Computational Mathematics - UFRJ Poli 2026.1
Authors: Gustavo Maia de Araujo 
         Gilson Batista Machado Martins 

Objective: merge precipitation data (INMET A249) with real turbidity
readings from the Mazagao WTP, applying a 24h lag time, and calibrate
the power law equation TRB = k * P^m per hydrological season via curve_fit.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path

# --- Parameters ---
LAG_DAYS   = 1        # Lag time: rain from day D affects turbidity of day D+1
MIN_RAIN   = 0.5      # minimum mm to include in curve_fit
ALPHA      = 0.85     # (mg/L)/NTU -- Abreu et al. 2022
OUT        = Path("outputs")
OUT.mkdir(parents=True, exist_ok=True)

SEASONS_MONTHS = {
    "Flood":      [1, 2, 3],
    "High_Water": [4, 5, 6],
    "Ebb":        [7, 8, 9],
    "Dry":        [10, 11, 12]
}


# --- Load data ---
def load_precipitation(filepath):
    """
    Reads the INMET hourly CSV (encoding ISO-8859-1, separator ';',
    8 header lines), aggregates it into accumulated daily precipitation,
    and applies the LAG_DAYS shift to simulate the hydrological Lag Time.
    """
    df = pd.read_csv(filepath, encoding="ISO-8859-1",
                     sep=";", skiprows=8, decimal=",")
    # Rename for internal English usage while reading original Portuguese columns
    df = df.rename(columns={"Data": "Date"})
    df["Date"] = pd.to_datetime(df["Date"],
                                format="%Y/%m/%d", errors="coerce")
    df = df.dropna(subset=["Date"])
    col_p = "PRECIPITACAO TOTAL, HORARIO (mm)"
    df[col_p] = pd.to_numeric(df[col_p], errors="coerce").fillna(0)
    rain = df.groupby("Date")[col_p].sum().reset_index()
    rain.columns = ["Date", "Rain_mm"]
    # Apply Lag Time: rain from day D -> turbidity of day D+1
    rain["Date_Turbidity"] = rain["Date"] + pd.Timedelta(days=LAG_DAYS)
    return rain[["Date_Turbidity", "Rain_mm"]]


def load_turbidity(filepath):
    """
    Reads the consolidated real turbidity CSV and calculates the daily mean
    for compatibility with the precipitation dataset.
    """
    df = pd.read_csv(filepath)
    df = df.rename(columns={"Data": "Date", "Turbidez": "Turbidity"})
    df["Date"] = pd.to_datetime(df["Date"])
    df["Turbidity"] = pd.to_numeric(df["Turbidity"], errors="coerce")
    df = df.dropna(subset=["Turbidity"])
    df = df[df["Turbidity"] > 0]
    return (df.groupby("Date")["Turbidity"]
              .mean()
              .reset_index()
              .rename(columns={"Turbidity": "Avg_Turbidity"}))


# --- Merge data ---
def merge_data(df_rain, df_turb):
    """
    Performs an inner join between precipitation (with lag applied) and daily
    turbidity, classifying each day by hydrological season.
    """
    df = df_turb.merge(
        df_rain.rename(columns={"Date_Turbidity": "Date"}),
        on="Date", how="inner"
    )
    df["Month"]  = df["Date"].dt.month
    df["Season"] = df["Month"].map(
        lambda m: next(
            (s for s, ms in SEASONS_MONTHS.items() if m in ms), "?"
        )
    )
    return df.sort_values("Date").reset_index(drop=True)


# --- Power law model ---
def power_law(P, k, m):
    """Power law model: TRB = k * P^m  (P > 0)."""
    return k * np.power(np.maximum(P, 0.1), m)


def calibrate(df_season, name):
    """
    Calibrates TRB = k * P^m via curve_fit for days with rain >= MIN_RAIN.
    Returns a dict with k, m, R2, N samples or None upon failure.
    """
    sub = df_season[df_season["Rain_mm"] >= MIN_RAIN]
    if len(sub) < 5:
        print(f"  [WARNING] {name}: insufficient points ({len(sub)}).")
        return None
    try:
        popt, _ = curve_fit(power_law, sub["Rain_mm"], sub["Avg_Turbidity"],
                            p0=[50, 0.3],
                            bounds=([0.001, -5], [10000, 5]),
                            maxfev=5000)
        T_pred = power_law(sub["Rain_mm"].values, *popt)
        ss_res = np.sum((sub["Avg_Turbidity"].values - T_pred) ** 2)
        ss_tot = np.sum((sub["Avg_Turbidity"].values
                         - sub["Avg_Turbidity"].mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        return {"k": popt[0], "m": popt[1], "R2": r2, "N": len(sub)}
    except Exception as e:
        print(f"  [WARNING] {name}: curve_fit failed ({e}).")
        return None


# --- Visualization ---
def plot_seasonal_dispersion(df, parameters, output_path):
    """
    Generates a figure with 4 subplots (2x2), one per hydrological season,
    showing the Rain x Turbidity dispersion and the fitted trend curve.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    order = list(SEASONS_MONTHS.keys())

    for ax, season in zip(axes.flatten(), order):
        sub = df[df["Season"] == season]
        ax.scatter(sub["Rain_mm"], sub["Avg_Turbidity"],
                   alpha=0.6, s=40, label=f"Real data (n={len(sub)})")

        params = parameters.get(season)
        if params:
            P_range = np.linspace(0.1, max(sub["Rain_mm"].max(), 1), 200)
            T_fit = power_law(P_range, params["k"], params["m"])
            ax.plot(P_range, T_fit, "--", lw=2,
                    label=f"TRB={params['k']:.1f}*P^{params['m']:.3f} "
                          f"(R2={params['R2']:.3f})")

        ax.set_title(f"{season}  ({SEASONS_MONTHS[season][0]}-{SEASONS_MONTHS[season][-1]})")
        ax.set_xlabel("Accumulated precipitation D-1 (mm)")
        ax.set_ylabel("Daily average turbidity (NTU)")
        ax.legend(fontsize=8)
        ax.grid(True, ls="--", alpha=0.5)

    fig.suptitle("Precipitation-Turbidity Relationship by Hydrological Season",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# --- Export CSV ---
def export_csv(df, filepath):
    """Exports the final merged table with season metadata."""
    out = df[["Date", "Month", "Season",
              "Avg_Turbidity", "Rain_mm"]].copy()
    out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")
    out.to_csv(filepath, index=False, sep=";",
               decimal=",", encoding="utf-8-sig")


# --- Main pipeline ---
if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 1 -- Data Merging and Seasonal Analysis")
    print("=" * 60)

    df_rain = load_precipitation(
        "INMET_N_AP_A249_MACAPA_01-01-2025_A_31-12-2025.CSV")
    df_turb = load_turbidity("consolidado_turbidez_real.csv")
    df      = merge_data(df_rain, df_turb)

    print(f"\nMerged days: {len(df)}")

    parameters = {}
    for season in ["Flood", "High_Water", "Ebb", "Dry"]:
        sub    = df[df["Season"] == season]
        params = calibrate(sub, season)
        parameters[season] = params
        if params:
            print(f"  {season}: k={params['k']:.2f}, m={params['m']:.4f}, "
                  f"R2={params['R2']:.3f}, N={params['N']}")

    plot_seasonal_dispersion(df, parameters, OUT / "fig1_seasonal_dispersion.png")
    export_csv(df, OUT / "merged_real_data_mazagao.csv")

    print(f"\nFiles saved at: {OUT.resolve()}")
    print("  - fig1_seasonal_dispersion.png")
    print("  - merged_real_data_mazagao.csv")
