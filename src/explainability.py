"""
explainability.py
-------------------
SHAP-based explainability for the XGBoost RUL model.

Why this matters for the "root cause" requirement:
A maintenance engineer doesn't just want "17 cycles left" - they want to
know WHY. SHAP decomposes each individual prediction into per-sensor
contributions, so we can say: "this engine's RUL dropped because s_4
(LPT outlet temperature) and s_11 (HPC outlet pressure) are trending
abnormally" - which maps directly to a root-cause hypothesis
(e.g. HPC degradation, the actual fault mode simulated in FD001).
"""
import sys
import pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from features import build_feature_set

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Human-readable sensor descriptions (from NASA C-MAPSS documentation /
# widely-used community mapping). Helps translate "s_11" into something
# a mechanical engineer / interviewer immediately understands.
SENSOR_DESCRIPTIONS = {
    "s_1": "Fan inlet temperature (T2)",
    "s_2": "LPC outlet temperature (T24)",
    "s_3": "HPC outlet temperature (T30)",
    "s_4": "LPT outlet temperature (T50)",
    "s_5": "Fan inlet pressure (P2)",
    "s_6": "Bypass-duct pressure (P15)",
    "s_7": "HPC outlet pressure (P30)",
    "s_8": "Physical fan speed (Nf)",
    "s_9": "Physical core speed (Nc)",
    "s_10": "Engine pressure ratio (epr)",
    "s_11": "HPC outlet static pressure (Ps30)",
    "s_12": "Ratio of fuel flow to Ps30 (phi)",
    "s_13": "Corrected fan speed (NRf)",
    "s_14": "Corrected core speed (NRc)",
    "s_15": "Bypass ratio (BPR)",
    "s_16": "Burner fuel-air ratio (farB)",
    "s_17": "Bleed enthalpy (htBleed)",
    "s_18": "Demanded fan speed (Nf_dmd)",
    "s_19": "Demanded corrected fan speed (PCNfR_dmd)",
    "s_20": "HPT coolant bleed (W31)",
    "s_21": "LPT coolant bleed (W32)",
}


def base_sensor(feature_name: str) -> str:
    """Strips '_rollmean' / '_rollstd' / '_slope' suffix to get back to s_N."""
    for suffix in ["_rollmean", "_rollstd", "_slope"]:
        if feature_name.endswith(suffix):
            return feature_name[: -len(suffix)]
    return feature_name


def describe(feature_name: str) -> str:
    s = base_sensor(feature_name)
    return SENSOR_DESCRIPTIONS.get(s, s)


def run_shap_analysis():
    with open(MODEL_DIR / "xgb_rul_model.pkl", "rb") as f:
        xgb = pickle.load(f)

    data = build_feature_set("FD001", rul_cap=125)
    X_test = data["X_test"]

    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer(X_test)

    # ---- Global summary: which sensors matter most across all engines ----
    fig = plt.figure(figsize=(9, 7))
    shap.summary_plot(shap_values, X_test, show=False, max_display=12)
    plt.title("Global feature importance (SHAP) - FD001 RUL model")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "shap_summary.png", bbox_inches="tight")
    plt.close()
    print("Saved shap_summary.png")

    # ---- Per-engine root cause: pick the engine with the LOWEST predicted RUL ----
    preds = xgb.predict(X_test)
    worst_idx = int(np.argmin(preds))
    worst_unit = data["X_test"].iloc[worst_idx].name

    fig = plt.figure(figsize=(9, 5))
    shap.plots.waterfall(shap_values[worst_idx], show=False, max_display=10)
    plt.title(f"Root cause breakdown - engine at row {worst_idx} (predicted RUL={preds[worst_idx]:.0f})")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "shap_root_cause_example.png", bbox_inches="tight")
    plt.close()
    print("Saved shap_root_cause_example.png")

    # ---- Translate top contributing features into a human-readable root cause note ----
    contributions = pd.Series(
        shap_values.values[worst_idx], index=data["feature_cols"]
    ).sort_values(key=abs, ascending=False)

    print(f"\nROOT CAUSE REPORT - engine (test row {worst_idx}), predicted RUL = {preds[worst_idx]:.1f} cycles")
    print("Top contributing sensors (pushing RUL DOWN = degradation signal):")
    for feat, val in contributions.head(5).items():
        direction = "lowering RUL (degradation)" if val < 0 else "raising RUL (healthy signal)"
        print(f"  - {describe(feat):45s} | SHAP contribution: {val:+.2f}  -> {direction}")

    return shap_values, data, preds


if __name__ == "__main__":
    run_shap_analysis()
