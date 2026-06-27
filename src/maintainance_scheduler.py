"""
maintenance_scheduler.py
--------------------------
Turns RUL predictions + failure probability into an actionable maintenance
decision per engine: WHEN to schedule maintenance, and HOW URGENT it is -
the actual deliverable a maintenance planner needs, not a bare number.

Two ideas borrowed from classical reliability engineering (the textbook
"age-based replacement policy"), but using the Random Forest's own tree
ensemble as an empirical RUL distribution instead of fitting a Weibull:

1. EXPECTED-COST OPTIMIZATION.
   Every candidate maintenance time t trades off two risks:
     - Servicing too LATE risks an unplanned failure (expensive: downtime,
       safety, secondary damage).
     - Servicing too EARLY wastes remaining useful life that was still
       good.
   We search candidate t and pick the one minimizing:
       E[cost(t)] = P(RUL < t) * COST_UNPLANNED_FAILURE
                    + COST_PLANNED_MAINTENANCE
                    + COST_PER_CYCLE_IDLE * max(median_RUL - t, 0)
   P(RUL < t) comes from the RF's 200 individual trees - each gives a
   slightly different RUL estimate for the same engine, and the spread of
   those 200 numbers is a genuine (if rough) uncertainty distribution,
   essentially free, from a model already trained in Phase 1.

2. URGENCY TIERS.
   A simple, explainable rule combining RUL + failure probability into
   CRITICAL / URGENT / PLAN / HEALTHY - the categorical label a fleet
   manager actually scans for, backed by the cost-optimal cycle as the
   precise number behind it.

NOTE on the cost constants below: they are illustrative placeholder values
(a real deployment would source these from actual downtime/spare-part/
labor costs). The point of this module is the DECISION FRAMEWORK - they're
exposed as constants specifically so they're easy to swap for real figures.
"""
import sys
import pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from features import build_feature_set
from explainability import describe

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "plots"
REPORT_DIR = Path(__file__).resolve().parent.parent / "outputs"

# --- illustrative cost assumptions (swap with real figures in a real deployment) ---
COST_UNPLANNED_FAILURE = 50_000
COST_PLANNED_MAINTENANCE = 5_000
COST_PER_CYCLE_IDLE = 60

PRIMARY_HORIZON = 30
MAINTENANCE_GRID = np.arange(0, 150, 1)


def load_artifacts():
    with open(MODEL_DIR / "rf_rul_model.pkl", "rb") as f:
        rf = pickle.load(f)
    with open(MODEL_DIR / "xgb_rul_model.pkl", "rb") as f:
        xgb = pickle.load(f)
    with open(MODEL_DIR / f"xgb_failure_clf_{PRIMARY_HORIZON}.pkl", "rb") as f:
        clf = pickle.load(f)
    with open(MODEL_DIR / "feature_cols.pkl", "rb") as f:
        feature_cols = pickle.load(f)
    return rf, xgb, clf, feature_cols


def rf_tree_distribution(rf, X_row):
    """Each tree's individual RUL prediction for one engine - a free,
    honest uncertainty distribution from the ensemble we already trained.
    Cast to a bare numpy array (no column names) before predicting, since
    the trees were fit without feature names and sklearn otherwise warns
    on every single call (200 trees x 100 engines = thousands of warnings)."""
    X_arr = X_row.values if hasattr(X_row, "values") else X_row
    return np.array([est.predict(X_arr)[0] for est in rf.estimators_])


def optimal_maintenance_cycle(tree_preds, grid=MAINTENANCE_GRID):
    """Grid-search the expected-cost-minimizing maintenance time, using the
    empirical CDF of the RF's tree predictions as P(RUL < t)."""
    median_rul = np.median(tree_preds)

    costs = []
    for t in grid:
        p_fail_before = np.mean(tree_preds < t)
        wasted_life = max(median_rul - t, 0)
        expected_cost = (
            p_fail_before * COST_UNPLANNED_FAILURE
            + COST_PLANNED_MAINTENANCE
            + COST_PER_CYCLE_IDLE * wasted_life
        )
        costs.append(expected_cost)
    costs = np.array(costs)
    best_idx = np.argmin(costs)
    return int(grid[best_idx]), costs, median_rul


def urgency_tier(rul_pred, p_fail):
    if rul_pred <= 10 or p_fail > 0.75:
        return "CRITICAL"
    elif rul_pred <= 30 or p_fail > 0.40:
        return "URGENT"
    elif rul_pred <= 60 or p_fail > 0.15:
        return "PLAN"
    else:
        return "HEALTHY"


def build_fleet_report():
    rf, xgb, clf, feature_cols = load_artifacts()
    data = build_feature_set("FD001", rul_cap=125)
    X_test = data["X_test"].reset_index(drop=True)

    # SHAP for per-engine root cause (reusing the Phase 1 explainer)
    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer(X_test)

    rows = []
    for i in range(len(X_test)):
        X_row = X_test.iloc[[i]]
        rul_pred = float(xgb.predict(X_row)[0])
        p_fail = float(clf.predict_proba(X_row)[0, 1])

        tree_preds = rf_tree_distribution(rf, X_row)
        best_t, costs, median_rul = optimal_maintenance_cycle(tree_preds)

        tier = urgency_tier(rul_pred, p_fail)

        # top SHAP-driven root cause sensor for this engine (most negative = biggest degradation driver)
        contributions = pd.Series(shap_values.values[i], index=feature_cols)
        top_feature = contributions.sort_values().index[0]
        root_cause = describe(top_feature)

        rows.append({
            "engine_id": i + 1,
            "predicted_RUL": round(rul_pred, 1),
            "rf_median_RUL": round(median_rul, 1),
            "rf_p10_RUL": round(float(np.percentile(tree_preds, 10)), 1),
            "rf_p90_RUL": round(float(np.percentile(tree_preds, 90)), 1),
            f"prob_fail_within_{PRIMARY_HORIZON}cyc": round(p_fail, 3),
            "urgency_tier": tier,
            "recommended_maintenance_in_cycles": best_t,
            "top_root_cause_sensor": root_cause,
        })

    report = pd.DataFrame(rows).sort_values("predicted_RUL").reset_index(drop=True)
    return report, X_test, rf


def plot_tier_distribution(report, fname="urgency_tier_distribution.png"):
    order = ["CRITICAL", "URGENT", "PLAN", "HEALTHY"]
    counts = report["urgency_tier"].value_counts().reindex(order).fillna(0)
    colors = {"CRITICAL": "crimson", "URGENT": "darkorange", "PLAN": "gold", "HEALTHY": "seagreen"}
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(counts.index, counts.values, color=[colors[t] for t in counts.index])
    for i, v in enumerate(counts.values):
        ax.text(i, v + 0.5, str(int(v)), ha="center")
    ax.set_title(f"Fleet maintenance urgency - {len(report)} engines (FD001 test set)")
    ax.set_ylabel("Number of engines")
    plt.tight_layout()
    plt.savefig(OUT_DIR / fname, bbox_inches="tight")
    plt.close()
    print(f"Saved {fname}")


def plot_cost_curve_example(report, X_test, rf, fname="maintenance_cost_curve_example.png"):
    """Visualizes the expected-cost optimization for the single most critical
    engine - the plot that makes the optimization tangible and defensible."""
    worst_row = report.iloc[0]
    idx = int(worst_row["engine_id"]) - 1
    X_row = X_test.iloc[[idx]]
    tree_preds = rf_tree_distribution(rf, X_row)
    best_t, costs, median_rul = optimal_maintenance_cycle(tree_preds)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(MAINTENANCE_GRID, costs, color="steelblue")
    ax.axvline(best_t, color="crimson", linestyle="--", label=f"Optimal: maintain at cycle +{best_t}")
    ax.axvline(median_rul, color="gray", linestyle=":", label=f"Median predicted RUL: {median_rul:.0f}")
    ax.set_xlabel("Candidate maintenance time (cycles from now)")
    ax.set_ylabel("Expected cost ($)")
    ax.set_title(f"Expected-cost optimization - Engine #{int(worst_row['engine_id'])} "
                 f"(predicted RUL={worst_row['predicted_RUL']})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / fname, bbox_inches="tight")
    plt.close()
    print(f"Saved {fname}")


if __name__ == "__main__":
    report, X_test, rf = build_fleet_report()

    print("=" * 70)
    print("FLEET MAINTENANCE REPORT (FD001 official test set, 100 engines)")
    print("=" * 70)
    print(report["urgency_tier"].value_counts().reindex(["CRITICAL", "URGENT", "PLAN", "HEALTHY"]).fillna(0))

    print("\nTop 5 most urgent engines:")
    print(report.head(5).to_string(index=False))

    report.to_csv(REPORT_DIR / "fleet_maintenance_report.csv", index=False)
    print(f"\nFull fleet report saved to {REPORT_DIR / 'fleet_maintenance_report.csv'}")

    plot_tier_distribution(report)
    plot_cost_curve_example(report, X_test, rf)