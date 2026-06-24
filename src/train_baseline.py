"""
train_baseline.py
-------------------
Trains Random Forest and XGBoost regressors to predict RUL, and evaluates
them with TWO metrics:

1. RMSE - standard regression error, in cycles.

2. PHM08 Score - the official asymmetric scoring function from the NASA
   PHM08 prognostics competition that this dataset originates from.
   It penalizes LATE predictions (predicting more life than the engine
   actually has left) much more harshly than EARLY predictions, because
   in real maintenance, missing a failure is catastrophic while an early
   maintenance call just costs you some unused engine life. RMSE alone
   doesn't capture this - a model can have decent RMSE while still being
   dangerously over-optimistic on the engines that matter most.
"""
import sys
import pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

from features import build_feature_set

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def phm08_score(y_true, y_pred):
    """
    Official PHM08 asymmetric scoring function.
    d = predicted - actual
        d < 0 (early/under-predicted RUL, i.e. conservative)  -> exp(-d/13)  - 1
        d >= 0 (late/over-predicted RUL, i.e. dangerous)      -> exp( d/10)  - 1
    Lower is better. Even a handful of badly "late" predictions blow this up fast.
    """
    d = np.array(y_pred) - np.array(y_true)
    score = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    return float(np.sum(score))


def evaluate(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    score = phm08_score(y_true, y_pred)
    print(f"\n--- {name} ---")
    print(f"  RMSE       : {rmse:.2f} cycles")
    print(f"  MAE        : {mae:.2f} cycles")
    print(f"  PHM08 Score: {score:.1f}  (lower=better, penalizes late predictions hard)")
    return {"rmse": rmse, "mae": mae, "phm08": score}


def main():
    data = build_feature_set("FD001", rul_cap=125)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    results = {}

    # ---------------- Random Forest ----------------
    rf = RandomForestRegressor(
        n_estimators=200, max_depth=12, min_samples_leaf=5,
        random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    print("\n" + "=" * 55)
    print("RANDOM FOREST")
    print("=" * 55)
    evaluate("RF - Validation (held-out engines)", y_val, rf.predict(X_val))
    results["rf_test"] = evaluate("RF - Official Test Set", y_test, rf.predict(X_test))

    # ---------------- XGBoost ----------------
    xgb = XGBRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        objective="reg:squarederror"
    )
    xgb.fit(X_train, y_train)
    print("\n" + "=" * 55)
    print("XGBOOST")
    print("=" * 55)
    evaluate("XGB - Validation (held-out engines)", y_val, xgb.predict(X_val))
    results["xgb_test"] = evaluate("XGB - Official Test Set", y_test, xgb.predict(X_test))

    # ---------------- Feature importance (quick view, SHAP comes next) ----------------
    importances = pd.Series(xgb.feature_importances_, index=data["feature_cols"]).sort_values(ascending=False)
    print("\nTop 10 features driving XGBoost RUL predictions:")
    print(importances.head(10))

    # ---------------- Save artifacts ----------------
    with open(MODEL_DIR / "rf_rul_model.pkl", "wb") as f:
        pickle.dump(rf, f)
    with open(MODEL_DIR / "xgb_rul_model.pkl", "wb") as f:
        pickle.dump(xgb, f)
    with open(MODEL_DIR / "feature_cols.pkl", "wb") as f:
        pickle.dump(data["feature_cols"], f)

    print(f"\nModels saved to {MODEL_DIR}")
    return results, rf, xgb, data


if __name__ == "__main__":
    main()
