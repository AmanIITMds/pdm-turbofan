"""
failure_probability.py
------------------------
Trains classifiers that directly answer the question a maintenance engineer
actually asks: "what's the probability this engine fails within the next W
cycles?" - rather than deriving probability indirectly from the RUL
regressor built in Phase 1.

Why a SEPARATE classifier instead of just thresholding the RUL prediction?
A regressor optimized for RMSE across the full 0-125 cycle range isn't
necessarily well-calibrated right at the specific boundary that matters
operationally (e.g. "30 cycles"). A classifier trained explicitly on that
boundary, evaluated with a calibration curve, gives a probability you can
actually trust for the cost-based decision in maintenance_scheduler.py.

We train classifiers for three maintenance-relevant horizons (15/30/45
cycles) to show the approach generalizes. W=30 is treated as the primary
"scheduled maintenance window" used downstream.
"""
import sys
import pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.calibration import calibration_curve
from xgboost import XGBClassifier

from features import build_feature_set

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "plots"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = [15, 30, 45]
PRIMARY_HORIZON = 30


def make_binary_labels(y, horizon):
    return (y <= horizon).astype(int)


def train_classifier(X_train, y_train_bin):
    clf = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        eval_metric="logloss"
    )
    clf.fit(X_train, y_train_bin)
    return clf


def evaluate_classifier(name, clf, X, y_true_bin):
    probs = clf.predict_proba(X)[:, 1]
    auc = roc_auc_score(y_true_bin, probs)
    ap = average_precision_score(y_true_bin, probs)
    print(f"  {name}: ROC-AUC={auc:.3f} | Avg.Precision={ap:.3f} | "
          f"positives={int(y_true_bin.sum())}/{len(y_true_bin)}")
    return probs, auc, ap


def plot_calibration(results, fname="failure_probability_calibration.png"):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    for horizon, (y_true, probs) in results.items():
        frac_pos, mean_pred = calibration_curve(y_true, probs, n_bins=6, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", label=f"Fail within {horizon} cycles")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed failure rate")
    ax.set_title("Calibration: predicted failure probability vs reality\n(Official test set)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / fname, bbox_inches="tight")
    plt.close()
    print(f"Saved {fname}")


def main():
    data = build_feature_set("FD001", rul_cap=125)
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]

    calib_results = {}
    models = {}

    print("=" * 60)
    print("FAILURE PROBABILITY CLASSIFIERS (multi-horizon)")
    print("=" * 60)
    for horizon in HORIZONS:
        y_train_bin = make_binary_labels(y_train, horizon)
        y_test_bin = make_binary_labels(y_test, horizon)

        clf = train_classifier(X_train, y_train_bin)
        print(f"\nHorizon: fail within {horizon} cycles")
        probs, auc, ap = evaluate_classifier("Official test set", clf, X_test, y_test_bin)

        calib_results[horizon] = (y_test_bin, probs)
        models[horizon] = clf

        if horizon == PRIMARY_HORIZON:
            with open(MODEL_DIR / f"xgb_failure_clf_{horizon}.pkl", "wb") as f:
                pickle.dump(clf, f)
            print(f"  -> Saved as PRIMARY model (xgb_failure_clf_{horizon}.pkl)")

    plot_calibration(calib_results)
    return models, data


if __name__ == "__main__":
    main()