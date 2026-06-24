"""
features.py
------------
Feature engineering for RUL prediction.

Two important domain-correct decisions baked into this file:

1. ROLLING WINDOW FEATURES instead of raw point sensor readings.
   A single noisy sensor reading at cycle t tells you little. But the
   rolling mean/std/slope over the last N cycles captures the actual
   degradation trend - which is the real physical signal we care about.

2. SPLIT BY ENGINE, NOT BY ROW.
   If you randomly split rows 80/20, you leak information: rows from the
   same engine's trajectory end up in both train and validation, so the
   model partially "memorizes" that engine's specific degradation curve.
   We split whole engines into train/val so validation truly tests
   generalization to unseen engines - this is the single most common
   mistake people make on this dataset.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data_loader import load_subset, clip_rul, SENSOR_NAMES, FD001_CONSTANT_SENSORS

ROLLING_WINDOW = 5


def add_rolling_features(df: pd.DataFrame, sensors: list, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """
    Adds rolling mean, rolling std, and a short-horizon slope (rate of change)
    per engine, per sensor. Computed independently per unit_nr so we never
    blend cycles from two different engines together.
    """
    df = df.sort_values(["unit_nr", "time_cycles"]).copy()
    grouped = df.groupby("unit_nr")

    for sensor in sensors:
        df[f"{sensor}_rollmean"] = grouped[sensor].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        df[f"{sensor}_rollstd"] = grouped[sensor].transform(
            lambda x: x.rolling(window, min_periods=1).std().fillna(0)
        )
        # simple slope: value now minus value `window` cycles ago, normalized
        df[f"{sensor}_slope"] = grouped[sensor].transform(
            lambda x: (x - x.shift(window)).fillna(0) / window
        )
    return df


def build_feature_set(subset: str = "FD001", rul_cap: int = 125):
    """
    Full pipeline: load raw data -> drop dead sensors -> add rolling features
    -> cap RUL -> split engines into train/val -> return model-ready arrays.
    """
    train_raw, test_raw, rul_truth = load_subset(subset)

    active_sensors = [s for s in SENSOR_NAMES if s not in FD001_CONSTANT_SENSORS]

    train_feat = add_rolling_features(train_raw, active_sensors)
    train_feat = clip_rul(train_feat, upper_limit=rul_cap)

    feature_cols = active_sensors + [c for c in train_feat.columns if "_rollmean" in c or "_rollstd" in c or "_slope" in c]

    # --- split by ENGINE id, not by row ---
    unit_ids = train_feat["unit_nr"].unique()
    train_units, val_units = train_test_split(unit_ids, test_size=0.2, random_state=42)

    train_df = train_feat[train_feat["unit_nr"].isin(train_units)]
    val_df = train_feat[train_feat["unit_nr"].isin(val_units)]

    X_train, y_train = train_df[feature_cols], train_df["RUL"]
    X_val, y_val = val_df[feature_cols], val_df["RUL"]

    # --- prepare the official test set (last cycle of each engine = "now") ---
    test_feat = add_rolling_features(test_raw, active_sensors)
    test_last = test_feat.groupby("unit_nr").last().reset_index()
    X_test = test_last[feature_cols]
    y_test = clip_rul(rul_truth, upper_limit=rul_cap, col="RUL")["RUL"]

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "feature_cols": feature_cols,
        "active_sensors": active_sensors,
    }


if __name__ == "__main__":
    data = build_feature_set("FD001")
    print("Feature columns:", len(data["feature_cols"]))
    print("Train rows:", data["X_train"].shape, "| Val rows:", data["X_val"].shape, "| Test engines:", data["X_test"].shape)
    print("\nSample features:\n", data["X_train"].head(3))
