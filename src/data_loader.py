"""
data_loader.py
----------------
Loads the NASA C-MAPSS turbofan engine degradation dataset and computes
Remaining Useful Life (RUL) labels.

Dataset background:
    - Each row = one engine ("unit_nr"), at one operating cycle ("time_cycles")
    - 3 operational settings (altitude, speed, throttle-equivalent)
    - 21 sensor measurements (temperatures, pressures, fan/core speeds, etc.)
    - TRAIN files: each engine runs from healthy -> failure (full trajectory)
    - TEST files: each engine trajectory is cut off BEFORE failure
    - RUL files: ground-truth Remaining Useful Life for each test engine,
      measured from the last recorded cycle in the test file

FD001 = 1 operating condition, 1 fault mode   (100 train / 100 test engines)
FD002 = 6 operating conditions, 1 fault mode  (260 train / 259 test engines)
FD003 = 1 operating condition, 2 fault modes  (100 train / 100 test engines)
FD004 = 6 operating conditions, 2 fault modes (248 train / 249 test engines)
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

INDEX_NAMES = ["unit_nr", "time_cycles"]
SETTING_NAMES = ["setting_1", "setting_2", "setting_3"]
SENSOR_NAMES = [f"s_{i}" for i in range(1, 22)]
COL_NAMES = INDEX_NAMES + SETTING_NAMES + SENSOR_NAMES

# Sensors that are constant (zero variance) in FD001 specifically -
# common in literature to drop these for single-condition subsets.
# We keep them in the loader and let downstream feature selection decide,
# but flag them here for reference.
FD001_CONSTANT_SENSORS = ["s_1", "s_5", "s_6", "s_10", "s_16", "s_18", "s_19"]


def load_subset(subset: str = "FD001", data_dir: Path = DATA_DIR):
    """
    Loads train, test, and RUL truth files for a given C-MAPSS subset.

    Parameters
    ----------
    subset : str
        One of 'FD001', 'FD002', 'FD003', 'FD004'

    Returns
    -------
    train : DataFrame with an added 'RUL' column (piecewise-true RUL per row)
    test  : DataFrame (raw, no RUL column - trajectories are truncated)
    rul_truth : DataFrame with the true RUL at the end of each test trajectory
    """
    train_path = data_dir / f"train_{subset}.txt"
    test_path = data_dir / f"test_{subset}.txt"
    rul_path = data_dir / f"RUL_{subset}.txt"

    train = pd.read_csv(train_path, sep=r"\s+", header=None, names=COL_NAMES)
    test = pd.read_csv(test_path, sep=r"\s+", header=None, names=COL_NAMES)
    rul_truth = pd.read_csv(rul_path, sep=r"\s+", header=None, names=["RUL"])

    train = add_rul_column(train)
    return train, test, rul_truth


def add_rul_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    For TRAINING data only: since every engine runs to failure, the RUL at
    any row = (max cycle for that engine) - (current cycle).

    Example: engine 1 fails at cycle 192.
        At cycle 1   -> RUL = 191
        At cycle 192 -> RUL = 0   (failure point)
    """
    df = df.copy()
    max_cycle = df.groupby("unit_nr")["time_cycles"].transform("max")
    df["RUL"] = max_cycle - df["time_cycles"]
    return df


def clip_rul(df: pd.DataFrame, upper_limit: int = 125, col: str = "RUL") -> pd.DataFrame:
    """
    Applies the standard 'piecewise-linear RUL' trick used in nearly all
    C-MAPSS literature (Heimes 2008, and most follow-on papers).

    Why: a brand-new engine doesn't have a meaningfully different RUL
    whether it has 300 or 320 cycles left - that distinction is unlearnable
    from early-life sensor data because nothing has degraded yet. Capping
    RUL at a ceiling (commonly 125-130) turns the early "healthy plateau"
    into a flat label, which is both more realistic and easier for models
    to learn, instead of forcing them to chase noise.
    """
    df = df.copy()
    df[col] = df[col].clip(upper=upper_limit)
    return df


def get_last_cycle_per_unit(df: pd.DataFrame) -> pd.DataFrame:
    """
    For TEST data: returns only the final recorded row of each engine's
    trajectory - i.e. the most recent sensor snapshot we have, which is
    what we'd actually have available in a real deployed system right
    before we need to predict RUL / failure probability.
    """
    return df.groupby("unit_nr").last().reset_index()


if __name__ == "__main__":
    train, test, rul_truth = load_subset("FD001")
    print("Train shape:", train.shape)
    print("Test shape:", test.shape)
    print("RUL truth shape:", rul_truth.shape)
    print("\nTrain head:\n", train.head())
    print("\nEngines in train:", train["unit_nr"].nunique())
    print("Engines in test:", test["unit_nr"].nunique())
    print("\nRUL stats (train):\n", train["RUL"].describe())
