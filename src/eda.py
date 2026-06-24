"""
eda.py
-------
Exploratory analysis on the C-MAPSS FD001 dataset.
Generates the core plots every PdM report/notebook needs:
  1. Degradation trajectories of key sensors across engine life
  2. Sensor variance check (find dead/constant sensors -> drop candidates)
  3. Correlation heatmap of sensors
  4. RUL distribution
  5. Operating settings sanity check (confirms FD001 = single condition)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import load_subset, SENSOR_NAMES, SETTING_NAMES

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110


def plot_sensor_trajectories(train, sensors, n_engines=8, fname="sensor_trajectories.png"):
    """Plot how a handful of sensors evolve over an engine's life for several engines.
    This is the single most important PdM plot: it's how you visually confirm
    which sensors actually carry a degradation signal vs which are just noise."""
    sample_units = train["unit_nr"].drop_duplicates().sample(n_engines, random_state=42)
    fig, axes = plt.subplots(len(sensors), 1, figsize=(9, 3 * len(sensors)), sharex=False)
    if len(sensors) == 1:
        axes = [axes]
    for ax, sensor in zip(axes, sensors):
        for unit in sample_units:
            sub = train[train["unit_nr"] == unit]
            ax.plot(sub["time_cycles"], sub[sensor], alpha=0.7, linewidth=1)
        ax.set_title(f"{sensor} vs operating cycle ({n_engines} sample engines)")
        ax.set_xlabel("time_cycles (engine life)")
        ax.set_ylabel(sensor)
    plt.tight_layout()
    plt.savefig(OUT_DIR / fname, bbox_inches="tight")
    plt.close()
    print(f"Saved {fname}")


def plot_sensor_variance(train, fname="sensor_variance.png"):
    """Identify near-constant sensors - these carry no information for a
    single-operating-condition dataset like FD001 and are safe to drop."""
    variances = train[SENSOR_NAMES].var().sort_values()
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["crimson" if v < 1e-6 else "steelblue" for v in variances]
    ax.barh(variances.index, variances.values, color=colors)
    ax.set_xscale("symlog")
    ax.set_xlabel("Variance (log scale)")
    ax.set_title("Sensor variance - red bars are near-constant (drop candidates)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / fname, bbox_inches="tight")
    plt.close()
    near_constant = variances[variances < 1e-6].index.tolist()
    print(f"Saved {fname}")
    print("Near-constant sensors (candidates to drop):", near_constant)
    return near_constant


def plot_correlation_heatmap(train, sensors, fname="sensor_correlation.png"):
    corr = train[sensors + ["RUL"]].corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False, ax=ax)
    ax.set_title("Sensor correlation matrix (incl. RUL)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / fname, bbox_inches="tight")
    plt.close()
    print(f"Saved {fname}")
    print("\nSensors most correlated with RUL:")
    print(corr["RUL"].drop("RUL").sort_values(key=abs, ascending=False).head(8))


def plot_rul_distribution(train, fname="rul_distribution.png"):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(train["RUL"], bins=40, kde=True, ax=ax, color="darkorange")
    ax.set_title("Distribution of RUL labels across all training rows (FD001)")
    ax.set_xlabel("Remaining Useful Life (cycles)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / fname, bbox_inches="tight")
    plt.close()
    print(f"Saved {fname}")


def plot_settings_sanity_check(train, fname="operating_settings.png"):
    """Confirms FD001 truly has one operating condition (settings barely vary)."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, setting in zip(axes, SETTING_NAMES):
        ax.hist(train[setting], bins=30, color="seagreen")
        ax.set_title(setting)
    plt.suptitle("Operating settings distribution (FD001 should look ~single-mode)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / fname, bbox_inches="tight")
    plt.close()
    print(f"Saved {fname}")


def plot_engine_life_lengths(train, fname="engine_life_lengths.png"):
    life_lengths = train.groupby("unit_nr")["time_cycles"].max().sort_values()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(range(len(life_lengths)), life_lengths.values, color="slateblue")
    ax.set_xlabel("Engine (sorted by lifespan)")
    ax.set_ylabel("Total cycles until failure")
    ax.set_title(f"Engine lifespans (FD001) - min={life_lengths.min()}, "
                 f"max={life_lengths.max()}, mean={life_lengths.mean():.0f}")
    plt.tight_layout()
    plt.savefig(OUT_DIR / fname, bbox_inches="tight")
    plt.close()
    print(f"Saved {fname}")


if __name__ == "__main__":
    train, test, rul_truth = load_subset("FD001")

    print("=" * 60)
    print("EDA: NASA C-MAPSS FD001 (Turbofan Engine Degradation)")
    print("=" * 60)
    print(f"Train rows: {len(train)} | Engines: {train['unit_nr'].nunique()}")
    print(f"Test rows:  {len(test)} | Engines: {test['unit_nr'].nunique()}")

    near_constant = plot_sensor_variance(train)
    active_sensors = [s for s in SENSOR_NAMES if s not in near_constant]

    # Sensors known from literature to show strong degradation trends
    key_sensors = ["s_2", "s_3", "s_4", "s_7", "s_11", "s_12", "s_15", "s_17", "s_20", "s_21"]
    key_sensors = [s for s in key_sensors if s in active_sensors]

    plot_sensor_trajectories(train, key_sensors[:6])
    plot_correlation_heatmap(train, active_sensors)
    plot_rul_distribution(train)
    plot_settings_sanity_check(train)
    plot_engine_life_lengths(train)

    print("\nAll plots saved to:", OUT_DIR)
