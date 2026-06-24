# AI-Powered Predictive Maintenance System — Turbofan Engines (Phase 1)

A predictive maintenance pipeline for industrial turbofan engines, built on
NASA's C-MAPSS Turbofan Engine Degradation Simulation dataset (the standard
academic/industry benchmark for Remaining Useful Life prediction).

This is **Phase 1** of a larger system. This phase delivers core RUL
prediction + explainable root-cause analysis. See "Roadmap" below for what's
next (failure probability, maintenance scheduling, anomaly detection, LSTM,
vibration analysis, and the interactive dashboard).

## Problem

Aircraft/industrial turbofan engines degrade gradually until failure.
Given streaming sensor data (temperatures, pressures, rotational speeds),
predict:
- **Remaining Useful Life (RUL)** — how many operating cycles are left
- **Root cause** — which sensors/subsystems are driving the degradation

## Dataset

NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation),
sourced from NASA's Prognostics Data Repository. Four subsets of
increasing difficulty (we use FD001 as the flagship; the loader supports
all four):

| Subset | Operating Conditions | Fault Modes | Train Engines | Test Engines |
|--------|----------------------|-------------|----------------|---------------|
| FD001  | 1                     | 1 (HPC Degradation)              | 100 | 100 |
| FD002  | 6                     | 1 (HPC Degradation)              | 260 | 259 |
| FD003  | 1                     | 2 (HPC + Fan Degradation)        | 100 | 100 |
| FD004  | 6                     | 2 (HPC + Fan Degradation)        | 248 | 249 |

Each row = one engine, one operating cycle, 3 operational settings + 21
sensor readings. Training data runs every engine to failure; test data is
truncated, and the true RUL at truncation is provided separately for
scoring.

## Project Structure

```
pdm-turbofan/
├── data/raw/              # Raw NASA C-MAPSS text files (all 4 subsets)
├── src/
│   ├── data_loader.py     # Load + label RUL for any subset
│   ├── features.py        # Rolling-window feature engineering, engine-level split
│   ├── eda.py              # Exploratory analysis -> outputs/plots/
│   ├── train_baseline.py  # Random Forest + XGBoost RUL regressors
│   └── explainability.py  # SHAP global + per-engine root cause analysis
├── models/                # Saved trained models (.pkl)
├── outputs/plots/         # All generated charts
└── requirements.txt
```

## How to run (copy-paste, in order)

```bash
pip install -r requirements.txt

python3 src/data_loader.py       # sanity check the data loads correctly
python3 src/eda.py               # generates all EDA plots
python3 src/train_baseline.py    # trains RF + XGBoost, prints metrics, saves models
python3 src/explainability.py    # SHAP root-cause analysis + plots
```

## Key Design Decisions (important for your viva/interview)

1. **Piecewise RUL capping (at 125 cycles).** A healthy engine's exact RUL
   (300 vs 320 cycles left) isn't learnable from early-life sensor data —
   nothing has degraded yet. Capping turns the "healthy plateau" into a
   flat, learnable label. This is standard practice in C-MAPSS literature
   (Heimes 2008 and most follow-on work).

2. **Engine-level train/validation split, not row-level.** Splitting
   individual rows randomly leaks information — rows from the same engine's
   trajectory would appear in both train and validation. We split whole
   engines, so validation genuinely tests generalization to unseen engines.

3. **Rolling-window features over raw readings.** A single sensor reading
   is noisy; the rolling mean/std/slope over 5 cycles captures the actual
   degradation trend.

4. **PHM08 asymmetric scoring**, not just RMSE. The official competition
   metric penalizes *late* predictions (predicting more life than is left)
   far more harshly than early ones — because in real maintenance, a missed
   failure is catastrophic while early maintenance just wastes some engine
   life. We report both RMSE and PHM08 score.

## Results (FD001, official test set)

| Model | RMSE (cycles) | MAE (cycles) |
|-------|---------------|--------------|
| Random Forest | ~17.9 | ~12.4 |
| XGBoost | ~17.4 | ~12.4 |

(LSTM, added in Phase 2, typically pushes this down to ~13-15 RMSE by
learning temporal patterns directly instead of relying on hand-crafted
rolling features.)

## SHAP Root-Cause Validation

The top SHAP features globally (`s_4` = LPT outlet temp, `s_11` = HPC
outlet static pressure, `s_9` = core speed) align with NASA's documented
fault mode for FD001: **HPC (High Pressure Compressor) degradation** — the
model recovered the real physical failure mechanism without being told
what it was. This is strong evidence the model is learning genuine
degradation physics, not spurious correlations.

## Roadmap (next phases)

- [ ] **Phase 2:** Failure probability (convert RUL + uncertainty into
      P(failure within N cycles)) + rule-based maintenance scheduler
- [ ] **Phase 3:** Isolation Forest anomaly detection (flag abnormal engines
      without needing failure labels — works for early-life anomalies too)
- [ ] **Phase 4:** LSTM sequence model for RUL (deep learning baseline)
- [ ] **Phase 5:** Bonus — vibration + temperature analysis module
      (separate bearing/rotating-machinery use case, demonstrates framework
      generalizes beyond turbofans)
- [ ] **Phase 6:** Streamlit dashboard tying everything together for demos
