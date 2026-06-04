# Simulation of Forest Fire Spread using AI/ML

A machine learning system that predicts next-day forest fire occurrence
using satellite-observed active fire data from NASA FIRMS (Fire
Information for Resource Management System).

## Overview

This project addresses forest fire prediction as a **pixel-level binary
classification task**: for each cell in a 128×128 spatial grid covering
South/Central Asia (60–100°E, 5–40°N), we predict whether a fire will
be detected on the next satellite pass.

### Pipeline

1. **Data Acquisition** — Downloads active fire detection data from NASA
   FIRMS (SUOMI NPP VIIRS instrument, 7-day rolling export).
2. **Rasterisation** — Converts point-level fire detections (lat/lon) to
   a 128×128 grid of binary fire masks per day.
3. **Feature Engineering** — For each grid cell, extracts 10 features:
   - Normalised spatial position (row, column)
   - Fire status on the previous 1–2 days
   - Neighbourhood fire counts at radii 1, 2, and 3
   - Mean brightness temperature (band I4, 3.74 µm)
   - Mean Fire Radiative Power (FRP)
   - Detection count per cell
4. **Model Comparison** — Evaluates four classifiers using time-series
   cross-validation (no temporal data leakage):
   - Logistic Regression (baseline)
   - Random Forest
   - Gradient Boosted Trees
   - XGBoost
5. **Hyperparameter Tuning** — GridSearchCV on the best-performing model
   with TimeSeriesSplit validation.
6. **Feature Ablation** — Measures the contribution of each feature
   group by removing it and re-evaluating.
7. **Evaluation** — Reports accuracy, precision, recall, F1, AUC-ROC,
   confusion matrix, and generates diagnostic plots.
8. **Dashboard** — A Next.js web application that displays real fire
   detections on an interactive Leaflet map alongside the trained
   model's evaluation metrics.

## Results

### Model Comparison (Time-Series Cross-Validation, 3 folds)

| Model               | AUC-ROC         | F1 Score        | Avg Time |
|---------------------|-----------------|-----------------|----------|
| Logistic Regression | 0.890 ± 0.025   | 0.300 ± 0.072   | 0.8s     |
| Random Forest       | 0.908 ± 0.008   | 0.286 ± 0.022   | 0.6s     |
| Gradient Boosting   | 0.907 ± 0.012   | 0.297 ± 0.068   | 4.6s     |
| **XGBoost** ★       | **0.913 ± 0.010** | **0.315 ± 0.081** | **0.3s** |

### Best Model: XGBoost (after hyperparameter tuning)

| Metric             | Value  |
|--------------------|--------|
| Accuracy           | 96.55% |
| AUC-ROC            | 0.907  |
| Precision          | 0.252  |
| Recall             | 0.358  |
| F1 Score           | 0.296  |
| Decision Threshold | 0.90   |

### Feature Ablation Study

| Feature Group Removed | AUC-ROC  | Δ AUC    | Interpretation           |
|-----------------------|----------|----------|--------------------------|
| (none — full model)   | 0.907    | —        | Baseline                 |
| Spatial               | 0.839    | −0.068   | Most impactful group     |
| Temporal              | 0.903    | −0.005   | Modest contribution      |
| Neighbourhood         | 0.927    | +0.019   | Slight overfitting       |
| Radiometric           | 0.909    | +0.001   | Minimal independent effect|

### Interpretation

The AUC-ROC of 0.907 shows strong discriminative ability — the model
correctly ranks fire-prone cells above non-fire cells ~91% of the time.
The lower F1 (0.30) reflects the precision/recall trade-off inherent
in rare-event prediction: at the chosen threshold (0.90), the model
catches ~36% of actual fires but generates some false alarms. This is
consistent with published baselines on comparable VIIRS data.

**Key design decisions:**
- **Inverse-frequency class weighting** — fire pixels receive ~15×
  higher weight, preventing the trivial "predict no fire" strategy.
- **Threshold tuning** — decision boundary optimised for F1 on the
  training set (0.90 vs. default 0.50).
- **Chronological split** — train on days 1–3, test on days 4–6,
  ensuring no temporal data leakage.
- **TimeSeriesSplit CV** — 3-fold time-series cross-validation for
  model selection, respecting temporal ordering.

## Project Structure

```
├── train_fire_model.py         # Main training pipeline
│                                 (model comparison, tuning, ablation, plots)
├── forestfire_sim.py           # U-Net + fire spread simulator
├── evaluation_metrics.py       # Pixel-level binary metrics
├── run_firms_benchmark.py      # Persistence baseline benchmark
├── fetch_firms_data.py         # NASA FIRMS data downloader
├── run.sh                      # Full pipeline runner
├── requirements.txt            # Python dependencies
├── notebooks/
│   └── 01_eda.ipynb            # Exploratory Data Analysis notebook
├── data/
│   └── suomi_viirs_7d.csv      # FIRMS VIIRS active fire CSV
├── models/
│   └── fire_model.pkl          # Trained model (auto-generated)
├── reports/
│   ├── model_metrics.json      # Final model metrics
│   ├── model_comparison.json   # Cross-validation comparison table
│   ├── predictions.npz         # Prediction arrays
│   ├── roc_curve.png           # ROC curve plot
│   ├── pr_curve.png            # Precision-Recall curve plot
│   ├── confusion_matrix.png    # Confusion matrix heatmap
│   ├── model_comparison.png    # Model comparison bar chart
│   └── feature_importance.png  # Feature importance bar chart
├── dashboard/                  # Next.js visualisation dashboard
│   ├── app/
│   │   ├── page.tsx            # Main page
│   │   ├── layout.tsx          # Root layout
│   │   ├── globals.css         # Vanilla CSS design system
│   │   └── api/                # API routes for data serving
│   └── components/
│       ├── fire-map.tsx        # Leaflet interactive map
│       ├── metrics-panel.tsx   # Model evaluation display
│       └── dashboard-header.tsx
└── output/                     # Raster outputs from simulation
```

## Quickstart

```bash
# Clone and enter the project
git clone https://github.com/ryanfer123/Simulation-of-Forest-Fire-Spread-using-AI-ML.git
cd Simulation-of-Forest-Fire-Spread-using-AI-ML

# Run the full pipeline (creates venv, downloads data, trains, starts dashboard)
bash run.sh
```

Or run each step manually:

```bash
# 1. Set up environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Download FIRMS data
python3 fetch_firms_data.py

# 3. Train and evaluate models (produces all plots and metrics)
python3 train_fire_model.py

# 4. Start the dashboard
cd dashboard && npm install --legacy-peer-deps && npm run dev
```

## Data Sources

- **NASA FIRMS**: Active fire detections from SUOMI NPP VIIRS instrument
  ([FIRMS](https://firms.modaps.eosdis.nasa.gov/))
- **ISRO Bhuvan / Bhoonidhi**: Terrain and land-use data (for extended module)
- **IMD**: Weather observations (for extended module)

## Dependencies

- Python 3.10+, scikit-learn, xgboost, numpy, matplotlib
- Node.js 18+, Next.js 15, Leaflet, Recharts

## License

MIT
