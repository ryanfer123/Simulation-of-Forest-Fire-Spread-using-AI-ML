#!/bin/bash
# Forest Fire Prediction Pipeline
# Runs the full pipeline: data download -> model training -> dashboard
set -e

echo "========================================"
echo " Forest Fire Prediction Pipeline"
echo "========================================"

# Step 1: Create virtual environment if needed
if [ ! -d ".venv" ]; then
    echo "[1/4] Creating Python virtual environment..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
else
    source .venv/bin/activate
fi

# Step 2: Download FIRMS data if not present
if [ ! -f "data/suomi_viirs_7d.csv" ]; then
    echo "[2/4] Downloading NASA FIRMS data..."
    python3 fetch_firms_data.py
else
    echo "[2/4] FIRMS data already present, skipping download."
fi

# Step 3: Train the model
echo "[3/4] Training fire prediction model..."
python3 train_fire_model.py

# Step 4: Start the dashboard
echo "[4/4] Starting dashboard..."
cd dashboard
if [ ! -d "node_modules" ]; then
    npm install --legacy-peer-deps
fi
npm run dev
