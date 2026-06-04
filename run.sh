#!/bin/bash
set -e

echo "========================================="
echo "   NASA FIRMS Benchmark & Dashboard"
echo "========================================="

echo "1. Running FIRMS Benchmark on Real Data..."
python3 run_firms_benchmark.py

echo ""
echo "2. Starting Interactive Dashboard..."
cd dashboard
npm install
npm run dev
