import argparse
import csv
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
from evaluation_metrics import sweep_thresholds, best_threshold, burned_area_metrics, write_metrics_json

def simple_blur(grid):
    """Simple 3x3 average blur for 2D numpy arrays."""
    blurred = np.copy(grid)
    for i in range(1, grid.shape[0]-1):
        for j in range(1, grid.shape[1]-1):
            blurred[i, j] = np.mean(grid[i-1:i+2, j-1:j+2])
    return blurred

def rasterize_firms_data(csv_path, bounds, grid_size=128):
    """Parse FIRMS CSV and create daily binary fire grids."""
    min_lon, min_lat, max_lon, max_lat = bounds
    
    daily_fires = {}
    total_points = 0
    in_bounds = 0
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_points += 1
            lat = float(row['latitude'])
            lon = float(row['longitude'])
            
            # Simple bounding box filter
            if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
                in_bounds += 1
                acq_date = row['acq_date']
                
                if acq_date not in daily_fires:
                    daily_fires[acq_date] = np.zeros((grid_size, grid_size), dtype=np.uint8)
                
                # Convert lat/lon to grid indices
                row_idx = int((max_lat - lat) / (max_lat - min_lat) * (grid_size - 1))
                col_idx = int((lon - min_lon) / (max_lon - min_lon) * (grid_size - 1))
                
                # Ensure within bounds
                row_idx = max(0, min(grid_size - 1, row_idx))
                col_idx = max(0, min(grid_size - 1, col_idx))
                
                daily_fires[acq_date][row_idx, col_idx] = 1

    print(f"Parsed {total_points} total rows, {in_bounds} points inside bounding box.")
    return daily_fires

def run_baseline_benchmark(daily_fires):
    """
    Evaluate a spatial persistence baseline:
    Predicted fire probability today is a blurred version of yesterday's actual fires.
    """
    sorted_dates = sorted(daily_fires.keys())
    if len(sorted_dates) < 2:
        print("Not enough consecutive days for evaluation.")
        return None
        
    all_y_true = []
    all_y_pred = []
    
    for i in range(1, len(sorted_dates)):
        prev_date = sorted_dates[i-1]
        curr_date = sorted_dates[i]
        
        # Check if they are consecutive days
        d1 = datetime.strptime(prev_date, "%Y-%m-%d")
        d2 = datetime.strptime(curr_date, "%Y-%m-%d")
        if (d2 - d1).days > 1:
            continue
            
        # Ground truth for current day
        y_true = daily_fires[curr_date]
        
        # Prediction: Blur yesterday's mask to create a probability/risk spread map
        prev_mask = daily_fires[prev_date].astype(float)
        y_pred = simple_blur(prev_mask)
        # Normalize prediction
        if y_pred.max() > 0:
            y_pred = y_pred / y_pred.max()
            
        all_y_true.append(y_true)
        all_y_pred.append(y_pred)
        
    if not all_y_true:
        print("No valid consecutive day pairs found.")
        return None
        
    y_true_combined = np.stack(all_y_true)
    y_pred_combined = np.stack(all_y_pred)
    
    # Split into validation and test (e.g., 50/50 split chronologically)
    split_idx = max(1, len(y_true_combined) // 2)
    
    y_true_val = y_true_combined[:split_idx]
    y_pred_val = y_pred_combined[:split_idx]
    
    y_true_test = y_true_combined[split_idx:]
    y_pred_test = y_pred_combined[split_idx:]
    
    print(f"Evaluating {len(y_true_val)} days for validation, {len(y_true_test)} days for testing.")
    
    # Tune threshold on validation
    best_thresh, val_metrics = best_threshold(y_true_val, y_pred_val, selection_metric="f1_score")
    
    # Evaluate on test
    test_metrics = burned_area_metrics(y_true_test, y_pred_test, threshold=best_thresh)
    
    return {
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "selected_threshold": best_thresh
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NASA FIRMS Real Dataset Benchmark")
    parser.add_argument("--csv", default="data/suomi_viirs_7d.csv", help="Path to FIRMS CSV")
    parser.add_argument("--bounds", default="77.0,28.5,81.0,31.5", help="min_lon,min_lat,max_lon,max_lat (Uttarakhand default)")
    parser.add_argument("--grid-size", type=int, default=128, help="Grid size for rasterization")
    args = parser.parse_args()
    
    bounds = [float(x) for x in args.bounds.split(',')]
    
    print("Rasterizing FIRMS data...")
    daily_fires = rasterize_firms_data(args.csv, bounds, args.grid_size)
    
    print("Running baseline evaluation...")
    results = run_baseline_benchmark(daily_fires)
    
    if results:
        Path("reports").mkdir(exist_ok=True)
        
        # Write JSON
        output_json = {
            "benchmark_name": "firms_real_data_baseline",
            "bounds": bounds,
            "grid_size": args.grid_size,
            "selected_threshold": results["selected_threshold"],
            "validation_metrics": results["val_metrics"],
            "test_metrics": results["test_metrics"]
        }
        write_metrics_json(output_json, "reports/firms_real_benchmark_metrics.json")
        print("Wrote JSON to reports/firms_real_benchmark_metrics.json")
        
        # Write CSV
        test = results["test_metrics"]
        csv_lines = [
            "split,threshold,accuracy,precision,recall,f1_score,iou,dice,balanced_accuracy",
            f"test,{results['selected_threshold']},{test['accuracy']},{test['precision']},{test['recall']},{test['f1_score']},{test['iou']},{test['dice']},{test['balanced_accuracy']}"
        ]
        with open("reports/firms_real_benchmark_metrics.csv", "w") as f:
            f.write("\n".join(csv_lines))
        print("Wrote CSV to reports/firms_real_benchmark_metrics.csv")
        
        print("\n--- TEST METRICS ---")
        for k, v in test.items():
            print(f"{k}: {v}")
