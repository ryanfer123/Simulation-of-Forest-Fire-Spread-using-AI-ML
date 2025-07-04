# Forest Fire Simulation using AI-ML

This repository contains an AI/ML-based forest fire prediction and spread simulation system (`forestfire_sim.py`) and a comprehensive test suite (`test_forestfire_sim.py`) for it.

The main script, `forestfire_sim.py`, performs the core prediction and simulation tasks, while `test_forestfire_sim.py` verifies its functionalities.

---

## 🚀 About `forestfire_sim.py`

The main application script, `forestfire_sim.py`, implements an AI/ML-based forest fire prediction and spread simulation system. It includes:

- Data collection (with synthetic data generation for standalone use).
- Preprocessing of geospatial and temporal data.
- A U-Net model for predicting fire probability for the next day.
- A Cellular Automata model for simulating fire spread, optimized with Numba for performance.
- Visualization of predictions and spread simulations.

The script can be run in a `demo_mode` for quick demonstrations with smaller datasets and simplified model training.

---

## 📁 File Structure

- `forestfire_sim.py` — Main script for AI/ML-based forest fire risk prediction and spread simulation.
- `synthetic_data_generator.py` — Module for generating synthetic data for testing and demonstration.
- `test_forestfire_sim.py` — Test suite for `forestfire_sim.py`.
- `requirements.txt` — Lists Python package dependencies for the project.
- `output/` — Default folder where simulation outputs (images, GeoTIFFs) from `forestfire_sim.py` are saved.
- `test_output/` — Folder where test-specific visualizations and GeoTIFFs are saved by `test_forestfire_sim.py`.

---

## What It Tests

1. **Module Importing** 
   - Ensures `forestfire_sim.py` is importable

2. **Data Collection** 🗂️
    -   Tests generation of synthetic weather, terrain, LULC, human factors, and historical fire data.

3. **Preprocessing** ⚙️⚙️
   - Merges and splits datasets into train/validation/test sets

4. **Model Creation**
   - Builds a lightweight ConvNet (instead of full U-NET) for testing purposes

5. **Fire Spread Simulation** 🔥
   - Tests a reduced fire spread run in `testing_mode=True`

6. **Visualization** 📊
   - Saves dummy prediction outputs, animations, and GeoTIFFs

---

## Running the Tests ⚙️

To run the tests:

```bash
python test_forestfire_sim.py

⚠️ Ensure that forestfire_sim.py exists in the same directory or Python path.

⸻

🗂️ Outputs

After execution, test artifacts are saved to the test_output/ directory:
	•	test_prediction.png — Predicted fire risk map
	•	test_animation.gif — Animated fire spread
	•	test_prediction.tif — GeoTIFF output of prediction

⸻

⚠️ Notes
	•	This test suite uses reduced date ranges and simplified models for fast, non-resource-intensive testing.
	•	Large models and long simulations are skipped or mocked using testing_mode=True.
	•	File outputs >50 MB may trigger GitHub LFS warnings. Consider using Git LFS if needed.

⸻

📌 Requirements

The primary Python packages required for this project are listed in `requirements.txt`.
Key dependencies include:
	•	`numpy` for numerical operations.
	•	`tensorflow` for the U-Net prediction model.
	•	`matplotlib` for visualizations.
	•	`rasterio` for GeoTIFF handling.
	•	`scikit-learn` for data splitting.
	•	`numba` for accelerating the fire spread simulation.
	•	`scipy` for utilities in data generation and simulation.

It is recommended to install all dependencies using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```


⸻

📧 Contact

Maintained by Ryan Fernandes.
For queries, suggestions, or contributions, feel free to open an issue or pull request.

⸻


---
