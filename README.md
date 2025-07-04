# Forest Fire Simulation using ai-ml

This repository contains a comprehensive testing script for the `forestfire_sim.py` module, which performs AI/ML-based forest fire risk prediction and fire spread simulation.

The test suite (`test_forestfire_sim.py`) verifies the core functionalities of the simulation pipeline, including data collection, preprocessing, modeling, simulation, and visualization.

---

## 📁 File Structure

- `test_forestfire_sim.py` — Main testing script for `forestfire_sim.py`
- `forestfire_sim.py` — (Expected) Implementation of the forest fire simulation and ML model
- `test_output/` — Folder where test visualizations and GeoTIFFs are saved

---

## ✅ What It Tests

1. **Module Importing**
   - Ensures `forestfire_sim.py` is importable

2. **Data Collection**
   - Weather, terrain, LULC, human factors, and historical fire data

3. **Preprocessing**
   - Merges and splits datasets into train/validation/test sets

4. **Model Creation**
   - Builds a lightweight ConvNet (instead of full U-NET) for testing purposes

5. **Fire Spread Simulation**
   - Tests a reduced fire spread run in `testing_mode=True`

6. **Visualization**
   - Saves dummy prediction outputs, animations, and GeoTIFFs

---

## 🧪 Running the Tests

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

Ensure the following Python packages are installed:
	•	numpy
	•	tensorflow
	•	matplotlib (for visualizations)
	•	rasterio (for GeoTIFF saving)
	•	any other dependencies used inside forestfire_sim.py

Install with:

pip install -r requirements.txt


⸻

📧 Contact

Maintained by Ryan Fernandes
For queries, suggestions, or contributions, feel free to open an issue or pull request.

⸻


---
