# Application Summary: Forest Fire Simulation using AI/ML

This document provides concise, truthful wording for portfolio or program applications such as ML summer schools, internships, and project showcases.

---

## Recommended project title

**AI-Based Forest Fire Prediction and Spread Simulation using U-Net and Cellular Automata**

Alternative shorter title:

**Geospatial Forest Fire Risk Prediction and Spread Simulation with Deep Learning**

---

## Portfolio/work sample link

Use the GitHub repository link for this project. If a demo video, notebook, or hosted app is later added, include that link as well.

Suggested wording:

> GitHub repository: `<paste repository URL>`

---

## Domain

Recommended domain selections:

- Machine Learning
- Deep Learning
- Geospatial AI
- Environmental AI
- Disaster Management
- Computer Vision / Spatial Prediction

If only one broad category is allowed, choose **AI/ML**.

---

## Project type

Suggested wording:

> End-to-end ML prototype for geospatial fire-risk prediction and short-term fire-spread simulation. The system includes synthetic geospatial data generation, preprocessing, a U-Net-style prediction model, cellular automata simulation, Numba optimization, and map/GeoTIFF/GIF outputs.

---

## Dataset used and approximate size

Do **not** mention COCO for this repository. COCO is an object-detection/image-captioning dataset and is not used by the codebase.

Suggested truthful wording:

> Synthetic geospatial forest-fire dataset generated in code. It includes weather variables, terrain/DEM-derived slope and aspect, land-use/land-cover fuel maps, settlement/road layers, and historical fire masks. Test/demo mode uses approximately 333 x 333 raster grids for a 10 km x 10 km region at 30 m resolution. Full synthetic mode can generate approximately 3333 x 3333 raster grids for a 100 km x 100 km region at 30 m resolution.

---

## Key metrics achieved

Current project status:

> Metrics are not finalized yet. The current version validates the end-to-end pipeline on synthetic data; quantitative benchmarking against real fire events is planned.

Recommended form entry:

> Prototype stage: pipeline validated on synthetic data. Planned metrics include F1 Score, IoU/Dice, Precision/Recall, ROC-AUC, and burned-area overlap for spread simulation.

Do not write placeholder metrics such as `Accuracy: 92%` unless those values are actually produced by an evaluation script.

---

## Components built

Select all applicable components if the application form supports multiple selections:

- Data generation / data preprocessing
- Deep learning model
- Prediction system
- Simulation engine
- Geospatial visualization
- Testing workflow
- Export pipeline for PNG, GIF, and GeoTIFF outputs

Do **not** select `Search` unless a real search/retrieval feature is added later.

---

## GitHub repository and/or demo link

Suggested wording:

> GitHub: `<paste repository URL>`
>
> Demo: In progress. The current repository includes runnable scripts and generated output artifacts.

---

## Is the project still in progress?

Recommended answer: **Yes**.

Reason:

> The prototype works with synthetic data, but real-data ingestion, formal metrics, experiment tracking, and a polished demo are still planned.

---

## Short project pitch

> I built an AI/ML prototype for forest fire risk prediction and spread simulation. The system generates synthetic geospatial layers for weather, terrain, land cover, human factors, and historical fires, preprocesses them into spatial tensors, predicts next-day fire probability using a U-Net-style CNN, and simulates fire spread using cellular automata influenced by fuel, wind, slope, temperature, and humidity. It also exports maps, animations, and GeoTIFFs for visualization. The project is currently in progress, with real satellite/weather data integration and formal metrics planned next.

---

## Improvement checklist before submitting as a portfolio highlight

- [ ] Add real fire data or a small curated sample dataset.
- [ ] Add an `evaluate.py` script for F1, IoU, Precision/Recall, and ROC-AUC.
- [ ] Add screenshots or a demo GIF directly to the README.
- [ ] Add dependency versions for reproducibility.
- [ ] Add a one-command demo script or notebook.
- [ ] Add a short architecture diagram.
- [ ] Add a results table once metrics are available.
