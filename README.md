# City Twin AI

City Twin AI is an end-to-end geospatial ML pipeline that downloads terrain/environmental data, builds flood-risk features, trains multiple models, and serves inference with a FastAPI backend and Leaflet frontend.

## Pipeline steps

1. Download raw geospatial datasets into `data/raw/`.
2. Process DEM, slope, aspect, TWI, rainfall, and distance-to-water layers into `data/processed/layers/`.
3. Build feature grid and tabular training dataset.
4. Train RandomForest, XGBoost, and GradientBoosting models.
5. Select best model by ROC-AUC and save in `backend/models/`.
6. Serve predictions through FastAPI endpoints.

## Run

```bash
python scripts/run_pipeline.py
PYTHONPATH=backend python backend/app/ml/train.py
PYTHONPATH=backend uvicorn app.main:app --reload --port 8000
```

Open `frontend/index.html` in a browser.
