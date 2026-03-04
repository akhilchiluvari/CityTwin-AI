from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import rasterio

from app.config import (
    FEATURE_GRID_FILE,
    MODEL_FILE,
    PROCESSED_DIR,
    RISK_MAP_FILE,
)
from app.core.geospatial import sample_raster

FEATURE_LAYER_DIR = PROCESSED_DIR / "layers"
FEATURE_ORDER = [
    "elevation",
    "slope",
    "aspect",
    "twi",
    "distance_to_water",
    "rainfall",
]


def _load_model():
    if not MODEL_FILE.exists():
        raise FileNotFoundError("Model not found. Train the model first.")
    return joblib.load(MODEL_FILE)


def _feature_paths() -> dict[str, Path]:
    return {name: FEATURE_LAYER_DIR / f"{name}.tif" for name in FEATURE_ORDER}


def predict_point(lat: float, lon: float) -> dict:
    model = _load_model()
    features = {}
    for name, path in _feature_paths().items():
        features[name] = sample_raster(path, lat, lon)

    x = np.array([[features[name] for name in FEATURE_ORDER]], dtype=np.float32)
    risk_probability = float(model.predict_proba(x)[0, 1])

    return {
        "lat": lat,
        "lon": lon,
        "risk_probability": risk_probability,
        **features,
    }


def predict_risk_map() -> Path:
    model = _load_model()
    feature_arrays = []
    profile = None

    for name, path in _feature_paths().items():
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float32)
            feature_arrays.append(arr)
            if profile is None:
                profile = src.profile

    stacked = np.stack(feature_arrays, axis=0)
    rows, cols = stacked.shape[1], stacked.shape[2]
    x = stacked.reshape(stacked.shape[0], -1).T

    probs = model.predict_proba(x)[:, 1].reshape(rows, cols).astype(np.float32)

    out_profile = profile.copy()
    out_profile.update(dtype="float32", count=1)
    RISK_MAP_FILE.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(RISK_MAP_FILE, "w", **out_profile) as dst:
        dst.write(probs, 1)

    return RISK_MAP_FILE
