from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize

from common import LAYERS_DIR, PROCESSED_DIR, RAW_DIR

FEATURE_ORDER = [
    "elevation",
    "slope",
    "aspect",
    "twi",
    "distance_to_water",
    "rainfall",
]


def _load_features() -> tuple[np.ndarray, dict]:
    feature_arrays = []
    profile = None
    for name in FEATURE_ORDER:
        with rasterio.open(LAYERS_DIR / f"{name}.tif") as src:
            feature_arrays.append(src.read(1).astype(np.float32))
            if profile is None:
                profile = src.profile.copy()
    return np.stack(feature_arrays, axis=0), profile


def _build_labels(profile: dict) -> np.ndarray:
    flood_geojsons = list((RAW_DIR / "flood_labels").glob("*.geojson")) + list(
        (RAW_DIR / "flood_labels").glob("*.json")
    )
    if not flood_geojsons:
        raise FileNotFoundError("No flood label GeoJSON files found in data/raw/flood_labels")

    gdf = gpd.read_file(flood_geojsons[0])
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    gdf = gdf.to_crs(profile["crs"])

    labels = rasterize(
        [(geom, 1) for geom in gdf.geometry if geom is not None],
        out_shape=(profile["height"], profile["width"]),
        transform=profile["transform"],
        fill=0,
        dtype="uint8",
    )
    return labels


def build_training_dataset() -> Path:
    features, profile = _load_features()
    labels = _build_labels(profile)

    x = features.reshape(features.shape[0], -1).T
    y = labels.flatten()

    data = pd.DataFrame(x, columns=FEATURE_ORDER)
    data["label"] = y
    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    out_path = PROCESSED_DIR / "training_dataset.csv"
    data.to_csv(out_path, index=False)
    print(f"Saved training dataset -> {out_path} ({len(data)} rows)")
    return out_path


if __name__ == "__main__":
    build_training_dataset()
