from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import rowcol
from rasterio.warp import reproject


def read_raster(path: Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        return src.read(1), src.profile


def write_raster(path: Path, array: np.ndarray, profile: dict) -> None:
    profile = profile.copy()
    profile.update(dtype=array.dtype, count=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array, 1)


def align_to_reference(reference_path: Path, raster_paths: Iterable[Path]) -> Dict[Path, np.ndarray]:
    aligned: Dict[Path, np.ndarray] = {}
    with rasterio.open(reference_path) as ref:
        ref_array = ref.read(1)
        aligned[reference_path] = ref_array
        for path in raster_paths:
            if path == reference_path:
                continue
            with rasterio.open(path) as src:
                destination = np.zeros((ref.height, ref.width), dtype=np.float32)
                reproject(
                    source=rasterio.band(src, 1),
                    destination=destination,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=ref.transform,
                    dst_crs=ref.crs,
                    resampling=Resampling.bilinear,
                )
                aligned[path] = destination
    return aligned


def sample_raster(path: Path, lat: float, lon: float) -> float:
    with rasterio.open(path) as src:
        row, col = rowcol(src.transform, lon, lat)
        value = src.read(1)[row, col]
        return float(value)


def stack_features(feature_arrays: Iterable[np.ndarray]) -> np.ndarray:
    arrays = [arr.astype(np.float32) for arr in feature_arrays]
    flat = [arr.flatten() for arr in arrays]
    return np.vstack(flat).T
