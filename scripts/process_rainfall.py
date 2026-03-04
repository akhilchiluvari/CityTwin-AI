from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

from common import LAYERS_DIR, RAW_DIR


def process_rainfall(
    rainfall_path: Path | None = None,
    dem_path: Path | None = None,
) -> Path:
    dem_path = dem_path or (LAYERS_DIR / "elevation.tif")
    if rainfall_path is None:
        candidates = list((RAW_DIR / "rainfall").glob("*.tif"))
        if not candidates:
            raise FileNotFoundError("No rainfall raster found in data/raw/rainfall")
        rainfall_path = candidates[0]

    with rasterio.open(dem_path) as dem_src, rasterio.open(rainfall_path) as rain_src:
        destination = np.zeros((dem_src.height, dem_src.width), dtype=np.float32)
        reproject(
            source=rasterio.band(rain_src, 1),
            destination=destination,
            src_transform=rain_src.transform,
            src_crs=rain_src.crs,
            dst_transform=dem_src.transform,
            dst_crs=dem_src.crs,
            resampling=Resampling.bilinear,
        )
        profile = dem_src.profile.copy()

    out_path = LAYERS_DIR / "rainfall.tif"
    profile.update(dtype="float32", count=1)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(destination.astype(np.float32), 1)

    print(f"Saved rainfall raster -> {out_path}")
    return out_path


if __name__ == "__main__":
    process_rainfall()
