from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio

from common import LAYERS_DIR


def calculate_twi(
    dem_path: Path | None = None,
    slope_path: Path | None = None,
) -> Path:
    dem_path = dem_path or (LAYERS_DIR / "elevation.tif")
    slope_path = slope_path or (LAYERS_DIR / "slope.tif")

    with rasterio.open(dem_path) as src_dem:
        dem = src_dem.read(1).astype(np.float32)
        profile = src_dem.profile.copy()
        x_res, y_res = src_dem.res

    with rasterio.open(slope_path) as src_slope:
        slope = src_slope.read(1).astype(np.float32)

    contributing_area = np.maximum(np.gradient(dem)[0] ** 2 + np.gradient(dem)[1] ** 2, 1e-6)
    contributing_area = np.sqrt(contributing_area) * (x_res * y_res)
    slope_radians = np.radians(np.clip(slope, 0.001, None))
    twi = np.log((contributing_area + 1e-6) / np.tan(slope_radians)).astype(np.float32)

    twi_path = LAYERS_DIR / "twi.tif"
    profile.update(dtype="float32", count=1)
    with rasterio.open(twi_path, "w", **profile) as dst:
        dst.write(twi, 1)

    print(f"Saved TWI -> {twi_path}")
    return twi_path


if __name__ == "__main__":
    calculate_twi()
