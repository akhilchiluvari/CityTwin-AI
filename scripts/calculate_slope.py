from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio

from common import LAYERS_DIR


def calculate_slope_aspect(dem_path: Path | None = None) -> tuple[Path, Path]:
    dem_path = dem_path or (LAYERS_DIR / "elevation.tif")
    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        x_res, y_res = src.res

    dz_dy, dz_dx = np.gradient(dem, y_res, x_res)
    slope = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy))).astype(np.float32)
    aspect = np.degrees(np.arctan2(-dz_dx, dz_dy))
    aspect = np.where(aspect < 0, 360 + aspect, aspect).astype(np.float32)

    profile.update(dtype="float32", count=1)
    slope_path = LAYERS_DIR / "slope.tif"
    aspect_path = LAYERS_DIR / "aspect.tif"

    with rasterio.open(slope_path, "w", **profile) as dst:
        dst.write(slope, 1)
    with rasterio.open(aspect_path, "w", **profile) as dst:
        dst.write(aspect, 1)

    print(f"Saved slope -> {slope_path}")
    print(f"Saved aspect -> {aspect_path}")
    return slope_path, aspect_path


if __name__ == "__main__":
    calculate_slope_aspect()
