from __future__ import annotations

from pathlib import Path

import rasterio
from rasterio.merge import merge

from common import LAYERS_DIR, RAW_DIR


def process_dem() -> Path:
    dem_dir = RAW_DIR / "dem"
    dem_tiles = sorted(dem_dir.glob("*.tif"))
    if not dem_tiles:
        raise FileNotFoundError("No DEM tiles found in data/raw/dem")

    src_files = [rasterio.open(tile) for tile in dem_tiles]
    mosaic, transform = merge(src_files)
    profile = src_files[0].profile.copy()
    profile.update(
        transform=transform,
        width=mosaic.shape[2],
        height=mosaic.shape[1],
        count=1,
        dtype=mosaic.dtype,
    )

    out_path = LAYERS_DIR / "elevation.tif"
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mosaic[0], 1)

    for src in src_files:
        src.close()

    print(f"Saved merged DEM to {out_path}")
    return out_path


if __name__ == "__main__":
    process_dem()
