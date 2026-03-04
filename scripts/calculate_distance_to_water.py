from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from sklearn.neighbors import KDTree

from common import LAYERS_DIR, RAW_DIR


def calculate_distance_to_water(
    water_path: Path | None = None,
    dem_path: Path | None = None,
) -> Path:
    dem_path = dem_path or (LAYERS_DIR / "elevation.tif")

    if water_path is None:
        candidates = list((RAW_DIR / "water").glob("*.shp"))
        if not candidates:
            raise FileNotFoundError("No water shapefile found in data/raw/water")
        water_path = candidates[0]

    with rasterio.open(dem_path) as dem_src:
        profile = dem_src.profile.copy()
        transform = dem_src.transform
        shape = (dem_src.height, dem_src.width)
        raster_crs = dem_src.crs

    gdf = gpd.read_file(water_path).to_crs(raster_crs)
    water_mask = rasterize(
        [(geom, 1) for geom in gdf.geometry if geom is not None],
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype="uint8",
    )

    water_pixels = np.argwhere(water_mask == 1)
    all_pixels = np.argwhere(np.ones(shape, dtype=bool))

    if water_pixels.size == 0:
        distances = np.full(shape, np.nan, dtype=np.float32)
    else:
        tree = KDTree(water_pixels)
        dists, _ = tree.query(all_pixels, k=1)
        distances = dists.reshape(shape).astype(np.float32)

    out_path = LAYERS_DIR / "distance_to_water.tif"
    profile.update(dtype="float32", count=1)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(distances, 1)

    print(f"Saved distance-to-water raster -> {out_path}")
    return out_path


if __name__ == "__main__":
    calculate_distance_to_water()
