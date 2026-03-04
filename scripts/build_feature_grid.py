from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio

from common import LAYERS_DIR, PROCESSED_DIR

FEATURE_ORDER = [
    "elevation",
    "slope",
    "aspect",
    "twi",
    "distance_to_water",
    "rainfall",
]


def build_feature_grid() -> Path:
    arrays = []
    profile = None

    for name in FEATURE_ORDER:
        path = LAYERS_DIR / f"{name}.tif"
        with rasterio.open(path) as src:
            arrays.append(src.read(1).astype(np.float32))
            if profile is None:
                profile = src.profile.copy()

    stack = np.stack(arrays, axis=0)
    out_path = PROCESSED_DIR / "feature_grid.tif"
    profile.update(count=len(FEATURE_ORDER), dtype="float32")

    with rasterio.open(out_path, "w", **profile) as dst:
        for idx in range(len(FEATURE_ORDER)):
            dst.write(stack[idx], idx + 1)
            dst.set_band_description(idx + 1, FEATURE_ORDER[idx])

    print(f"Saved feature grid -> {out_path}")
    return out_path


if __name__ == "__main__":
    build_feature_grid()
