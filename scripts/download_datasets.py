from __future__ import annotations

from pathlib import Path

from common import RAW_DIR, download_if_missing, filename_from_url

DATA_SOURCES = {
    "dem": [
        "https://elevation-tiles-prod.s3.amazonaws.com/geotiff/10/163/395.tif",
    ],
    "rainfall": [
        "https://github.com/GeoTIFF/geotiff.io/raw/master/examples/usgs/o41078a5.tif",
    ],
    "water": [
        "https://download.geofabrik.de/europe/monaco-latest-free.shp.zip",
    ],
    "flood_labels": [
        "https://github.com/johan/world.geo.json/raw/master/countries/BGD.geo.json",
    ],
}


def download_all() -> list[Path]:
    downloaded = []
    for category, urls in DATA_SOURCES.items():
        category_dir = RAW_DIR / category
        for url in urls:
            destination = category_dir / filename_from_url(url)
            print(f"Downloading {url} -> {destination}")
            downloaded.append(download_if_missing(url, destination))
    return downloaded


if __name__ == "__main__":
    files = download_all()
    print(f"Downloaded or reused {len(files)} files")
