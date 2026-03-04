from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import requests

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
LAYERS_DIR = PROCESSED_DIR / "layers"

for path in [RAW_DIR, PROCESSED_DIR, LAYERS_DIR]:
    path.mkdir(parents=True, exist_ok=True)


def download_if_missing(url: str, destination: Path, timeout: int = 60) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return destination

    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    destination.write_bytes(response.content)
    return destination


def filename_from_url(url: str) -> str:
    return Path(urlparse(url).path).name
