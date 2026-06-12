"""Download an ann-benchmarks HDF5 dataset into data/.

Usage:
    python scripts/download_dataset.py [name]

Defaults to the dev/smoke dataset: fashion-mnist-784-euclidean.
"""

from __future__ import annotations

import sys
from pathlib import Path

import requests
from tqdm import tqdm

BASE_URL = "http://ann-benchmarks.com"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def download(name: str) -> Path:
    url = f"{BASE_URL}/{name}.hdf5"
    dest = DATA_DIR / f"{name}.hdf5"
    if dest.exists():
        print(f"Already present: {dest}")
        return dest

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {dest}")
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    return dest


if __name__ == "__main__":
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else "fashion-mnist-784-euclidean"
    download(dataset_name)
