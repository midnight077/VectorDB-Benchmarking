from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class DatasetConfig(BaseModel):
    name: str
    path: str
    # Metric override; falls back to the HDF5 file's `distance` attr if unset.
    metric: str | None = None


class BackendConfig(BaseModel):
    name: str
    index_type: str = "flat"
    build_params: dict = Field(default_factory=dict)
    search_params: dict = Field(default_factory=dict)


class ResultsConfig(BaseModel):
    backend: str = "sqlite"
    path: str = "results/results.db"


class Config(BaseModel):
    lane: str = "A"
    dataset: DatasetConfig
    k_values: list[int] = Field(default_factory=lambda: [10])
    n_warmup: int = 50
    n_repeats: int = 3
    backends: list[BackendConfig]
    results: ResultsConfig = Field(default_factory=ResultsConfig)


def load_config(path: str | Path) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return Config(**raw)
