from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.utils.paths import ALGO_DIR


def read_config(filepath: Path | str) -> dict[str, Any]:
    with open(filepath) as file:
        config = yaml.safe_load(file.read())
    return config


@dataclass(frozen=True)
class RealworldConfig:
    dataset_name: str
    predictor_name: str
    algo_name: str
    num_of_prices: int
    train_size: float


@dataclass(frozen=True)
class SyntheticConfig:
    num_iteration: int
    num_of_items: list[int]
    num_of_prices: int
    num_of_other_features: int
    depth_of_trees: int
    base_price: int
    predictor_names: list[str]
    algo_names: list[str]


# アルゴリズムの設定
ALGO_CONFIG = read_config(ALGO_DIR / "algo_config.yaml")
