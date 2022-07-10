from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.utils.paths import ALGO_DIR, DATA_DIR, OPT_DIR


def read_config(filepath: Path | str) -> dict[str, Any]:
    with open(filepath) as file:
        config = yaml.safe_load(file.read())
    return config


# シミュレーションの設定を取得
CONFIG_SIM = read_config("config_simulation.yaml")

# データの設定を取得
CONFIG_DATA = read_config(DATA_DIR / "config_data.yaml")

# 最適化の設定を取得
CONFIG_OPT = read_config(OPT_DIR / "config_optimization.yaml")

# アルゴリズムの設定を取得
CONFIG_ALG = read_config(ALGO_DIR / "config_algorithm.yaml")
