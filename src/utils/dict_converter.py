from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def dict2json(target_dict: dict[Any, Any], save_path: Path) -> None:
    """辞書をjson形式で保存"""
    with open(save_path, "w") as f:
        json.dump(target_dict, f, indent=4)


def dict2df(
    dict_: dict[Any, Any], value_name: str, index_name: str = "num_of_items"
) -> pd.DataFrame:
    """辞書をデータフレームに変換"""
    df = (
        pd.DataFrame.from_dict(dict_, orient="index")
        .reset_index()
        .rename(columns={"index": index_name, 0: value_name})
    )
    return df.round(3)
