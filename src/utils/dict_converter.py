from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def dict2json(target_dict: dict[Any, Any], save_path: Path) -> None:
    """Save dict as json"""
    with open(save_path, "w") as f:
        json.dump(target_dict, f, indent=4)


def dict2yaml(target_dict: dict[Any, Any], save_path: Path) -> None:
    """Save dict as yaml"""
    with open(save_path, "w") as f:
        yaml.dump(target_dict, f)


def dict2df(
    dict_: dict[Any, Any], value_name: str, index_name: str = "num_of_items"
) -> pd.DataFrame:
    """Convert dict to DataFrame"""
    df = (
        pd.DataFrame.from_dict(dict_, orient="index")
        .reset_index()
        .rename(columns={"index": index_name, 0: value_name})
    )
    return df.round(3)
