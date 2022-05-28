from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def dict2json(target_dict: dict[Any, Any], save_path: Path) -> None:
    """辞書をjson形式で保存"""
    with open(save_path, "w") as f:
        json.dump(target_dict, f, indent=4)
