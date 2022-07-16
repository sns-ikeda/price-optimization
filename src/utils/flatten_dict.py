from __future__ import annotations

from typing import Any


def flatten_dict(target_dict: dict[str, dict[str, Any]]) -> dict[str, Any]:
    flattened_dict = dict()
    for k1 in target_dict.keys():
        for k2 in target_dict[k1].keys():
            k = k1 + ": " + k2
            flattened_dict[k] = target_dict[k1][k2]
    return flattened_dict
