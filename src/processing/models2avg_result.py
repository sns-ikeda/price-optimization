from __future__ import annotations

from typing import Any

import numpy as np


def models2avg_result(models_dict: dict[int, list[Any]], attribute: str) -> dict[int, float]:
    avg_values_dict = dict()
    for num_of_items, models in models_dict.items():
        values = []
        for model in models:
            values.append(getattr(model, attribute))
        avg_values_dict[num_of_items] = np.mean(values)
    return avg_values_dict
