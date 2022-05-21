from __future__ import annotations

from typing import Any

import numpy as np


def models2avg_cal_time(models_dict: dict[int, list[Any]]) -> dict[int, float]:
    cal_time_avg = dict()
    for num_of_items, models in models_dict.items():
        cal_times = []
        for model in models:
            cal_times.append(model.calculation_time)
        cal_time_avg[num_of_items] = np.mean(cal_times)
    return cal_time_avg
