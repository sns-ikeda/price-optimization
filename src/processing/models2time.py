from typing import Any, Dict, List

import numpy as np


def models2avg_cal_time(models_dict: Dict[int, List[Any]]) -> Dict[int, float]:
    cal_time_avg = dict()
    for num_of_items, models in models_dict.items():
        cal_times = []
        for model in models:
            cal_times.append(model.calculation_time)
        cal_time_avg[num_of_items] = np.mean(cal_times)
    return cal_time_avg
