from __future__ import annotations

import numpy as np

from src.algorithm.result import Result


def average_results(
    results_dict: dict[str, dict[int, list[Result]]], attribute: str
) -> dict[str, dict[int, float]]:
    """各手法ごとの計算結果を平均する"""
    avg_results_dict = dict()
    for method, results_dict_method in results_dict.items():
        avg_results_dict[method] = average_results_method(
            results_dict_method=results_dict_method, attribute=attribute
        )
    return avg_results_dict


def average_results_method(
    results_dict_method: dict[int, list[Result]], attribute: str
) -> dict[int, float]:
    """手法単位の計算結果を平均する"""
    avg_values_dict = dict()
    for num_of_items, results in results_dict_method.items():
        values = []
        for result in results:
            values.append(getattr(result, attribute))
        avg_values_dict[num_of_items] = np.mean(values).round(3)
    return avg_values_dict
