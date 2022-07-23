from __future__ import annotations

from collections import defaultdict

import numpy as np

from src.optimization.result import Result


def transform_artificial_results(
    results: list[int, dict[str, dict[str, Result]]], attribute: str
) -> dict[str, dict[int, list(float)]]:
    """計算結果を整形"""
    transformed_results_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for result_dict in results:
        for num_of_item, model_result_dict in result_dict.items():
            for model, algo_result_dict in model_result_dict.items():
                for algo, result in algo_result_dict.items():
                    transformed_results_dict[model][algo][num_of_item].append(
                        getattr(result, attribute)
                    )
    return transformed_results_dict


def average_results_dict(
    results_dict: dict[str, dict[str, dict[int, list[float]]]], attribute: str
) -> dict[str, dict[str, dict[int, float]]]:
    """計算結果を平均"""
    avg_results_dict = defaultdict(lambda: defaultdict(dict))
    for model, algo_result_dict in results_dict.items():
        for algo, noi_result_dict in algo_result_dict.items():
            for num_of_item, results in noi_result_dict.items():
                values = [getattr(result, attribute) for result in results]
                avg_results_dict[model][algo][num_of_item] = np.mean(values).round(3)
    return avg_results_dict
