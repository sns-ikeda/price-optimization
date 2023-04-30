from __future__ import annotations

import json

import numpy as np
from tqdm import tqdm

from src.configs import ALGO_CONFIG
from src.optimize.algorithms import ALGORITHMS
from src.optimize.optimizer import Optimizer
from src.optimize.params import SyntheticDataParameter
from src.optimize.predictor2model import predictor2model
from src.utils.module_handler import get_object_from_module
from src.utils.paths import OPT_MODEL_DIR, RESULT_DIR

if __name__ == "__main__":
    use_predictor_name = "ORT_LH"
    true_predictor_name = "ORT_LH"
    use_model_name = predictor2model[use_predictor_name]
    true_model_name = predictor2model[true_predictor_name]
    algo_names = ["solver_naive", "milo_relax", "coord_descent"]

    num_iteration = 3
    num_of_items_list = [5]
    num_of_prices = 5
    depth_of_trees = 3
    base_price = 5
    price_min = 0.8
    price_max = 1.0
    tune = False
    calc_time_only = False

    result_output = dict()
    for num_of_items in num_of_items_list:
        num_of_g = 2 * num_of_items * 0
        results = []
        for i in tqdm(range(num_iteration)):
            params = SyntheticDataParameter(
                num_of_items=num_of_items,
                num_of_prices=num_of_prices,
                num_of_other_features=num_of_g,
                depth_of_trees=depth_of_trees,
                base_price=base_price,
                price_min=price_min,
                price_max=price_max,
                base_quantity=300,
                seed=i,
            )
            # アルゴリズムごとに最適価格を計算
            result_dict = dict()
            for algo_name in algo_names:
                model_input = Optimizer.make_model_input(
                    model_name=true_model_name, data_param=params
                )
                index_set, constant = model_input.index_set, model_input.constant
                module_path = OPT_MODEL_DIR / true_model_name / "model.py"
                model_class = get_object_from_module(module_path, "Model")
                model = model_class(index_set=index_set, constant=constant)
                algo_class = ALGORITHMS.get(algo_name, None)
                algorithm = algo_class(model=model, **ALGO_CONFIG[algo_name])
                algorithm.run()
                result_dict[algo_name] = {
                    "obj": algorithm.result.objective,
                    "calculation_time": algorithm.result.calculation_time,
                    "opt_price": algorithm.result.opt_prices,
                }
            results.append(result_dict)

        # 計算結果の集約（平均や標準偏差を計算）
        result_summary = {algo_name: dict() for algo_name in algo_names}
        for algo_name in algo_names:
            result_summary[algo_name]["mean (calculation_time)"] = np.mean(
                [r[algo_name]["calculation_time"] for r in results]
            )
            result_summary[algo_name]["mean (obj)"] = np.mean(
                [r[algo_name]["obj"] / r["solver_naive"]["obj"] for r in results]
            )
            result_summary[algo_name]["std (calculation_time)"] = np.std(
                [r[algo_name]["calculation_time"] for r in results]
            )
            result_summary[algo_name]["std (obj)"] = np.std(
                [r[algo_name]["obj"] / r["solver_naive"]["obj"] for r in results]
            )
            json_name = (
                RESULT_DIR
                / "synthetic"
                / f"result_{num_of_items}_{depth_of_trees}_{algo_name}.json"
            )
            with open(json_name, "w") as fp:
                json.dump(result_summary[algo_name], fp)

            import pandas as pd

            result_df = pd.DataFrame.from_dict(result_summary[algo_name], orient="index").T
            result_df.to_csv(
                RESULT_DIR
                / "synthetic"
                / f"result_p{num_of_items}_d{depth_of_trees}_{algo_name}.csv"
            )
            print(algo_name, result_df)
