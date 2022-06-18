from __future__ import annotations

import yaml
from tqdm import tqdm

from src.algorithm.assign import AlgorithmAssigner
from src.algorithm.result import Result
from src.models.po_L.params import Parameter
from src.processing.average_results import average_results
from src.processing.dict2json import dict2json
from src.utils.paths import RESULT_DIR
from src.utils.plot import save_image


def main():
    # configファイルからシミュレーションの設定を取得
    with open("config.yaml") as file:
        config = yaml.safe_load(file.read())

    # 最適化計算のパラメータ
    num_of_items: list[int] = config["params"]["num_of_items"]
    num_of_prices: int = config["params"]["num_of_prices"]
    num_of_other_features: int = config["params"]["num_of_other_features"]
    depth_of_trees: int = config["params"]["depth_of_trees"]
    base_price: int = config["params"]["base_price"]

    # シミュレーションの設定
    solver: str = config["option"]["solver"]
    TimeLimit: int = int(config["option"]["TimeLimit"])
    methods: list[str] = [method for method, tf in config["algorithm"].items() if tf]
    num_of_simulations: int = config["option"]["num_of_simulations"]
    if "solver_heuristic" in methods:
        NoRelHeurTime: float = config["solver_heuristic"]["NoRelHeurTime"]
        MIPFocus: int = config["solver_heuristic"]["MIPFocus"]
    else:
        NoRelHeurTime: float = 0
        MIPFocus: int = 0
    if "multi_start_local_search" in methods:
        num_multi_start = config["multi_start_local_search"]["num_multi_start"]
    else:
        num_multi_start = 0

    # シミュレーションを実行
    results_dict: dict[str, dict[int, list[Result]]] = dict()
    for method in methods:
        results_method_dict = dict()
        for i in tqdm(num_of_items):
            results: list[Result] = []
            for s in range(num_of_simulations):
                # モデルで使用するパラメータ
                params = Parameter(
                    num_of_items=i,
                    num_of_prices=num_of_prices,
                    num_of_other_features=num_of_other_features,
                    depth_of_trees=depth_of_trees,
                    base_price=base_price,
                    num_of_simulations=num_of_simulations,
                    solver=solver,
                    TimeLimit=TimeLimit,
                    NoRelHeurTime=NoRelHeurTime,
                    MIPFocus=MIPFocus,
                    num_multi_start=num_multi_start,
                    base_seed=s + 42,
                )
                # アルゴリズムを実行
                algorithm = AlgorithmAssigner(params=params, method=method)
                algorithm.run()
                results.append(algorithm.result)
            results_method_dict[i] = results
        results_dict[method] = results_method_dict

    # 結果の後処理
    result_prefixes = ["calculation_time", "objective"]
    for result_prefix in result_prefixes:
        avg_results_dict: dict[str, dict[int, float]] = average_results(
            results_dict=results_dict, attribute=result_prefix
        )
        save_name = (
            f"{result_prefix}_K{params.num_of_prices}_"
            + f"D{params.num_of_other_features}_DoT{params.depth_of_trees}"
        )
        # jsonで保存
        dict2json(
            target_dict=avg_results_dict, save_path=RESULT_DIR / "json" / (save_name + ".json")
        )
        # png形式で保存
        if result_prefix == "calculation_time":
            y_label = "calculation time [sec]"
        else:
            y_label = result_prefix
        save_image(
            avg_results_dict=avg_results_dict,
            dir_path=RESULT_DIR / "png",
            image_name=save_name,
            y_label=y_label,
        )


if __name__ == "__main__":
    main()
