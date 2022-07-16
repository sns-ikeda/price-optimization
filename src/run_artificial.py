from __future__ import annotations

from tqdm import tqdm

from src.configs import CONFIG_DATA, CONFIG_SIM
from src.optimization.optimizer import Optimizer
from src.optimization.params import ArtificialDataParameter
from src.optimization.processing.result_processor import (
    average_results,
    transform_artificial_results,
)
from src.utils.dict_converter import dict2json
from src.utils.flatten_dict import flatten_dict
from src.utils.paths import RESULT_DIR
from src.utils.plot import save_image


def main():
    data_type = "artificial"
    results = []
    iteration = CONFIG_SIM["iteration"]
    params = CONFIG_DATA[data_type]["params"]
    num_of_items = params["num_of_items"]
    params.pop("num_of_items", None)

    # 最適化計算を実行
    for _ in tqdm(range(iteration)):
        result_dict = dict()
        for i in num_of_items:
            data_params = {data_type: ArtificialDataParameter(num_of_items=i, **params)}
            optimizer = Optimizer(data_type=data_type, data_params=data_params)
            optimizer.run()
            result_dict[i] = optimizer.result
        results.append(result_dict)

    # 結果の後処理
    result_prefixes = ["calculation_time", "objective"]
    for result_prefix in result_prefixes:
        transformed_results_dict: dict[str, dict[int, list[float]]] = transform_artificial_results(
            results=results, attribute=result_prefix
        )
        avg_results_dict = average_results(results_dict=transformed_results_dict)
        flattened_results_dict = flatten_dict(avg_results_dict)
        # 結果を保存
        save_name = (
            f"{result_prefix}_K{params['num_of_prices']}_"
            + f"D{params['num_of_other_features']}_DoT{params['depth_of_trees']}"
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
            avg_results_dict=flattened_results_dict,
            dir_path=RESULT_DIR / "png",
            image_name=save_name,
            y_label=y_label,
        )
        # モデルごとの結果を保存
        for model, avg_results_dict_ in avg_results_dict.items():
            save_name = (
                f"{result_prefix}_{model}_K{params['num_of_prices']}_"
                + f"D{params['num_of_other_features']}_DoT{params['depth_of_trees']}"
            )
            # png形式で保存
            save_image(
                avg_results_dict=avg_results_dict_,
                dir_path=RESULT_DIR / "png",
                image_name=save_name,
                y_label=y_label,
            )


if __name__ == "__main__":
    main()
