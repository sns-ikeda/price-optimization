from __future__ import annotations

from argparse import ArgumentParser
from collections import defaultdict
from typing import Optional

from src.configs import CONFIG_ALGO, CONFIG_DATA, CONFIG_OPT, CONFIG_PRED, CONFIG_SIM
from src.optimize.params import ArtificialDataParameter, RealDataParameter
from src.optimize.processing.result_processor import average_results_dict
from src.optimize.result import Result
from src.simulator import Simulator
from src.utils.dict_converter import dict2json
from src.utils.flatten_dict import flatten_dict
from src.utils.paths import RESULT_DIR
from src.utils.plot import save_image


def get_args() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--data_type", type=str, default=None)
    args = parser.parse_args()
    return args


def find_results_by_data_param(
    results: list[Result],
    data_param: ArtificialDataParameter | RealDataParameter,
    attribute: Optional[str] = None,
) -> list[Result | float]:
    if attribute is None:
        return [result for result in results if result.data_param == data_param]
    else:
        return [getattr(result, attribute) for result in results if result.data_param == data_param]


def main():
    args = get_args()
    data_type = args.data_type
    if data_type is None:
        return

    num_iteration = CONFIG_SIM["num_iteration"]
    simulator = Simulator(
        data_type=data_type,
        config_data=CONFIG_DATA,
        config_opt=CONFIG_OPT,
        config_algo=CONFIG_ALGO,
        config_pred=CONFIG_PRED,
    )
    simulator.run(num_iteration)

    if data_type == "artificial":
        # 可視化しやすいように結果を整形
        transformed_results_dict = defaultdict(lambda: defaultdict(dict))
        for data_param in simulator.data_params:
            _result_dict = {
                model_algo_pair: find_results_by_data_param(results=results, data_param=data_param)
                for model_algo_pair, results in simulator.artificial_results_dict.items()
            }
            for model_algo_pair, results in _result_dict.items():
                model_name = model_algo_pair[0]
                algo_name = model_algo_pair[1]
                transformed_results_dict[model_name][algo_name][data_param.num_of_items] = results

        # 結果の格納
        result_prefixes = ["calculation_time", "objective"]
        for result_prefix in result_prefixes:
            # 結果を保存
            fixed_param = CONFIG_DATA[data_type]["params"]
            save_name = (
                f"{result_prefix}_K{fixed_param['num_of_prices']}_"
                + f"D{fixed_param['num_of_other_features']}_DoT{fixed_param['depth_of_trees']}"
            )
            avg_results_dict = average_results_dict(
                results_dict=transformed_results_dict, attribute=result_prefix
            )
            flattened_results_dict = flatten_dict(avg_results_dict)

            # jsonで保存
            dict2json(
                target_dict=avg_results_dict,
                save_path=RESULT_DIR / data_type / "optimize" / (save_name + ".json"),
            )
            # png形式で保存
            if result_prefix == "calculation_time":
                y_label = "calculation time [sec]"
            else:
                y_label = result_prefix
            save_image(
                avg_results_dict=flattened_results_dict,
                dir_path=RESULT_DIR / data_type / "optimize",
                image_name=save_name,
                y_label=y_label,
            )
            # モデルごとの結果を保存
            for model, avg_results_dict_ in avg_results_dict.items():
                save_name = (
                    f"{result_prefix}_{model}_K{fixed_param['num_of_prices']}_"
                    + f"D{fixed_param['num_of_other_features']}_DoT{fixed_param['depth_of_trees']}"
                )
                # png形式で保存
                save_image(
                    avg_results_dict=avg_results_dict_,
                    dir_path=RESULT_DIR / data_type / "optimize",
                    image_name=save_name,
                    y_label=y_label,
                )
    elif data_type == "realworld":
        # 結果の出力
        dict2json(
            target_dict=simulator.pred_results_dict,
            save_path=RESULT_DIR / data_type / "predict" / "pred_result.json",
        )
        dict2json(
            target_dict=simulator.realworld_results_dict,
            save_path=RESULT_DIR / data_type / "optimize" / "result.json",
        )
        dict2json(
            target_dict=simulator.realworld_results_detail_dict,
            save_path=RESULT_DIR / data_type / "optimize" / "result_detail.json",
        )


if __name__ == "__main__":
    main()
