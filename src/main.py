from __future__ import annotations

import itertools
from argparse import ArgumentParser
from typing import Optional

from src.configs import ArtificialConfig, RealworldConfig, read_config
from src.optimize.params import ArtificialDataParameter, RealDataParameter
from src.optimize.predictor2model import predictor2model
from src.optimize.processing.result_processor import average_results_dict
from src.optimize.result import OptResult
from src.simulator import Simulator
from src.utils.dict_converter import dict2json
from src.utils.dict_flattener import flatten_dict
from src.utils.paths import RESULT_DIR
from src.utils.plot import save_image


def get_args() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--data_type", type=str, default=None)
    args = parser.parse_args()
    return args


def find_results_by_data_param(
    results: list[OptResult],
    data_param: ArtificialDataParameter | RealDataParameter,
    attribute: Optional[str] = None,
) -> list[OptResult | float]:
    if attribute is None:
        return [result for result in results if result.data_param == data_param]
    else:
        return [getattr(result, attribute) for result in results if result.data_param == data_param]


def main():
    args = get_args()
    data_type = args.data_type
    if data_type is None:
        return

    _config = read_config("config.yaml")[data_type]
    if data_type == "artificial":
        artificial_config = ArtificialConfig(**_config)
        simulator = Simulator(data_type=data_type, artificial_config=artificial_config)
        simulator.run()
        # 可視化しやすいように結果を整形
        transformed_results_dict = dict()
        for data_param in simulator.data_params:
            _result_dict = {
                model_algo_pair: find_results_by_data_param(results=results, data_param=data_param)
                for model_algo_pair, results in simulator.artificial_results_dict.items()
            }
            for model_algo_pair, results in _result_dict.items():
                model_name = model_algo_pair[0]
                algo_name = model_algo_pair[1]
                transformed_results_dict.setdefault(model_name, dict()).setdefault(
                    algo_name, dict()
                )[data_param.num_of_items] = results

        # 結果の格納
        result_prefixes = ["calculation_time", "objective"]
        for result_prefix in result_prefixes:
            # 結果を保存
            save_name = (
                f"{result_prefix}_K{artificial_config.num_of_prices}_"
                + f"D{artificial_config.num_of_other_features}_DoT{artificial_config.depth_of_trees}"
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
                    f"{result_prefix}_{model}_K{artificial_config.num_of_prices}_"
                    + f"D{artificial_config.num_of_other_features}_DoT{artificial_config.depth_of_trees}"
                )
                # png形式で保存
                save_image(
                    avg_results_dict=avg_results_dict_,
                    dir_path=RESULT_DIR / data_type / "optimize",
                    image_name=save_name,
                    y_label=y_label,
                )

    elif data_type == "realworld":
        pred_result_train, pred_result_test = dict(), dict()
        eval_result, eval_result_detail = dict(), dict()

        dataset_name = _config["dataset_name"]
        predictor_names = _config["predictor_names"]
        algo_names = _config["algo_names"]
        num_of_prices = _config["num_of_prices"]
        train_sizes = _config["train_sizes"]
        for train_size, predictor_name, algo_name in itertools.product(
            train_sizes, predictor_names, algo_names
        ):
            realworld_config = RealworldConfig(
                dataset_name=dataset_name,
                predictor_name=predictor_name,
                algo_name=algo_name,
                num_of_prices=num_of_prices,
                train_size=train_size,
            )
            simulator = Simulator(
                data_type=data_type,
                realworld_config=realworld_config,
            )
            simulator.run()
            # 学習データへの学習結果を格納
            pred_result_train.setdefault(train_size, dict())[
                predictor_name
            ] = simulator.pred_result_train

            # テストデータへの学習結果を格納
            pred_result_test.setdefault(train_size, dict())[
                predictor_name
            ] = simulator.pred_result_test

            # 価格の評価結果を格納
            opt_prices = {"opt_prices": simulator.opt_result.opt_prices}
            try:
                q_train = {"q_train": simulator.opt_result.variable.q}
            except AttributeError:
                q_train = {"q_train": None}
            model_name = predictor2model[predictor_name]
            eval_result.setdefault(train_size, dict()).setdefault(model_name, dict())[
                algo_name
            ] = simulator.eval_result
            eval_result_detail.setdefault(train_size, dict()).setdefault(model_name, dict())[
                algo_name
            ] = dict(**simulator.eval_result_item, **opt_prices, **q_train)

        # json形式で結果を出力
        dict2json(
            target_dict=pred_result_test,
            save_path=RESULT_DIR / data_type / "predict" / "pred_result_test.json",
        )
        dict2json(
            target_dict=pred_result_train,
            save_path=RESULT_DIR / data_type / "predict" / "pred_result_train.json",
        )
        dict2json(
            target_dict=eval_result,
            save_path=RESULT_DIR / data_type / "optimize" / "eval_result.json",
        )
        dict2json(
            target_dict=eval_result_detail,
            save_path=RESULT_DIR / data_type / "optimize" / "eval_result_detail.json",
        )


if __name__ == "__main__":
    main()
