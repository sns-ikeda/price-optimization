from __future__ import annotations

from argparse import ArgumentParser
from collections import defaultdict
from typing import Optional

import pandas as pd

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

    if data_type == "artificial":
        num_iteration = CONFIG_SIM["num_iteration"]
        simulator = Simulator(
            data_type=data_type,
            config_data=CONFIG_DATA,
            config_opt=CONFIG_OPT,
            config_algo=CONFIG_ALGO,
            config_pred=CONFIG_PRED,
        )
        simulator.run(num_iteration)
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
        pred_result_train = defaultdict(lambda: defaultdict(dict))
        pred_result_test = defaultdict(lambda: defaultdict(dict))
        eval_result = defaultdict(lambda: defaultdict(dict))
        eval_result_detail = defaultdict(lambda: defaultdict(dict))
        actual_sales_df = pd.DataFrame()
        pred_sales_at_actual_price_df = pd.DataFrame()
        pred_sales_at_average_price_df = pd.DataFrame()
        pred_sales_at_opt_price_df = pd.DataFrame()
        r2_train_train_df = pd.DataFrame()
        r2_train_test_df = pd.DataFrame()
        r2_test_train_df = pd.DataFrame()
        rmse_train_train_df = pd.DataFrame()
        rmse_train_test_df = pd.DataFrame()
        rmse_test_train_df = pd.DataFrame()

        train_sizes = CONFIG_SIM["train_size"]
        for train_size in train_sizes:
            simulator = Simulator(
                data_type=data_type,
                config_data=CONFIG_DATA,
                config_opt=CONFIG_OPT,
                config_algo=CONFIG_ALGO,
                config_pred=CONFIG_PRED,
            )
            simulator.run(train_size=train_size)
            # 学習データへの学習結果を格納
            for dataset_name, predictor_name in simulator.train_predictors.keys():
                predictors = simulator.train_predictors[dataset_name, predictor_name]
                pred_result_train[train_size][predictor_name] = predictors.result
                r2_train_train_df.loc[predictor_name, train_size] = predictors.result["r2"][
                    "train"
                ]["mean"]
                r2_train_test_df.loc[predictor_name, train_size] = predictors.result["r2"]["test"][
                    "mean"
                ]
                rmse_train_train_df.loc[predictor_name, train_size] = predictors.result["rmse"][
                    "train"
                ]["mean"]
                rmse_train_test_df.loc[predictor_name, train_size] = predictors.result["rmse"][
                    "test"
                ]["mean"]

            # テストデータへの学習結果を格納
            for dataset_name, predictor_name in simulator.test_predictors.keys():
                predictors = simulator.test_predictors[dataset_name, predictor_name]
                pred_result_test[train_size][predictor_name] = predictors.result
                r2_test_train_df.loc[predictor_name, train_size] = predictors.result["r2"]["train"][
                    "mean"
                ]
                rmse_test_train_df.loc[predictor_name, train_size] = predictors.result["rmse"][
                    "train"
                ]["mean"]

            # 価格の評価結果を格納
            for dataset_name, model_name, algo_name in simulator.evaluators.keys():
                optimizer = simulator.optimizers[dataset_name, model_name, algo_name]
                opt_prices = {"opt_prices": optimizer.result.opt_prices}
                try:
                    q_train = {"q_train": optimizer.result.variable.q}
                except AttributeError:
                    q_train = {"q_train": None}
                evaluator = simulator.evaluators[dataset_name, model_name, algo_name]
                eval_result[train_size][model_name][algo_name] = evaluator.result
                eval_result_detail[train_size][model_name][algo_name] = dict(
                    **evaluator.result_item, **opt_prices, **q_train
                )
                actual_sales_df.loc[model_name, train_size] = evaluator.result["actual_sales"]
                pred_sales_at_actual_price_df.loc[model_name, train_size] = evaluator.result[
                    "pred_sales_at_actual_price"
                ]
                pred_sales_at_average_price_df.loc[model_name, train_size] = evaluator.result[
                    "pred_sales_at_average_price"
                ]
                pred_sales_at_opt_price_df.loc[model_name, train_size] = evaluator.result[
                    "pred_sales_at_opt_price"
                ]
                evaluator.result_df.to_csv(
                    RESULT_DIR / data_type / "predict" / f"result_df_{model_name}.csv"
                )

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
        # csv形式で結果を整形・出力
        actual_sales_df.to_csv(RESULT_DIR / data_type / "optimize" / "actual_sales.csv")
        pred_sales_at_actual_price_df.to_csv(
            RESULT_DIR / data_type / "optimize" / "pred_sales_at_actual_price.csv"
        )
        pred_sales_at_average_price_df.to_csv(
            RESULT_DIR / data_type / "optimize" / "pred_sales_at_average_price.csv"
        )
        pred_sales_at_opt_price_df.to_csv(
            RESULT_DIR / data_type / "optimize" / "pred_sales_at_opt_price.csv"
        )
        r2_train_train_df.to_csv(RESULT_DIR / data_type / "predict" / "r2_train_train.csv")
        r2_train_test_df.to_csv(RESULT_DIR / data_type / "predict" / "r2_train_test.csv")
        r2_test_train_df.to_csv(RESULT_DIR / data_type / "predict" / "r2_test_train.csv")
        rmse_train_train_df.to_csv(RESULT_DIR / data_type / "predict" / "rmse_train_train.csv")
        rmse_train_test_df.to_csv(RESULT_DIR / data_type / "predict" / "rmse_train_test.csv")
        rmse_test_train_df.to_csv(RESULT_DIR / data_type / "predict" / "rmse_test_train.csv")


if __name__ == "__main__":
    main()
