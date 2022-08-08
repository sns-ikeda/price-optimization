from __future__ import annotations

from collections import defaultdict
from typing import Any, TypeVar

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_preprocess.preprocessor import DataPreprocessor, select_scaler
from src.evaluate.evaluator import Evaluator
from src.optimize.optimizer import Optimizer
from src.optimize.params import ArtificialDataParameter, RealDataParameter
from src.optimize.result import Result
from src.predict.predictor import Predictor, PredictorHandler
from src.utils.dict_converter import dict2json
from src.utils.paths import RESULT_DIR

IndexSet = TypeVar("IndexSet")
Constant = TypeVar("Constant")
Algorithm = TypeVar("Algorithm")


class Simulator:
    def __init__(
        self,
        data_type: str,
        config_data: dict[str, Any],
        config_opt: dict[str, Any],
        config_algo: dict[str, Any],
        config_pred: dict[str, Any],
    ) -> None:
        self.data_type = data_type
        self.config_data = config_data
        self.config_opt = config_opt
        self.config_algo = config_algo
        self.config_pred = config_pred
        self.artificial_results_dict: dict[tuple[str, str], list[Result]] = dict()
        self.realworld_results_dict: dict[tuple[str, str], list[Result]] = dict()
        self.realworld_results_detail_dict: dict[tuple[str, str], list[Result]] = dict()
        self.data_params: list[ArtificialDataParameter | RealDataParameter] = self.make_data_params(
            config_data=config_data, data_type=data_type
        )

    def run(self, iteration: int = 1) -> None:
        if self.data_type == "artificial":
            self.run_artificial(iteration)
        elif self.data_type == "realworld":
            self.run_realworld()

    def run_artificial(self, iteration: int) -> None:
        """人工データによるシミュレーションを実行"""
        possible_models = ["POORT_L", "POORT_L_alpha"]
        model_algo_names = self.make_model_algo_names()
        algo_settings = self.config_algo
        for model_name, algo_name in model_algo_names:
            if model_name not in possible_models:
                continue

            results: list[Result] = []
            for data_param in self.data_params:
                for i in range(iteration):
                    data_param.seed = i
                    optimizer = Optimizer(
                        model_name=model_name, algo_name=algo_name, data_param=data_param
                    )
                    optimizer.run(**algo_settings[algo_name])
                    results.append(optimizer.result)
            self.artificial_results_dict[(model_name, algo_name)] = results

    def run_realworld(self) -> None:
        """実データによるシミュレーションを実行"""
        model_algo_names = self.make_model_algo_names()
        algo_settings = self.config_algo
        pred_results_dict = defaultdict(lambda: defaultdict(dict))

        # データセットごとの評価
        for dataset_name, data_settings in self.config_data["realworld"].items():
            # データの前処理
            dp = DataPreprocessor(dataset_name)
            processed_df = dp.preprocess(**data_settings)
            target_cols = dp.get_target_cols(prefix=data_settings["target_col"])
            feature_cols = dp.get_feature_cols(target_cols=target_cols)
            label2item = dp.get_label2item(target_cols=target_cols)

            train_df, test_df = train_test_split(
                processed_df, train_size=0.7, test_size=0.3, shuffle=False
            )
            test_df.reset_index(drop=True, inplace=True)

            # モデルごとの評価
            for model_name, algo_name in model_algo_names:
                predictor_name = self.config_opt["model"][model_name]["prediction"]
                # データのスケーリング
                scaling_type = self.config_pred[predictor_name]["scaling"]
                scaler = select_scaler(scaling_type=scaling_type)
                X_train = pd.DataFrame(
                    scaler.fit_transform(train_df[feature_cols]), columns=feature_cols
                )
                X_test = pd.DataFrame(scaler.transform(test_df[feature_cols]), columns=feature_cols)
                scaled_train_df = pd.concat([X_train, train_df[target_cols]], axis=1)
                scaled_test_df = pd.concat([X_test, test_df[target_cols]], axis=1)
                assert len(train_df) == len(scaled_train_df)
                assert len(test_df) == len(scaled_test_df)

                # 予測モデルを構築
                predictors = PredictorHandler(
                    train_df=scaled_train_df,
                    test_df=scaled_test_df,
                    label2item=label2item,
                    predictor_name=predictor_name,
                )
                predictors.run()

                # 予測モデルの結果を格納
                pred_results_dict[dataset_name][predictor_name] = predictors.result
                dict2json(
                    target_dict=pred_results_dict,
                    save_path=RESULT_DIR / "json" / "pred_result.json",
                )
                # 商品ごとの価格候補を取得
                item2prices = dp.get_item2prices(
                    df=scaled_train_df,
                    num_of_prices=data_settings["num_of_prices"],
                    items=list(label2item.values()),
                )
                data_param = RealDataParameter(
                    item2predictor=predictors.item2predictor, item2prices=item2prices
                )
                # 価格最適化を実行
                optimizer = Optimizer(
                    model_name=model_name, algo_name=algo_name, data_param=data_param
                )
                optimizer.run(**algo_settings[algo_name])

                # 計算した最適価格の評価
                result, result_detail = self.evaluate(
                    test_df=scaled_test_df,
                    label2item=label2item,
                    item2predictor=predictors.item2predictor,
                    opt_prices=optimizer.opt_prices,
                )
                self.realworld_results_dict[(model_name, algo_name)] = result
                self.realworld_results_detail_dict[(model_name, algo_name)] = result_detail

    @staticmethod
    def evaluate(
        test_df: pd.DataFrame,
        label2item: dict[str, str],
        item2predictor: dict[str, Predictor],
        opt_prices: dict[str, float],
    ):
        evaluator = Evaluator(
            test_df=test_df,
            label2item=label2item,
            item2predictor=item2predictor,
            opt_prices=opt_prices,
        )
        evaluator.run()
        return evaluator.result, evaluator.result_item

    def make_model_algo_names(self) -> tuple[str, str]:
        model_settings = self.config_opt["model"]
        model_algo_names = [
            (model_name, algo_name)
            for model_name in model_settings
            for algo_name in model_settings[model_name]["algorithm"]
        ]
        return model_algo_names

    @staticmethod
    def make_data_params(
        config_data: dict[str, Any], data_type: str, **kwargs
    ) -> list[ArtificialDataParameter | RealDataParameter]:
        """シミュレーションで設定するパラメータの生成"""
        data_params = []
        if data_type == "artificial":
            param = config_data[data_type]["params"]
            for num_of_items in param["num_of_items"]:
                data_param = ArtificialDataParameter(
                    num_of_items=num_of_items,
                    num_of_prices=param["num_of_prices"],
                    num_of_other_features=param["num_of_other_features"],
                    depth_of_trees=param["depth_of_trees"],
                    base_price=param["base_price"],
                )
                data_params.append(data_param)
        return data_params
