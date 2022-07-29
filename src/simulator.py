from __future__ import annotations

from typing import Any, TypeVar

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_preprocess.data_preprocessor import DataPreprocessor
from src.optimize.optimizer import Optimizer
from src.optimize.params import ArtificialDataParameter, RealDataParameter
from src.optimize.result import Result
from src.predict.predictor import MakePredictor, Predictor

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
    ) -> None:
        self.data_type = data_type
        self.config_data = config_data
        self.config_opt = config_opt
        self.config_algo = config_algo
        self.artificial_results_dict: dict[tuple[str, str], list[Result]] = dict()
        self.realworld_results_dict: dict[tuple[str, str], list[Result]] = dict()
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
        model_algo_names = self.make_model_algo_names()
        algo_settings = self.config_algo
        for model_name, algo_name in model_algo_names:
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

        for dataset, data_settings in self.config_data["realworld"].items():
            data_preprocessor = DataPreprocessor(dataset)
            processed_df = data_preprocessor.run(**data_settings)
            train_df, test_df = train_test_split(
                processed_df, train_size=0.7, test_size=0.3, shuffle=False
            )
            num_of_prices = data_settings["num_of_prices"]
            base_target_col = data_settings["target_col"]
            target_cols = [col for col in train_df.columns if base_target_col in col]

            for model_name, algo_name in model_algo_names:
                predictor_name = self.config_opt["model"][model_name]["prediction"]
                make_predictor = MakePredictor(
                    train_df=train_df, target_cols=target_cols, model_name=predictor_name
                )
                make_predictor.run()
                data_param = RealDataParameter(
                    item2predictor=make_predictor.item2predictor, num_of_prices=num_of_prices
                )
                optimizer = Optimizer(
                    model_name=model_name, algo_name=algo_name, data_param=data_param
                )
                optimizer.run(**algo_settings[algo_name])
                result = Simulator.evaluate(
                    test_df=test_df,
                    target_cols=target_cols,
                    item2predictor=make_predictor.item2predictor,
                    opt_prices=optimizer.opt_prices,
                )
                self.realworld_results_dict[model_name][algo_name] = result

    @staticmethod
    def evaluate(
        test_df: pd.DataFrame,
        target_cols: list[str],
        item2predictor: dict[str, Predictor],
        opt_prices: dict[str, float],
    ):
        pass

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
