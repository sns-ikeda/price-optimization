from __future__ import annotations

from typing import Any, Optional, TypeVar

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_preprocess.preprocessor import (
    DataPreprocessor,
    get_item2avg_prices,
    get_item2prices,
    get_label2item,
    inverse_transform,
    select_scaler,
)
from src.evaluate.evaluator import Evaluator
from src.optimize.optimizer import Optimizer
from src.optimize.params import ArtificialDataParameter, RealDataParameter, make_data_params
from src.optimize.result import Result
from src.predict.predictor import PredictorHandler

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
        self.train_predictors: dict[tuple[str, str], PredictorHandler] = dict()
        self.test_predictors: dict[tuple[str, str], PredictorHandler] = dict()
        self.optimizers: dict[tuple[str, str, str], Optimizer] = dict()
        self.evaluators: dict[tuple[str, str, str], Evaluator] = dict()
        self.data_params: list[ArtificialDataParameter | RealDataParameter] = make_data_params(
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

        # データセットごとの評価
        for dataset_name, data_settings in self.config_data["realworld"].items():
            # データの前処理
            self.preprocess(dataset_name, **data_settings)

            # モデルごとの評価
            for model_name, algo_name in model_algo_names:
                predictor_name = self.config_opt["model"][model_name]["prediction"]
                # データのスケーリング
                scaling_type = self.config_pred[predictor_name]["scaling"]
                self.scale(scaling_type=scaling_type)

                # 予測モデルの構築
                self.train(dataset_name=dataset_name, predictor_name=predictor_name)

                # 商品ごとの価格候補を取得
                item2prices = get_item2prices(
                    df=self.scaled_train_df,
                    num_of_prices=data_settings["num_of_prices"],
                    items=list(self.label2item.values()),
                )
                # 最適化モデルの入力データを作成
                data_param = RealDataParameter(
                    num_of_prices=data_settings["num_of_prices"],
                    item2predictor=self.train_predictors[
                        dataset_name, predictor_name
                    ].item2predictor,
                    item2prices=item2prices,
                    g=self.calc_g(
                        X=self.scaled_train_df[self.feature_cols].tail(1),
                        items=self.items,
                    ),
                )
                # 価格最適化を実行
                optimizer = Optimizer(
                    model_name=model_name, algo_name=algo_name, data_param=data_param
                )
                optimizer.run(**algo_settings[algo_name])
                self.optimizers[dataset_name, model_name, algo_name] = optimizer

                # 計算した最適価格の評価
                avg_prices = get_item2avg_prices(df=self.train_df, items=self.items)
                scaled_avg_prices = inverse_transform(
                    scaler=self.train_scaler,
                    item2prices=avg_prices,
                    X_df=self.train_df[self.feature_cols],
                )
                evaluator = Evaluator(
                    scaled_test_df=self.scaled_test_df,
                    original_test_df=self.test_df,
                    label2item=self.label2item,
                    item2predictor=self.test_predictors[
                        dataset_name, predictor_name
                    ].item2predictor,
                    scaled_opt_prices=optimizer.result.opt_prices,
                    scaled_avg_prices=scaled_avg_prices,
                    scaler=self.train_scaler,
                )
                evaluator.run()
                self.evaluators[dataset_name, model_name, algo_name] = evaluator

    def make_model_algo_names(self) -> tuple[str, str]:
        """実行するモデルとアルゴリズムの名前のtupleを生成"""
        model_settings = self.config_opt["model"]
        model_algo_names = [
            (model_name, algo_name)
            for model_name in model_settings
            for algo_name in model_settings[model_name]["algorithm"]
        ]
        return model_algo_names

    def preprocess(self, dataset_name, **data_settings) -> None:
        """データの基本的な前処理"""
        dp = DataPreprocessor(dataset_name)
        processed_df = dp.preprocess(**data_settings)
        self.target_cols = dp.get_target_cols(prefix=data_settings["target_col"])
        self.feature_cols = dp.get_feature_cols(target_cols=self.target_cols)
        self.label2item = get_label2item(target_cols=self.target_cols)
        self.items = list(self.label2item.values())

        self.train_df, self.test_df = train_test_split(
            processed_df, train_size=0.5, test_size=0.5, shuffle=False
        )
        self.test_df.reset_index(drop=True, inplace=True)

    def scale(self, scaling_type: Optional[str]) -> None:
        """正規化・標準化などのスケーリングを実施"""
        self.train_scaler = select_scaler(scaling_type=scaling_type)
        self.test_scaler = select_scaler(scaling_type=scaling_type)
        if select_scaler(scaling_type=scaling_type) is None:
            self.scaled_train_df = self.train_df.copy()
            self.scaled_test_df = self.test_df.copy()
        else:
            X_train = pd.DataFrame(
                self.train_scaler.fit_transform(self.train_df[self.feature_cols]),
                columns=self.feature_cols,
            )
            X_test = pd.DataFrame(
                self.test_scaler.fit_transform(self.test_df[self.feature_cols]),
                columns=self.feature_cols,
            )
            self.scaled_train_df = pd.concat([X_train, self.train_df[self.target_cols]], axis=1)
            self.scaled_test_df = pd.concat([X_test, self.test_df[self.target_cols]], axis=1)

        assert self.train_df.shape == self.scaled_train_df.shape
        assert self.test_df.shape == self.scaled_test_df.shape
        assert self.train_df.columns.tolist() == self.scaled_train_df.columns.tolist()
        assert self.test_df.columns.tolist() == self.scaled_test_df.columns.tolist()

    def train(self, dataset_name: str, predictor_name: str) -> None:
        """学習・テストデータに対する予測モデルを作成"""
        # 学習データに対する予測モデルを構築
        train_predictors = PredictorHandler(
            train_df=self.scaled_train_df,
            label2item=self.label2item,
            predictor_name=predictor_name,
            prefix="train",
        )
        train_predictors.run()
        self.train_predictors[dataset_name, predictor_name] = train_predictors

        # テストデータに対する予測モデルを構築
        test_predictors = PredictorHandler(
            train_df=self.scaled_test_df,
            label2item=self.label2item,
            predictor_name=predictor_name,
            prefix="test",
        )
        test_predictors.run()
        self.test_predictors[dataset_name, predictor_name] = test_predictors

    @staticmethod
    def calc_g(X: pd.DataFrame, items: list[str]) -> dict[str, float]:
        df = X.copy().head(1)
        feature_cols = X.columns.tolist()
        price_cols = ["PRICE" + "_" + item for item in items]
        other_feature_cols = [col for col in feature_cols if col not in price_cols]
        g = {col: float(df[col]) for col in other_feature_cols}
        return g

    @staticmethod
    def calc_actual_sales(df: pd.Series, items: list[str]) -> float:
        sales_item = dict()
        for item in items:
            price_col = "PRICE" + "_" + item
            target_col = "UNITS" + "_" + item
            sales = df[price_col] * df[target_col]
            sales_item[item] = sales
        return sales_item
