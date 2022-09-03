from __future__ import annotations

from typing import Any, TypeVar

import pandas as pd
import yaml
from logzero import logger
from sklearn.model_selection import train_test_split

from src.data_preprocess.preprocessor import (
    DataPreprocessor,
    get_item2avg_prices,
    get_item2prices,
    get_label2item,
)
from src.evaluate.evaluator import Evaluator
from src.optimize.optimizer import Optimizer
from src.optimize.params import ArtificialDataParameter, RealDataParameter, make_data_params
from src.optimize.result import Result
from src.predict.predictor import PredictorHandler
from src.utils.dict_converter import dict2yaml
from src.utils.handle_module import get_object_from_module
from src.utils.paths import PRED_DIR

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
        self.train_size = None
        self.test_size = None

    def run(self, iteration: int = 1, train_size: float = 0.5) -> None:
        if self.data_type == "artificial":
            self.run_artificial(iteration)
        elif self.data_type == "realworld":
            self.run_realworld(train_size)

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

    def run_realworld(self, train_size: float) -> None:
        """実データによるシミュレーションを実行"""
        self.train_size = round(train_size, 3)
        self.test_size = round(1 - self.train_size, 3)
        model_algo_names = self.make_model_algo_names()
        algo_settings = self.config_algo

        # データセットごとの評価
        for dataset_name, data_settings in self.config_data["realworld"].items():
            # データの前処理
            self.preprocess(dataset_name, **data_settings)

            # モデルごとの評価
            for model_name, algo_name in model_algo_names:
                predictor_name = self.config_opt["model"][model_name]["prediction"]

                # 予測モデルの構築
                self.train(dataset_name=dataset_name, predictor_name=predictor_name)

                # 商品ごとの価格候補を取得
                item2prices = get_item2prices(
                    df=self.train_df,
                    num_of_prices=data_settings["num_of_prices"],
                    items=list(self.label2item.values()),
                )
                logger.info(f"price candidates: {item2prices}")
                # 最適化モデルの入力データを作成
                data_param = RealDataParameter(
                    num_of_prices=data_settings["num_of_prices"],
                    item2predictor=self.train_predictors[
                        dataset_name, predictor_name
                    ].item2predictor,
                    item2prices=item2prices,
                    g=self.calc_g(
                        X=self.train_df[self.feature_cols].tail(1),
                        items=self.items,
                    ),
                )
                # 価格最適化しない場合の結果
                actual_sales_item = self.calc_actual_sales(self.train_df.tail(1), self.items)
                actual_total_sales = sum(actual_sales_item.values())
                logger.info(f"actual_total_sales: {actual_total_sales}")

                # 価格最適化を実行
                optimizer = Optimizer(
                    model_name=model_name, algo_name=algo_name, data_param=data_param
                )
                optimizer.run(**algo_settings[algo_name])
                self.optimizers[dataset_name, model_name, algo_name] = optimizer

                # 計算した最適価格の評価
                avg_prices = get_item2avg_prices(df=self.train_df, items=self.items)
                if optimizer.result.opt_prices:
                    opt_prices = optimizer.result.opt_prices
                else:
                    raise Exception("couldn't get optimal prices")
                evaluator = Evaluator(
                    test_df=self.test_df,
                    item2predictor=self.test_predictors[
                        dataset_name, predictor_name
                    ].item2predictor,
                    opt_prices=opt_prices,
                    avg_prices=avg_prices,
                    item2prices=item2prices,
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
        logger.info(f"columns: {list(processed_df.columns)}")
        self.target_cols = dp.get_target_cols(prefix=data_settings["target_col"])
        self.feature_cols = dp.get_feature_cols(target_cols=self.target_cols)
        self.label2item = get_label2item(target_cols=self.target_cols)
        self.items = list(self.label2item.values())

        logger.info(f"train_size: {self.train_size}, test_size: {self.test_size}")
        self.train_df, self.test_df = train_test_split(
            processed_df, train_size=self.train_size, test_size=self.test_size, shuffle=False
        )
        logger.info(
            f"# of rows [train]: {len(self.train_df)}, # of rows [test]: {len(self.test_df)}"
        )
        self.test_df.reset_index(drop=True, inplace=True)

    def train(self, dataset_name: str, predictor_name: str) -> None:
        """学習・テストデータに対する予測モデルを作成"""
        yaml_path = PRED_DIR / predictor_name / "hyper_parameter.yaml"
        try:
            with open(yaml_path) as file:
                params = yaml.safe_load(file.read())
                if params is None:
                    params_train = None
                    params_test = None
                else:
                    params_train = params[dataset_name][predictor_name]["train"]
                    params_test = params[dataset_name][predictor_name]["test"]
        except FileNotFoundError or AttributeError:
            params_train = None
            params_test = None
        logger.info(f"params_train: {params_train}, params_test: {params_test}")
        # item2features = {item: ["PRICE_" + item] for item in self.items}

        # 学習データに対する予測モデルを構築
        train_predictors = PredictorHandler(
            train_df=self.train_df,
            test_df=self.test_df,
            label2item=self.label2item,
            predictor_name=predictor_name,
            prefix="train",
            suffix=f"tr{self.train_size}",
            params=params_train,
        )
        train_predictors.run()
        self.train_predictors[dataset_name, predictor_name] = train_predictors

        params_test = {item: {"max_depth": 5, "cp": 0.001} for item in self.items}
        # テストデータに対する予測モデルを構築
        test_predictors = PredictorHandler(
            train_df=self.test_df,
            label2item=self.label2item,
            predictor_name="ORT_LH",
            prefix="test",
            suffix=f"tr{self.train_size}",
            params=params_test,
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

    def tune_params(self) -> None:
        """ハイパラチューニングを実行"""
        model_algo_names = self.make_model_algo_names()
        self.config_algo

        # データセットごとの評価
        best_params = dict()
        for dataset_name, data_settings in self.config_data["realworld"].items():
            best_params[dataset_name] = dict()
            # データの前処理
            self.preprocess(dataset_name, **data_settings)

            # モデルごとに実行
            for model_name, algo_name in model_algo_names:
                predictor_name = self.config_opt["model"][model_name]["prediction"]
                if algo_name == "solver_naive":
                    best_params[dataset_name][predictor_name] = self._tune_params(predictor_name)
        dict2yaml(
            target_dict=best_params, save_path=PRED_DIR / predictor_name / "hyper_parameter.yaml"
        )

    def _tune_params(self, predictor_name: str) -> dict[str, float]:
        module_path = PRED_DIR / predictor_name / "tune_params.py"
        tune_params = get_object_from_module(module_path, "tune_params")
        X_train = self.train_df.drop(columns=self.target_cols)
        X_test = self.test_df.drop(columns=self.target_cols)

        # 商品ごとにモデルのハイパラをチューニング
        best_params = dict()
        best_params["train"] = dict()
        best_params["test"] = dict()
        for target_col, item in self.label2item.items():
            y_train = self.train_df[[target_col]]
            y_test = self.test_df[[target_col]]
            best_params["train"][item] = tune_params(X_train, y_train)
            best_params["test"][item] = tune_params(X_test, y_test)
        return best_params
