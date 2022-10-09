from __future__ import annotations

from typing import Optional, TypeVar

import numpy as np
import pandas as pd
from logzero import logger
from sklearn.model_selection import train_test_split

from src.configs import ALGO_CONFIG, ArtificialConfig, RealworldConfig
from src.data_preprocess.preprocessor import (
    DataPreprocessor,
    get_item2avg_prices,
    get_item2prices,
    get_label_from_item,
)
from src.evaluate.evaluator import Evaluator
from src.optimize.optimizer import Optimizer
from src.optimize.params import ArtificialDataParameter, RealDataParameter, calc_g, make_data_params
from src.optimize.predictor2model import predictor2model
from src.optimize.result import OptResult
from src.predict.predictor import Predictor, PredictorMaker

IndexSet = TypeVar("IndexSet")
Constant = TypeVar("Constant")
Algorithm = TypeVar("Algorithm")


class Simulator:
    def __init__(
        self,
        data_type: str,
        realworld_config: Optional[RealworldConfig] = None,
        artificial_config: Optional[ArtificialConfig] = None,
    ) -> None:
        self.data_type = data_type
        if data_type == "realworld":
            self.config = realworld_config
        elif data_type == "artificial":
            self.config = artificial_config

        self.dp = None
        self.items = []
        self.artificial_results_dict: dict[tuple[str, str], list[OptResult]] = dict()
        self.optimizer: Optional[Optimizer] = None
        self.evaluator: Optional[Evaluator] = None
        self.data_params: list[ArtificialDataParameter | RealDataParameter] = make_data_params(
            config=self.config, data_type=data_type
        )
        self.pred_result_train = dict()
        self.pred_result_test = dict()
        self.opt_results = []
        self.eval_results = []
        self.eval_results_item = []

    def run(self) -> None:
        if self.data_type == "artificial":
            self.run_artificial()
        elif self.data_type == "realworld":
            self.run_realworld()

    def run_artificial(self) -> None:
        """人工データによるシミュレーションを実行"""

        for predictor_name in self.config.predictor_names:
            model_name = predictor2model[predictor_name]
            for algo_name in self.config.algo_names:
                results: list[OptResult] = []
                for data_param in self.data_params:
                    for i in range(self.config.num_iteration):
                        data_param.seed = i
                        optimizer = Optimizer(
                            model_name=model_name, algo_name=algo_name, data_param=data_param
                        )
                        optimizer.run(**ALGO_CONFIG[algo_name])
                        results.append(optimizer.result)
                self.artificial_results_dict[(model_name, algo_name)] = results

    def data_preprocess(self) -> None:
        self.dp = DataPreprocessor(self.config.dataset_name)
        self.dp.run()
        self.items = list(self.dp.item2df.keys())

    def make_item2predictor(self) -> tuple[dict[str, Predictor], dict[str, Predictor]]:
        item2predictor_train, item2predictor_test = dict(), dict()
        _pred_result_train, _pred_result_test = dict(), dict()
        for item, df in self.dp.item2df.items():
            train_size = self.config.train_size
            test_size = round(1 - train_size, 3)
            train_df, test_df = train_test_split(
                df, train_size=train_size, test_size=test_size, shuffle=False
            )
            target_col = get_label_from_item(item=item)

            # 訓練データに対する予測モデルを構築
            pm_train = PredictorMaker(
                predictor_name=self.config.predictor_name,
                train_df=train_df,
                test_df=test_df,
                target_col=target_col,
            )
            predictor_train = pm_train.run(
                train_or_test="train", suffix=str(self.config.train_size)
            )
            item2predictor_train[item] = predictor_train
            _pred_result_train[item] = pm_train.result

            # 検証データに対する予測モデルを構築
            pm_test = PredictorMaker(
                predictor_name="ORT_LH",
                train_df=test_df,
                target_col=target_col,
            )
            predictor_test = pm_test.run(train_or_test="test", suffix=str(self.config.train_size))
            item2predictor_test[item] = predictor_test
            _pred_result_test[item] = pm_test.result

        # 後処理
        self.pred_result_train = postproceess_pred_result(_pred_result_train)
        self.pred_result_test = postproceess_pred_result(_pred_result_test)
        return item2predictor_train, item2predictor_test

    def run_realworld(self) -> None:
        """実データによるシミュレーションを実行"""
        # データの前処理
        self.data_preprocess()

        # 商品ごとの予測モデルを構築
        item2predictor_train, item2predictor_test = self.make_item2predictor()

        # 訓練データと検証データに分割
        train_size = self.config.train_size
        test_size = round(1 - train_size, 3)
        train_df, test_df = train_test_split(
            self.dp.processed_df, train_size=train_size, test_size=test_size, shuffle=False
        )
        # 最適化モデルの入力データを作成
        item2prices = get_item2prices(
            df=train_df,
            num_of_prices=self.config.num_of_prices,
            items=self.items,
        )
        logger.info(f"price candidates: {item2prices}")
        avg_prices = get_item2avg_prices(df=train_df, items=self.items)

        for _, row in test_df.iterrows():
            row_df = row.to_frame().T
            g = calc_g(
                df=row_df,
                item2predictor=item2predictor_train,
            )
            data_param = RealDataParameter(
                num_of_prices=self.config.num_of_prices,
                item2predictor=item2predictor_train,
                item2prices=item2prices,
                g=g,
            )
            # 価格最適化しない場合の結果
            actual_sales_item = calc_actual_sales(row_df, self.items)
            actual_total_sales = sum(actual_sales_item.values())
            logger.info(f"actual_total_sales: {actual_total_sales}")

            # 価格最適化を実行
            model_name = predictor2model[self.config.predictor_name]
            self.optimizer = Optimizer(
                model_name=model_name, algo_name=self.config.algo_name, data_param=data_param
            )
            self.optimizer.run(**ALGO_CONFIG[self.config.algo_name])
            self.opt_results.append(self.optimizer.result)

            # 計算した最適価格の評価
            if self.optimizer.result.opt_prices:
                opt_prices = self.optimizer.result.opt_prices
                logger.info(f"opt_prices: {opt_prices}")
                logger.info(f"q: {self.optimizer.result.variable.q}")
            else:
                raise Exception("couldn't get optimal prices")
            if g:
                test_df_ = row_df
            else:
                test_df_ = test_df
            self.evaluator = Evaluator(
                test_df=test_df_,
                item2predictor=item2predictor_test,
                opt_prices=opt_prices,
                avg_prices=avg_prices,
                item2prices=item2prices,
            )
            self.evaluator.run()
            self.eval_results.append(self.evaluator.result)
            self.eval_results_item.append(self.evaluator.result_item)
            if g:
                pass
            else:
                break


def postproceess_pred_result(target_dict: dict[str, dict[str, dict[str, float]]]):
    d = dict()
    for item, dict_ in target_dict.items():
        d.setdefault("rmse", dict()).setdefault("train", dict())[item] = dict_["rmse"]["train"]
        d.setdefault("r2", dict()).setdefault("train", dict())[item] = dict_["r2"]["train"]
        try:
            d.setdefault("rmse", dict()).setdefault("test", dict())[item] = dict_["rmse"]["test"]
            d.setdefault("r2", dict()).setdefault("test", dict())[item] = dict_["r2"]["test"]
        except KeyError:
            pass
    d["rmse"]["train"]["mean"] = round(np.mean(list(d["rmse"]["train"].values())), 3)
    d["r2"]["train"]["mean"] = round(np.mean(list(d["r2"]["train"].values())), 3)
    try:
        d["rmse"]["test"]["mean"] = round(np.mean(list(d["rmse"]["test"].values())), 3)
        d["r2"]["test"]["mean"] = round(np.mean(list(d["r2"]["test"].values())), 3)
    except KeyError:
        pass
    return d


def calc_actual_sales(df: pd.Series, items: list[str]) -> float:
    sales_item = dict()
    for item in items:
        price_col = "PRICE" + "_" + item
        target_col = "UNITS" + "_" + item
        sales = df[price_col] * df[target_col]
        sales_item[item] = sales
    return sales_item
