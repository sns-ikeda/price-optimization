from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd
from logzero import logger
from sklearn.metrics import mean_squared_error

from src.predict.plot import plot
from src.utils.handle_module import get_object_from_module
from src.utils.paths import PRED_DIR


class Predictor:
    def __init__(self, model, feature_cols: list[str], target_col: str) -> None:
        self.model = model
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.item = None
        self.coef_dict = dict()

    def predict(self, X: pd.DataFrame) -> np.array:
        return self.model.predict(X)


class PredictorHandler:
    def __init__(
        self,
        train_df: pd.DataFrame,
        label2item: dict[str, str],
        predictor_name: str,
        test_df: Optional[pd.DataFrame] = None,
    ) -> None:
        self.train_df = train_df
        self.test_df = test_df
        self.label2item = label2item
        self.predictor_name = predictor_name
        self.item2predictor: dict[str, Predictor] = dict()
        self.result = defaultdict(lambda: defaultdict(dict))

    def run(self) -> None:
        module_path = PRED_DIR / self.predictor_name / "train.py"
        train = get_object_from_module(module_path, "train")
        target_cols = list(self.label2item.keys())
        X_train = self.train_df.drop(columns=target_cols)

        # 商品ごとにモデルを構築・評価
        for target_col, item in self.label2item.items():
            # 学習
            y_train = self.train_df[[target_col]]
            predictor = train(X=X_train, y=y_train)
            predictor.item = item
            self.item2predictor[item] = predictor

            # 学習データに対する目的変数を予測
            y_pred_train = predictor.predict(X_train)
            plot(
                X=X_train,
                y=y_train,
                y_pred=y_pred_train,
                predictor_name=self.predictor_name,
                target_item=item,
                suffix="train",
            )

            # 学習データを用いたときの平均二乗誤差を出力
            rmse_train = round(np.sqrt(mean_squared_error(y_train, y_pred_train)), 2)
            self.result["rmse"]["train"][item] = rmse_train
            logger.info(f"RMSE for train data [{item}]: {rmse_train}")

            # テストデータに対する精度評価
            if self.test_df is not None:
                X_test = self.test_df.drop(columns=target_cols)
                y_test = self.test_df[[target_col]]

                # テストデータを用いて目的変数を予測
                y_pred_test = predictor.predict(X_test)
                plot(
                    X=X_test,
                    y=y_test,
                    y_pred=y_pred_test,
                    predictor_name=self.predictor_name,
                    target_item=item,
                    suffix="test",
                )

                # テストデータを用いたときの平均二乗誤差を出力
                rmse_test = round(np.sqrt(mean_squared_error(y_test, y_pred_test)), 2)
                self.result["rmse"]["test"][item] = rmse_test
                logger.info(f"RMSE for test data [{item}]: {rmse_test}")
