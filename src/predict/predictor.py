from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd
from logzero import logger
from sklearn.metrics import mean_squared_error, r2_score

from src.predict.plot import plot
from src.utils.handle_module import get_object_from_module
from src.utils.paths import PRED_DIR, RESULT_DIR


class Predictor:
    def __init__(self, model, feature_cols: list[str], target_col: str) -> None:
        self.model = model
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.item = None

    def predict(self, X: pd.DataFrame) -> np.array:
        return np.clip(self.model.predict(X), 0, None)


class PredictorHandler:
    def __init__(
        self,
        train_df: pd.DataFrame,
        label2item: dict[str, str],
        predictor_name: str,
        test_df: Optional[pd.DataFrame] = None,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
        params: Optional[dict[str, float]] = None,
    ) -> None:
        self.train_df = train_df
        self.test_df = test_df
        self.label2item = label2item
        self.predictor_name = predictor_name
        self.prefix = prefix
        self.suffix = suffix
        self.params = params
        self.item2predictor: dict[str, Predictor] = dict()
        self.result = defaultdict(lambda: defaultdict(dict))

    def run(self, item2features: Optional[dict[str, list[str]]] = None) -> None:
        module_path = PRED_DIR / self.predictor_name / "train.py"
        train = get_object_from_module(module_path, "train")
        target_cols = list(self.label2item.keys())
        items = list(self.label2item.values())

        X_train_base = self.train_df.drop(columns=target_cols)
        if self.test_df is not None:
            X_test_base = self.test_df.drop(columns=target_cols)
        else:
            X_test_base = None
        if item2features is not None:
            X_trains = {item: X_train_base[item2features[item]] for item in items}
            try:
                X_tests = {item: X_test_base[item2features[item]] for item in items}
            except TypeError:
                X_tests = {item: X_test_base for item in items}
        else:
            X_trains = {item: X_train_base for item in items}
            X_tests = {item: X_test_base for item in items}

        # 商品ごとにモデルを構築・評価
        for target_col, item in self.label2item.items():
            if self.params is None:
                params = None
            else:
                params = self.params[item]
            # 学習
            X_train = X_trains[item]
            y_train = self.train_df[[target_col]]
            predictor = train(X=X_train, y=y_train, prefix=self.prefix, params=params)
            predictor.item = item
            self.item2predictor[item] = predictor

            # 学習データに対する目的変数を予測
            y_pred_train = predictor.predict(X_train)
            if self.prefix is not None:
                train_suffix = f"{self.prefix}_train"
                test_suffix = f"{self.prefix}_test"
            else:
                train_suffix = "train"
                test_suffix = "test"
            if self.suffix is not None:
                train_suffix = train_suffix + f"_{self.suffix}"
                test_suffix = test_suffix + f"_{self.suffix}"
            plot(
                y=y_train,
                y_pred=y_pred_train,
                predictor_name=self.predictor_name,
                target_item=item,
                suffix=train_suffix,
                dir_path=RESULT_DIR / "realworld" / "predict",
            )
            # 学習データを用いたときの評価
            self.evaluate(y=y_train, y_pred=y_pred_train, split_type="train", item=item)

            # テストデータに対する精度評価
            if self.test_df is not None:
                X_test = X_tests[item]
                y_test = self.test_df[[target_col]]

                # テストデータを用いて目的変数を予測
                y_pred_test = predictor.predict(X_test)
                plot(
                    y=y_test,
                    y_pred=y_pred_test,
                    predictor_name=self.predictor_name,
                    target_item=item,
                    suffix=test_suffix,
                    dir_path=RESULT_DIR / "realworld" / "predict",
                )
                # テストデータを用いたときの評価
                self.evaluate(y=y_test, y_pred=y_pred_test, split_type="test", item=item)
        # 結果の後処理
        self.postprocess_result()

    def evaluate(self, y: np.array, y_pred: np.array, split_type: str, item: str) -> None:
        # 二乗平均平方根誤差
        rmse = round(np.sqrt(mean_squared_error(y, y_pred)), 1)
        self.result["rmse"][split_type][item] = rmse
        logger.info(f"RMSE for {split_type} data [{item}]: {rmse}")
        # 決定係数
        r2 = round(r2_score(y, y_pred), 2)
        self.result["r2"][split_type][item] = r2
        logger.info(f"R^2 for {split_type} data [{item}]: {r2}")

    def postprocess_result(self) -> None:
        metrics = ["rmse", "r2"]
        split_types = ["train"]
        if self.test_df is not None:
            split_types.append("test")
        for metric in metrics:
            for split_type in split_types:
                results_dict = self.result[metric][split_type]
                mean = round(np.mean(list(results_dict.values())), 2)
                self.result[metric][split_type]["mean"] = mean
