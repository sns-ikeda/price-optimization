from __future__ import annotations

from collections import defaultdict
from typing import TypeVar

import pandas as pd

from src.predict.predictor import Predictor

ScalerClass = TypeVar("ScalerClass")


class Evaluator:
    def __init__(
        self,
        scaled_test_df: pd.DataFrame,
        original_test_df: pd.DataFrame,
        label2item: dict[str, str],
        item2predictor: dict[str, Predictor],
        scaled_opt_prices: dict[str, float],
        scaled_avg_prices: dict[str, float],
        scaler: ScalerClass,
    ) -> None:
        self.scaled_test_df = scaled_test_df
        self.original_test_df = original_test_df
        self.label2item = label2item
        self.item2predictor = item2predictor
        self.scaled_opt_prices = scaled_opt_prices
        self.scaled_avg_prices = scaled_avg_prices

        self.target_cols = list(self.label2item.keys())
        self.feature_cols = [
            col for col in self.scaled_test_df.columns if col not in self.target_cols
        ]
        self.scaler = scaler
        self.result = dict()
        self.result_item = defaultdict(lambda: defaultdict(dict))

    def run(self) -> None:
        X_opt = self.make_X_opt()
        X_avg = self.make_X_avg()
        if self.scaler is None:
            X_opt_inv = X_opt.copy()
            X_avg_inv = X_avg.copy()
        else:
            X_opt_inv = pd.DataFrame(
                self.scaler.inverse_transform(X_opt), columns=self.feature_cols
            )
            X_avg_inv = pd.DataFrame(
                self.scaler.inverse_transform(X_avg), columns=self.feature_cols
            )
        for target_col, item in self.label2item.items():
            # 推論
            X_test = self.scaled_test_df[self.feature_cols]
            y_test = self.scaled_test_df[target_col]
            predictor = self.item2predictor[item]
            y_pred = predictor.predict(X_test)
            y_pred_opt = predictor.predict(X_opt)
            y_pred_avg = predictor.predict(X_avg)

            price_col = "PRICE" + "_" + item
            actual_price = self.original_test_df[price_col].values
            opt_price = X_opt_inv[price_col].values
            avg_price = X_avg_inv[price_col].values

            # 売上の実績値
            self.result_item["actual_sales"][item] = round(
                float(sum(y_test.values * actual_price)), 1
            )
            # 実際価格での予測売上
            self.result_item["pred_sales_at_actual_price"][item] = round(
                float(sum(y_pred.flatten() * actual_price)), 1
            )
            # 平均価格での予測売上
            self.result_item["pred_sales_at_average_price"][item] = round(
                float(sum(y_pred_avg.flatten() * avg_price)), 1
            )
            # 最適価格での予測売上
            self.result_item["pred_sales_at_opt_price"][item] = round(
                float(sum(y_pred_opt.flatten() * opt_price)), 1
            )
        for metric, result_item in self.result_item.items():
            self.result[metric] = float(sum(result_item.values()))

    def make_X_opt(self) -> pd.DataFrame:
        X_opt = self.scaled_test_df[self.feature_cols].copy()
        for item, scaled_opt_price in self.scaled_opt_prices.items():
            price_col = "PRICE" + "_" + item
            X_opt[price_col] = scaled_opt_price
        return X_opt

    def make_X_avg(self) -> pd.DataFrame:
        X_avg = self.scaled_test_df[self.feature_cols].copy()
        for item, price in self.scaled_avg_prices.items():
            price_col = "PRICE" + "_" + item
            X_avg[price_col] = price
        return X_avg
