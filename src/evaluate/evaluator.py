from __future__ import annotations

from collections import defaultdict

import pandas as pd

from src.predict.predictor import Predictor


class Evaluator:
    def __init__(
        self,
        test_df: pd.DataFrame,
        label2item: dict[str, str],
        item2predictor: dict[str, Predictor],
        opt_prices: dict[str, float],
        avg_prices: dict[str, float],
    ) -> None:
        self.test_df = test_df
        self.label2item = label2item
        self.item2predictor = item2predictor
        self.opt_prices = opt_prices
        self.avg_prices = avg_prices

        self.target_cols = list(self.label2item.keys())
        self.feature_cols = [col for col in self.test_df.columns if col not in self.target_cols]
        self.result = dict()
        self.result_item = defaultdict(lambda: defaultdict(dict))

    def run(self) -> None:
        X_opt = self.make_X(item_prices=self.opt_prices)
        X_avg = self.make_X(item_prices=self.avg_prices)

        X_test = self.test_df[self.feature_cols]
        for target_col, item in self.label2item.items():
            y_test = self.test_df[target_col]
            # 推論
            predictor = self.item2predictor[item]
            y_pred = predictor.predict(X_test)
            y_pred_opt = predictor.predict(X_opt)
            y_pred_avg = predictor.predict(X_avg)

            price_col = "PRICE" + "_" + item
            actual_price = self.test_df[price_col].values
            opt_price = X_opt[price_col].values
            avg_price = X_avg[price_col].values

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

    def make_X(self, item_prices: dict[str, float]) -> pd.DataFrame:
        X = self.test_df[self.feature_cols].copy()
        for item, opt_price in item_prices.items():
            price_col = "PRICE" + "_" + item
            X[price_col] = opt_price
        return X
