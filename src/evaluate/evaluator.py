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
    ) -> None:
        self.test_df = test_df
        self.label2item = label2item
        self.item2predictor = item2predictor
        self.opt_prices = opt_prices
        self.result = dict()
        self.result_item = defaultdict(lambda: defaultdict(dict))

    def run(self) -> None:
        target_cols = list(self.label2item.keys())
        feature_cols = [col for col in self.test_df.columns if col not in target_cols]
        for target_col, item in self.label2item.items():
            X_test = self.test_df[feature_cols]
            y_test = self.test_df[[target_col]]
            predictor = self.item2predictor[item]
            y_pred = predictor.predict(X_test)

            # 売上の実績値
            self.result_item["actual_sales"][item] = round(float(sum(y_test.values)), 1)
            # 実際価格での予測売上
            self.result_item["pred_sales_at_actual_price"][item] = round(float(sum(y_pred)), 1)

        for metric, result_item in self.result_item.items():
            self.result[metric] = float(sum(result_item.values()))
