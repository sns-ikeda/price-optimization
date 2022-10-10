from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from src.data_preprocess.preprocessor import get_item_from_label, get_labels_from_items
from src.predict.predictor import Predictor


class Evaluator:
    def __init__(
        self,
        test_df: pd.DataFrame,
        item2predictor: dict[str, Predictor],
        opt_prices: dict[str, float],
        item2prices: dict[str, list[float]],
    ) -> None:
        self.test_df = test_df
        self.item2predictor = item2predictor
        self.opt_prices = opt_prices
        self.avg_prices = {item: np.mean(prices) for item, prices in item2prices.items()}
        self.item2prices = item2prices

        items = list(self.item2predictor.keys())
        self.target_cols = get_labels_from_items(items)
        self.feature_cols = [col for col in self.test_df.columns if col not in self.target_cols]
        self.result = dict()
        self.result_item = dict()
        self.result_df = self.make_result_df()

    def run(self) -> None:
        X_opt = self.make_X(item_prices=self.opt_prices)
        X_avg = self.make_X(item_prices=self.avg_prices)

        X_test = self.test_df[self.feature_cols]
        for target_col in self.target_cols:
            item = get_item_from_label(target_col)
            y_test = self.test_df[target_col]

            # 推論
            predictor = self.item2predictor[item]
            y_pred = predictor.predict(X_test)
            y_pred_opt = predictor.predict(X_opt)
            y_pred_avg = predictor.predict(X_avg)

            # result_dfに格納
            self.result_df = pd.concat(
                [self.result_df, pd.DataFrame(y_pred, columns=[target_col + "_pred"])], axis=1
            )
            self.result_df = pd.concat(
                [self.result_df, pd.DataFrame(y_pred_opt, columns=[target_col + "_pred_opt"])],
                axis=1,
            )
            self.result_df = pd.concat(
                [self.result_df, pd.DataFrame(y_pred_avg, columns=[target_col + "_pred_avg"])],
                axis=1,
            )

            # 各種数値を計算
            price_col = "PRICE" + "_" + item
            actual_price = self.test_df[price_col].values
            opt_price = X_opt[price_col].values
            avg_price = X_avg[price_col].values

            # 売上の実績値
            self.result_item.setdefault("actual_sales", dict())[item] = round(
                float(sum(y_test.values * actual_price)), 1
            )
            # 実際価格での予測売上
            self.result_item.setdefault("pred_sales_at_actual_price", dict())[item] = round(
                float(sum(y_pred.flatten() * actual_price)), 1
            )
            # 平均価格での予測売上
            self.result_item.setdefault("pred_sales_at_average_price", dict())[item] = round(
                float(sum(y_pred_avg.flatten() * avg_price)), 1
            )
            # 最適価格での予測売上
            self.result_item.setdefault("pred_sales_at_opt_price", dict())[item] = round(
                float(sum(y_pred_opt.flatten() * opt_price)), 1
            )
            # 価格候補
            self.result_item.setdefault("price_candidates", dict())[item] = self.item2prices[item]
        # # 理論値
        # if len(self.target_cols) <= 3 and len(list(self.item2prices.values())[0]) <= 9:
        #     theoretical_sales_item, theoretical_opt_prices = self.calc_theoretical_values()
        #     self.result_item["theoretical_sales"] = theoretical_sales_item
        #     self.result_item["theoretical_opt_prices"] = theoretical_opt_prices

        for metric, result_item in self.result_item.items():
            if metric == "theoretical_opt_prices":
                continue
            try:
                self.result[metric] = float(sum(result_item.values()))
            except TypeError:
                pass
        self.result_item["opt_prices"] = self.opt_prices

    def make_X(self, item_prices: dict[str, float]) -> pd.DataFrame:
        X = self.test_df[self.feature_cols].copy()
        for item, price in item_prices.items():
            price_col = "PRICE" + "_" + item
            X[price_col] = price
        return X

    def calc_theoretical_values(self) -> tuple[dict[str, float], dict[str, float]]:
        theoretical_opt_prices = dict()
        theoretical_sales_item = dict()
        theoretical_sales = 0
        for prices in itertools.product(*self.item2prices.values()):
            item_prices = dict(zip(self.item2prices.keys(), prices))
            X = self.make_X(item_prices=item_prices)

            tmp_sales_dict = dict()
            for target_col in self.target_cols:
                item = get_item_from_label(target_col)
                # 推論
                predictor = self.item2predictor[item]
                y_pred = predictor.predict(X)
                price_col = "PRICE" + "_" + item
                price = X[price_col].values
                tmp_sales_dict[item] = round(float(sum(y_pred.flatten() * price)), 1)
            if sum(tmp_sales_dict.values()) > theoretical_sales:
                theoretical_sales_item = tmp_sales_dict
                theoretical_sales = sum(tmp_sales_dict.values())
                theoretical_opt_prices = item_prices
        return theoretical_sales_item, theoretical_opt_prices

    def make_result_df(self) -> pd.DataFrame:
        result_df = self.test_df.copy().reset_index(drop=True)
        # 最適価格のdf
        X_opt = self.make_X(item_prices=self.opt_prices)
        opt_columns = {col: col + "_opt" for col in X_opt.columns}
        X_opt = X_opt.rename(columns=opt_columns).reset_index(drop=True)
        # 平均価格のdf
        X_avg = self.make_X(item_prices=self.avg_prices)
        avg_columns = {col: col + "_avg" for col in X_avg.columns}
        X_avg = X_avg.rename(columns=avg_columns).reset_index(drop=True)
        # concat
        result_df = pd.concat([result_df, X_opt, X_avg], axis=1)
        return result_df
