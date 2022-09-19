from __future__ import annotations

from typing import Optional, TypeVar

import numpy as np
import pandas as pd
from logzero import logger

from src.utils.module_handler import get_object_from_module
from src.utils.paths import DATA_DIR, DATA_PRE_DIR

ScalerClass = TypeVar("ScalerClass")


class DataPreprocessor:
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.raw_df: pd.DataFrame = DataPreprocessor.load_data(self.dataset_name)
        self.processed_df: Optional[pd.DataFrame] = None
        self.item2df: dict[str, pd.DataFrame] = dict()

    def run(self) -> None:
        self.preprocess()
        dir_path = DATA_DIR / self.dataset_name / "processed"
        self.processed_df.to_csv(dir_path / "processed.csv", index=False)
        for item, df in self.item2df.items():
            file_name = f"processed_{item}.csv"
            df.to_csv(dir_path / file_name, index=False)

    @staticmethod
    def load_data(dataset_name: str) -> pd.DataFrame:
        module_path = DATA_PRE_DIR / dataset_name / "load_data.py"
        load_data = get_object_from_module(module_path, "load_data")
        df = load_data()
        return df

    def preprocess(self) -> None:
        logger.info(f"dataset: {self.dataset_name}")
        logger.info(f"# of rows [raw data]: {len(self.raw_df)}")

        # 大元となるデータの前処理
        module_path = DATA_PRE_DIR / self.dataset_name / "preprocess.py"
        preprocess = get_object_from_module(module_path, "preprocess")
        self.processed_df = preprocess(self.raw_df)
        logger.info(f"# of rows [processed data]: {len(self.processed_df)}")

        # itemごとのデータに分割
        target_cols = self.get_target_cols()
        base_feature_cols = self.get_feature_cols(target_cols)
        for target_col in target_cols:
            cols = base_feature_cols + [target_col]
            df = self.processed_df[cols]
            item = get_item_from_label(target_col)
            self.item2df[item] = df

    def get_target_cols(self, prefix: str = "UNITS") -> list[str]:
        if self.processed_df is not None:
            target_cols = [col for col in self.processed_df.columns if prefix in col]
        else:
            raise Exception("Run preprocess before executing this method")
        return target_cols

    def get_feature_cols(self, target_cols: list[str]) -> list[str]:
        if self.processed_df is not None:
            feature_cols = [col for col in self.processed_df.columns if col not in target_cols]
        else:
            raise Exception("Run preprocess before executing this method")
        return feature_cols


def get_item_from_label(target_col: str) -> str:
    item = target_col.split("_")[-1]
    return item


def get_items_from_labels(target_cols: list[str]) -> list[str]:
    items = [get_item_from_label(target_col) for target_col in target_cols]
    return items


def get_label_from_item(item: str, prefix: str = "UNITS") -> str:
    label = prefix + "_" + item
    return label


def get_labels_from_items(items: list[str], prefix: str = "UNITS") -> list[str]:
    labels = [get_label_from_item(item, prefix=prefix) for item in items]
    return labels


def get_label2item(target_cols: list[str]) -> dict[str, str]:
    label2item = dict()
    for target_col in target_cols:
        item = get_item_from_label(target_col)
        label2item[target_col] = item
    return label2item


def get_item2avg_prices(
    df: pd.DataFrame, items: list[str], prefix: str = "PRICE"
) -> dict[str, float]:
    avg_prices = dict()
    for item in items:
        price_col = prefix + "_" + item
        avg_prices[item] = df[price_col].mean()
    return avg_prices


def get_item2prices(
    df: pd.DataFrame,
    num_of_prices: int,
    items: list[str],
    prefix: str = "PRICE",
    interval: float = 0.05,
) -> dict[str, list[float]]:
    item2prices = dict()
    for item in items:
        price_col = prefix + "_" + item
        price_mean = df[price_col].mean()
        price_min = (1 - (num_of_prices // 2) * interval) * price_mean
        price_max = (1 + (num_of_prices // 2) * interval) * price_mean
        prices = list(np.round(np.linspace(price_min, price_max, num_of_prices), 2))
        item2prices[item] = prices
    return item2prices


if __name__ == "__main__":
    from src.configs import read_config

    config = read_config("config.yaml")["realworld"]
    dp = DataPreprocessor(config["dataset_name"])
    dp.run()
