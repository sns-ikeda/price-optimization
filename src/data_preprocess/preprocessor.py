from __future__ import annotations

import numpy as np
import pandas as pd
from logzero import logger

from src.utils.handle_module import get_object_from_module
from src.utils.paths import DATA_PRE_DIR


class DataPreprocessor:
    def __init__(self, dataset: str):
        self.dataset = dataset
        self.raw_df = DataPreprocessor.load_data(self.dataset)
        self.item2index = dict()
        self.index2item = dict()
        self.processed_df = pd.DataFrame()
        self.preprocessed = False

    @staticmethod
    def load_data(dataset: str) -> pd.DataFrame:
        module_path = DATA_PRE_DIR / dataset / "load_data.py"
        load_data = get_object_from_module(module_path, "load_data")
        df = load_data()
        return df

    def preprocess(self, **kwargs) -> pd.DataFrame:
        module_path = DATA_PRE_DIR / self.dataset / "preprocess.py"
        preprocess = get_object_from_module(module_path, "preprocess")
        logger.info(f"# of rows [raw data]: {len(self.raw_df)}")
        self.processed_df = preprocess(self.raw_df, **kwargs)
        logger.info(f"# of rows [processed data]: {len(self.processed_df)}")
        self.preprocessed = True
        return self.processed_df

    @staticmethod
    def get_item2prices(
        df: pd.DataFrame, num_of_prices: int, items: list[str], prefix: str = "PRICE"
    ) -> dict[str, list[float]]:
        item2prices = dict()
        for item in items:
            price_col = prefix + "_" + item
            price_max = df[price_col].max()
            price_min = df[price_col].min()
            item2prices[item] = list(np.linspace(price_min, price_max, num_of_prices))
        return item2prices

    def get_target_cols(self, prefix: str = "UNITS") -> list[str]:
        if self.preprocessed:
            target_cols = [col for col in self.processed_df.columns if prefix in col]
        else:
            raise Exception("Run preprocess before executing this method")
        return target_cols

    def get_feature_cols(self, target_cols: list[str]) -> list[str]:
        if self.preprocessed:
            feature_cols = [col for col in self.processed_df.columns if col not in target_cols]
        else:
            raise Exception("Run preprocess before executing this method")
        return feature_cols

    @staticmethod
    def get_label2item(target_cols: list[str]) -> dict[str, str]:
        label2item = dict()
        for col in target_cols:
            item = col.split("_")[-1]
            label2item[col] = item
        return label2item


def select_scaler(scaling_type: str = "standard"):
    from sklearn import preprocessing

    if scaling_type == "standard":
        return preprocessing.StandardScaler()
    elif scaling_type == "minmax":
        return preprocessing.MinMaxScaler()
