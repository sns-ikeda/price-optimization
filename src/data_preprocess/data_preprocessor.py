from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.handle_module import get_object_from_module
from src.utils.paths import DATA_PRE_DIR


class DataPreprocessor:
    def __init__(self, dataset: str):
        self.dataset = dataset
        self.row_df = DataPreprocessor.load_data(self.dataset)
        self.item2index = dict()
        self.index2item = dict()
        self.processed_df = pd.DataFrame()

    @staticmethod
    def load_data(dataset: str) -> pd.DataFrame:
        module_path = DATA_PRE_DIR / dataset / "load_data.py"
        load_data = get_object_from_module(module_path, "load_data")
        df = load_data()
        return df

    def preprocess(self, **kwargs) -> pd.DataFrame:
        module_path = DATA_PRE_DIR / self.dataset / "preprocess.py"
        preprocess = get_object_from_module(module_path, "preprocess")
        self.processed_df = preprocess(self.row_df, **kwargs)
        return self.processed_df

    @staticmethod
    def get_item2prices(
        df: pd.DataFrame, num_of_prices: int, items: list[str], prefix: str = "PRICE"
    ) -> dict[str, list[float]]:
        item2prices = dict()
        for item in items:
            price_col = prefix + item
            price_max = df[price_col].max()
            price_min = df[price_col].min()
            item2prices[item] = list(np.linspace(price_min, price_max, num_of_prices))
        return item2prices

    def get_target_cols(self, prefix: str = "UNITS") -> list[str]:
        target_cols = [col for col in self.row_df.columns if prefix in col]
        return target_cols
