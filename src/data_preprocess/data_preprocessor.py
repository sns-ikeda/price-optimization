from __future__ import annotations

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

    def run(self, **kwargs) -> None:
        module_path = DATA_PRE_DIR / self.dataset / "preprocess.py"
        preprocess = get_object_from_module(module_path, "preprocess")
        self.processed_df = preprocess(self.row_df, **kwargs)
