from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.handle_module import get_object_from_module
from src.utils.paths import PRED_DIR


class Predictor:
    def __init__(self, model, feature_cols: list[str], target_col: str) -> None:
        self.model = model
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.item = target_col.split("_")[1]

    def predict(self, X: pd.DataFrame) -> np.array:
        return self.model.predict(X)


class MakePredictor:
    def __init__(self, train_df: pd.DataFrame, target_cols: list[str], model_name: str) -> None:
        self.train_df = train_df
        self.target_cols = target_cols
        self.model_name = model_name
        self.items = [col.split("_")[1] for col in self.target_cols]
        self.item2predictor: dict[str, Predictor] = dict()

    def run(self) -> None:
        module_path = PRED_DIR / self.model_name / "train.py"
        train = get_object_from_module(module_path, "train")

        df = self.train_df.copy()
        for target_col in self.target_cols:
            y = df[target_col]
            X = df.drop(self.target_cols)
            predictor = train(X=X, y=y)
            self.item2predictor[predictor.item] = predictor
