from __future__ import annotations

import copy
import warnings
from typing import Optional

import pandas as pd
from logzero import logger
from sklearn.neural_network import MLPRegressor

from src.predict.predictor import Predictor

warnings.simplefilter("ignore")


def train(
    X: pd.DataFrame, y: pd.DataFrame, params: Optional[dict[str, float]] = None, **kwargs
) -> Predictor:
    feature_cols = X.columns.tolist()
    logger.info(f"feature_cols: {feature_cols}")
    target_col = y.columns[0]
    if params is None or len(params) == 0:
        params_ = {
            "hidden_layer_sizes": (len(X),),
            # "hidden_layer_sizes": (10, ),
            "max_iter": 30000,
            # "learning_rate_init": 0.001,
            # "alpha": 100
        }
    else:
        params_ = copy.deepcopy(params)
    # 学習
    logger.info("fitting by nn...")
    model = MLPRegressor(random_state=42, verbose=True, **params_)
    model.fit(X, y)
    predictor = Predictor(model=model, feature_cols=feature_cols, target_col=target_col)
    return predictor
