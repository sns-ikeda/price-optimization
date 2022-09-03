from __future__ import annotations

import copy
from typing import Optional

import pandas as pd
from logzero import logger
from sklearn.linear_model import Lasso

from src.predict.predictor import Predictor


def train(
    X: pd.DataFrame, y: pd.DataFrame, params: Optional[dict[str, float]] = None, **kwargs
) -> Predictor:
    feature_cols = X.columns.tolist()
    target_col = y.columns[0]
    if params is None or len(params) == 0:
        params_ = {"alpha": 0}
    else:
        params_ = copy.deepcopy(params)
    # 学習
    logger.info("fitting by lasso...")
    model = Lasso(**params_)
    model.fit(X, y)
    predictor = Predictor(model=model, feature_cols=feature_cols, target_col=target_col)
    logger.info(f"coefficients: {model.coef_}")
    return predictor
