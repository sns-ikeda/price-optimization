from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from interpretableai import iai
from logzero import logger
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def tune_params(X: pd.DataFrame, y: pd.DataFrame, **kwargs) -> dict[str, Any]:
    target_col = y.columns[0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    rmse_best = 100000
    for max_depth in [0, 1, 2]:
        params = {"max_depth": max_depth}
        # ハイパラチューニング
        model = iai.OptimalTreeRegressor(
            random_seed=1,
            normalize_y=False,
            normalize_X=False,
            hyperplane_config={"sparsity": "all"},
            regression_features="all",
            regression_weighted_betas=True,
            regression_sparsity="all",
            regression_lambda=0,
            cp=0,
            **params,
        )
        model.fit(X_train, y_train[target_col].values)
        y_pred = model.predict(X_test)
        rmse = round(np.sqrt(mean_squared_error(y_test, y_pred)), 2)
        logger.info(f"max_depth: {max_depth}, rmse: {rmse}")
        if rmse < rmse_best:
            rmse_best = rmse
            tune_params = params.copy()
    logger.info(f"tune_params: {tune_params}")
    return tune_params
