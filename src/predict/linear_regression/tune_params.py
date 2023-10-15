from __future__ import annotations

import numpy as np
import pandas as pd
from logzero import logger
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def tune_params(X: pd.DataFrame, y: pd.DataFrame, **kwargs) -> dict[str, float]:
    target_col = y.columns[0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    rmse_best = 100000
    for alpha in [0, 0.001, 0.01, 0.1]:
        params = {"alpha": alpha}
        # tune hyperparameters
        model = Lasso(**params)
        model.fit(X_train, y_train[target_col].values)
        y_pred = model.predict(X_test)
        rmse = round(np.sqrt(mean_squared_error(y_test, y_pred)), 2)
        logger.info(f"alpha: {alpha}, rmse: {rmse}")
        if rmse < rmse_best:
            rmse_best = rmse
            tune_params = params.copy()
    logger.info(f"tune_params: {tune_params}")
    return tune_params
