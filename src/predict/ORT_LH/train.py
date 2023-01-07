from __future__ import annotations

import os
from typing import Optional

os.environ["JULIA_NUM_THREADS"] = "8"

import pandas as pd
from interpretableai import iai
from logzero import logger

from src.predict.predictor import Predictor
from src.utils.paths import RESULT_DIR


def train(
    X: pd.DataFrame,
    y: pd.DataFrame,
    suffix: Optional[str] = None,
    params: Optional[dict[str, float]] = None,
    **kwargs,
) -> Predictor:
    feature_cols = X.columns.tolist()
    logger.info(f"feature_cols: {feature_cols}")
    target_col = y.columns[0]
    item = target_col.split("_")[-1]
    if params is None or len(params) == 0:
        logger.info("parameters are not set")
        params_ = {"max_depth": 2, "regression_lambda": 0}
    else:
        logger.info(f"params: {params}")
        params_ = params.copy()
    if suffix is not None:
        params_ = {"max_depth": 2, "regression_lambda": 0} if "test_train" in suffix else params_
    # 学習
    logger.info("fitting by ORT_LH...")
    model = iai.OptimalTreeRegressor(
        random_seed=1,
        normalize_y=False,
        normalize_X=False,
        # hyperplane_config={"sparsity": "all"},
        regression_features="all",
        regression_weighted_betas=True,
        regression_sparsity="all",
        # regression_lambda=0,
        cp=0,
        # ls_num_tree_restarts=1000,
        # ls_num_hyper_restarts=100,
        # num_threads=8,
        **params_,
    )
    print(model.get_params())
    model.fit(X, y[target_col].values)
    if suffix is not None:
        save_path = RESULT_DIR / "synthetic" / "predict" / f"ORT_{item}_{suffix}.html"
    else:
        save_path = RESULT_DIR / "synthetic" / "predict" / f"ORT_{item}.html"
    model.write_html(str(save_path))
    predictor = Predictor(model=model, feature_cols=feature_cols, target_col=target_col)
    return predictor
