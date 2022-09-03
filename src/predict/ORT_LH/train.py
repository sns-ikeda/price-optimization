from __future__ import annotations

import copy
from typing import Optional

import pandas as pd
from interpretableai import iai
from logzero import logger

from src.predict.predictor import Predictor
from src.utils.paths import RESULT_DIR


def train(
    X: pd.DataFrame,
    y: pd.DataFrame,
    prefix: Optional[str] = None,
    params: Optional[dict[str, float]] = None,
    **kwargs,
) -> Predictor:
    feature_cols = X.columns.tolist()
    target_col = y.columns[0]
    item = target_col.split("_")[-1]
    if params is None or len(params) == 0:
        params_ = {"max_depth": 2, "cp": 0.001}
    else:
        params_ = copy.deepcopy(params)
    # 学習
    logger.info("fitting by ORT_LH...")
    model = iai.OptimalTreeRegressor(
        random_seed=1,
        normalize_y=False,
        normalize_X=False,
        hyperplane_config={"sparsity": "all"},
        **params_,
        # regression_sparsity="all",
        # regression_lambda=0.01,
    )
    model.fit(X, y[target_col].values)
    if prefix is not None:
        save_path = RESULT_DIR / "realworld" / "predict" / f"ORT_{item}_{prefix}.html"
    else:
        save_path = RESULT_DIR / "realworld" / "predict" / f"ORT_{item}.html"
    model.write_html(str(save_path))
    predictor = Predictor(model=model, feature_cols=feature_cols, target_col=target_col)
    return predictor
