from __future__ import annotations

import pandas as pd
from interpretableai import iai


def tune_params(X: pd.DataFrame, y: pd.DataFrame, **kwargs) -> dict[str, float]:
    target_col = y.columns[0]
    # ハイパラチューニング
    model = iai.GridSearch(
        iai.OptimalTreeRegressor(
            random_seed=1,
            normalize_y=False,
            normalize_X=False,
            hyperplane_config={"sparsity": "all"},  # with hyperplanes
            # regression_sparsity="all",  # with linear predictions
        ),
        max_depth=range(1, 6),
        # regression_lambda=[0.005, 0.01, 0.05],
    )
    model.fit(X, y[target_col].values)
    return model.get_best_params()
