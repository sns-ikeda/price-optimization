import pandas as pd
from interpretableai import iai

from src.predict.predictor import Predictor
from src.utils.paths import RESULT_DIR


def train(X: pd.DataFrame, y: pd.DataFrame, prefix: str, **kwargs) -> Predictor:
    feature_cols = X.columns.tolist()
    target_col = y.columns[0]
    item = target_col.split("_")[-1]
    # 学習
    # model = iai.GridSearch(
    #     iai.OptimalTreeRegressor(
    #         random_seed=1,
    #         normalize_y=False,
    #         hyperplane_config={'sparsity': 'all'},
    #         regression_sparsity="all",
    #     ),
    #     max_depth=range(1, 6),
    #     regression_lambda=[0.005, 0.01, 0.05],
    # )
    model = iai.OptimalTreeRegressor(
        random_seed=1,
        normalize_y=False,
        normalize_X=False,
        hyperplane_config={"sparsity": "all"},
        max_depth=3,
        cp=0.0012145072743561601,
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
