import pandas as pd
from sklearn.linear_model import LinearRegression

from src.predict.predictor import Predictor


def train(X: pd.DataFrame, y: pd.DataFrame, **kwargs) -> Predictor:
    feature_cols = X.columns.tolist()
    target_col = y.columns[0]

    # 学習
    model = LinearRegression()
    model.fit(X, y)
    predictor = Predictor(model=model, feature_cols=feature_cols, target_col=target_col)
    return predictor
