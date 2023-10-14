from __future__ import annotations

import pickle
from typing import Any, Optional, TypeVar

import numpy as np
import pandas as pd
from logzero import logger
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score

from src.data_preprocess.preprocessor import get_item_from_label
from src.predict.plot import plot
from src.utils.module_handler import get_object_from_module
from src.utils.paths import MODEL_DIR, PRED_DIR, RESULT_DIR

PredictorClass = TypeVar("PredictorClass")


class Predictor:
    def __init__(self, model: PredictorClass, feature_cols: list[str], target_col: str) -> None:
        self.model = model
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.item = get_item_from_label(self.target_col)

    def predict(self, X: pd.DataFrame) -> np.array:
        return np.clip(self.model.predict(X), 0, None)


class PredictorMaker:
    def __init__(
        self,
        predictor_name: str,
        train_df: pd.DataFrame,
        target_col: str,
        test_df: Optional[pd.DataFrame] = None,
        params: Optional[dict[str, Any]] = None,
        plot: bool = True,
        data_type: str = "realworld",
    ) -> None:
        self.predictor_name = predictor_name
        self.train_df = train_df
        self.test_df = test_df
        self.plot = plot
        self.data_type = data_type
        self.target_col = target_col
        self.item = get_item_from_label(target_col)
        self.params = params
        self.result: dict[str, dict[str, float]] = dict()

    def run(
        self, train_or_test: str, suffix: Optional[str] = None, tune: bool = False
    ) -> Predictor:
        # 説明変数と目的変数を分離
        X_train = self.train_df.drop(columns=[self.target_col])
        y_train = self.train_df[[self.target_col]]
        if self.test_df is not None:
            X_test = self.test_df.drop(columns=[self.target_col])
            y_test = self.test_df[[self.target_col]]

        train_suffix = f"{train_or_test}_train"
        test_suffix = f"{train_or_test}_test"
        if suffix is not None:
            train_suffix = train_suffix + f"_{suffix}"
            test_suffix = test_suffix + f"_{suffix}"

        # 使用するモデルを取得
        module_path = PRED_DIR / self.predictor_name / "train.py"
        train = get_object_from_module(module_path, "train")

        # ハイパラチューニング
        if tune:
            logger.info(f"ハイパラチューニングを実行")
            self.params = self.tune_params(X_train, y_train)

        # モデルを構築・評価
        logger.info(f"item: {self.item}")
        predictor = train(X=X_train, y=y_train, suffix=train_suffix, params=self.params)

        # 訓練データ&検証データに対する目的変数を予測
        y_pred_train = predictor.predict(X_train)
        self.evaluate(y=y_train, y_pred=y_pred_train, split_type="train")
        if self.plot:
            plot(
                y=y_train,
                y_pred=y_pred_train,
                predictor_name=self.predictor_name,
                target_item=self.item,
                suffix=train_suffix,
                dir_path=RESULT_DIR / self.data_type / "predict",
            )
        if self.test_df is not None:
            y_pred_test = predictor.predict(X_test)
            self.evaluate(y=y_test, y_pred=y_pred_test, split_type="test")
            if self.plot:
                plot(
                    y=y_test,
                    y_pred=y_pred_test,
                    predictor_name=self.predictor_name,
                    target_item=self.item,
                    suffix=test_suffix,
                    dir_path=RESULT_DIR / self.data_type / "predict",
                )
        # モデルを保存
        save_path = MODEL_DIR / self.predictor_name / train_or_test / f"{self.item}_{suffix}.pickle"
        with open(save_path, "wb") as f:
            pickle.dump(predictor, f)
        return predictor

    def evaluate(self, y: np.array, y_pred: np.array, split_type: str) -> None:
        # 二乗平均平方根誤差
        rmse = round(np.sqrt(mean_squared_error(y, y_pred)), 2)
        self.result.setdefault("rmse", dict())[split_type] = rmse
        logger.info(f"RMSE for {split_type} data [{self.item}]: {rmse}")

        # 平均絶対パーセント誤差
        mape = mean_absolute_percentage_error(y, y_pred)
        self.result.setdefault("mape", dict())[split_type] = mape
        logger.info(f"MAPE for {split_type} data [{self.item}]: {mape}")

        # 決定係数
        r2 = round(r2_score(y, y_pred), 2)
        self.result.setdefault("r2", dict())[split_type] = r2
        logger.info(f"R^2 for {split_type} data [{self.item}]: {r2}")

    def tune_params(self, X, y) -> dict[str, Any]:
        module_path = PRED_DIR / self.predictor_name / "tune_params.py"
        tune_params = get_object_from_module(module_path, "tune_params")
        params = tune_params(X, y)
        return params
