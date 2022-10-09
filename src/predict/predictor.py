from __future__ import annotations

import pickle
from typing import Optional, TypeVar

import numpy as np
import pandas as pd
from logzero import logger
from sklearn.metrics import mean_squared_error, r2_score

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
        params: Optional[dict[str, dict[str, float]]] = None,
    ) -> None:
        self.predictor_name = predictor_name
        self.train_df = train_df
        self.test_df = test_df
        self.target_col = target_col
        self.item = get_item_from_label(target_col)
        self.params = params
        self.result: dict[str, dict[str, float]] = dict()

    def run(self, train_or_test: str, suffix: Optional[str] = None) -> Predictor:
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

        # モデルを構築・評価
        logger.info(f"item: {self.item}")
        if self.params is None:
            params = None
        else:
            params = self.params[item]
        predictor = train(X=X_train, y=y_train, suffix=train_suffix, params=params)

        # 訓練データ&検証データに対する目的変数を予測
        y_pred_train = predictor.predict(X_train)
        self.evaluate(y=y_train, y_pred=y_pred_train, split_type="train")
        plot(
            y=y_train,
            y_pred=y_pred_train,
            predictor_name=self.predictor_name,
            target_item=self.item,
            suffix=train_suffix,
            dir_path=RESULT_DIR / "realworld" / "predict",
        )
        if self.test_df is not None:
            y_pred_test = predictor.predict(X_test)
            self.evaluate(y=y_test, y_pred=y_pred_test, split_type="test")
            plot(
                y=y_test,
                y_pred=y_pred_test,
                predictor_name=self.predictor_name,
                target_item=self.item,
                suffix=test_suffix,
                dir_path=RESULT_DIR / "realworld" / "predict",
            )
        # モデルを保存
        save_path = MODEL_DIR / self.predictor_name / train_or_test / f"{self.item}_{suffix}.pickle"
        with open(save_path, "wb") as f:
            pickle.dump(predictor, f)
        return predictor

    def evaluate(self, y: np.array, y_pred: np.array, split_type: str) -> None:
        # 二乗平均平方根誤差
        rmse = round(np.sqrt(mean_squared_error(y, y_pred)), 1)
        self.result.setdefault("rmse", dict())[split_type] = rmse
        logger.info(f"RMSE for {split_type} data [{self.item}]: {rmse}")
        # 決定係数
        r2 = round(r2_score(y, y_pred), 2)
        self.result.setdefault("r2", dict())[split_type] = r2
        logger.info(f"R^2 for {split_type} data [{self.item}]: {r2}")


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    from src.configs import read_config
    from src.data_preprocess.preprocessor import DataPreprocessor, get_label_from_item
    from src.simulator import postproceess_pred_result
    from src.utils.dict_converter import dict2json
    from src.utils.paths import RESULT_DIR

    config = read_config("config.yaml")["realworld"]

    dp = DataPreprocessor(config["dataset_name"])
    dp.run()

    _pred_result_train, _pred_result_test = dict(), dict()
    for predictor_name in config["predictor_names"]:
        for item, df in dp.item2df.items():
            for train_size in config["train_sizes"]:
                test_size = round(1 - train_size, 3)
                train_df, test_df = train_test_split(
                    df, train_size=train_size, test_size=test_size, shuffle=False
                )
                target_col = get_label_from_item(item=item)

                # 訓練データに対する予測モデルを構築
                pm_train = PredictorMaker(
                    predictor_name=predictor_name,
                    train_df=train_df,
                    test_df=test_df,
                    target_col=target_col,
                )
                pm_train.run(train_or_test="train", suffix=str(train_size))

                # 学習データへの学習結果を格納
                _pred_result_train[item] = pm_train.result
                # pred_result_train.setdefault(train_size, dict())[
                #     predictor_name
                # ] = pm_train.result

                # # 検証データに対する予測モデルを構築
                # pm_test = PredictorMaker(
                #     predictor_name=predictor_name,
                #     train_df=test_df,
                #     target_col=target_col,
                # )
                # pm_test.run(train_or_test="test", suffix=str(train_size))

                # # テストデータへの学習結果を格納
                # pred_result_test.setdefault(train_size, dict())[
                #     predictor_name
                # ] = pm_test.result

    pred_result_train = postproceess_pred_result(_pred_result_train)
    # json形式で結果を出力
    dict2json(
        target_dict=pred_result_train,
        save_path=RESULT_DIR / "realworld" / "predict" / "pred_result_train.json",
    )
    # dict2json(
    #     target_dict=pred_result_test,
    #     save_path=RESULT_DIR / "realworld" / "predict" / "pred_result_test.json",
    # )
