from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TypeVar

from src.optimize.algorithms import ALGORITHMS
from src.optimize.params import SyntheticDataParameter, RealDataParameter
from src.utils.module_handler import get_object_from_module
from src.utils.paths import OPT_MODEL_DIR

IndexSet = TypeVar("IndexSet")
Constant = TypeVar("Constant")
Model = TypeVar("Model")


@dataclass
class ModelInput:
    model_name: str
    index_set: IndexSet
    constant: Constant


class Optimizer:
    def __init__(
        self,
        model_name: str,
        algo_name: str,
        data_param: ArtificialDataParameter | RealDataParameter,
        x: Optional[dict[str, int]] = None,
    ):
        self.model_name = model_name
        self.algo_name = algo_name
        self.data_param = data_param
        self.x = x
        self.algorithm = None
        self.result = None

    @staticmethod
    def make_model_input(
        model_name: str, data_param: SyntheticDataParameter | RealDataParameter
    ) -> ModelInput:
        """モデルの入力データを作成"""
        module_path = OPT_MODEL_DIR / model_name / "make_input.py"
        make_input = get_object_from_module(module_path, f"make_{data_param.data_type}_input")
        index_set, constant = make_input(params=data_param)
        model_input = ModelInput(model_name=model_name, index_set=index_set, constant=constant)
        return model_input

    @staticmethod
    def make_model(model_input: ModelInput, x: Optional[dict[str, int]]) -> Model:
        """最適化モデルを構築"""
        module_path = OPT_MODEL_DIR / model_input.model_name / "model.py"
        model_class = get_object_from_module(module_path, "Model")
        model = model_class(index_set=model_input.index_set, constant=model_input.constant, x=x)
        return model

    def run(self, **kwargs) -> None:
        model_input = Optimizer.make_model_input(
            model_name=self.model_name, data_param=self.data_param
        )
        model = Optimizer.make_model(model_input, self.x)
        algo_class = ALGORITHMS.get(self.algo_name, None)
        if algo_class is not None:
            self.algorithm = algo_class(model=model, **kwargs)
            self.algorithm.run()
            self.algorithm.result.data_param = self.data_param
            self.result = self.algorithm.result


if __name__ == "__main__":
    import pickle

    from logzero import logger
    from sklearn.model_selection import train_test_split

    from src.configs import ALGO_CONFIG as algo_config
    from src.configs import read_config
    from src.data_preprocess.preprocessor import DataPreprocessor, get_item2prices
    from src.optimize.params import calc_g
    from src.optimize.predictor2model import predictor2model
    from src.simulator import calc_actual_sales
    from src.utils.paths import MODEL_DIR

    config = read_config("config.yaml")["realworld"]
    dp = DataPreprocessor(config["dataset_name"])
    dp.run()
    items = list(dp.item2df.keys())

    for predictor_name in config["predictor_names"]:
        for train_size in config["train_sizes"]:
            # 商品ごとの予測モデルを取得
            item2predictor = dict()
            for item in items:
                load_path = MODEL_DIR / predictor_name / "train" / f"{item}_{train_size}.pickle"
                with open(load_path, "rb") as f:
                    predictor = pickle.load(f)
                item2predictor[item] = predictor

            # 訓練データと検証データに分割
            test_size = round(1 - train_size, 3)
            train_df, test_df = train_test_split(
                dp.processed_df, train_size=train_size, test_size=test_size, shuffle=False
            )
            # 最適化モデルの入力データを作成
            item2prices = get_item2prices(
                df=train_df,
                num_of_prices=config["num_of_prices"],
                items=items,
            )
            logger.info(f"price candidates: {item2prices}")
            data_param = RealDataParameter(
                num_of_prices=config["num_of_prices"],
                item2predictor=item2predictor,
                item2prices=item2prices,
                g=calc_g(
                    df=test_df,
                    item2predictor=item2predictor,
                ),
            )
            # 価格最適化しない場合の結果
            actual_sales_item = calc_actual_sales(train_df.tail(1), items)
            actual_total_sales = sum(actual_sales_item.values())
            logger.info(f"actual_total_sales: {actual_total_sales}")

            # 価格最適化を実行
            for algo_name in config["algo_names"]:
                model_name = predictor2model[predictor_name]
                optimizer = Optimizer(
                    model_name=model_name, algo_name=algo_name, data_param=data_param
                )
                optimizer.run(**algo_config[algo_name])
