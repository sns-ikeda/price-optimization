from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

from src.optimize.algorithms import ALGORITHMS
from src.optimize.params import ArtificialDataParameter, RealDataParameter
from src.utils.handle_module import get_object_from_module
from src.utils.paths import MODEL_DIR

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
    ):
        self.model_name = model_name
        self.algo_name = algo_name
        self.data_param = data_param
        self.algorithm = None
        self.result = None

    @staticmethod
    def make_model_input(
        model_name: str, data_param: ArtificialDataParameter | RealDataParameter
    ) -> ModelInput:
        """モデルの入力データを作成"""
        module_path = MODEL_DIR / model_name / "make_input.py"
        make_input = get_object_from_module(module_path, f"make_{data_param.data_type}_input")
        index_set, constant = make_input(params=data_param)
        model_input = ModelInput(model_name=model_name, index_set=index_set, constant=constant)
        return model_input

    @staticmethod
    def make_model(model_input: ModelInput) -> Model:
        """最適化モデルを構築"""
        module_path = MODEL_DIR / model_input.model_name / "model.py"
        model_class = get_object_from_module(module_path, "Model")
        model = model_class(index_set=model_input.index_set, constant=model_input.constant)
        return model

    def run(self, **kwargs) -> None:
        model_input = Optimizer.make_model_input(
            model_name=self.model_name, data_param=self.data_param
        )
        model = Optimizer.make_model(model_input)
        algo_class = ALGORITHMS.get(self.algo_name, None)
        if algo_class is not None:
            self.algorithm = algo_class(model=model, **kwargs)
            self.algorithm.run()
            self.algorithm.result.data_param = self.data_param
            self.result = self.algorithm.result
