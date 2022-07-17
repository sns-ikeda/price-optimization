from __future__ import annotations

from collections import defaultdict
from importlib.machinery import SourceFileLoader
from typing import Any, TypeVar

from src.configs import CONFIG_ALG, CONFIG_OPT
from src.optimization.algorithms import ALGORITHMS
from src.optimization.params import ArtificialDataParameter, RealDataParameter
from src.utils.paths import MODEL_DIR

IndexSet = TypeVar("IndexSet")
Constant = TypeVar("Constant")
Model = TypeVar("Model")


class Optimizer:
    def __init__(
        self,
        data_type: str,
        data_params: dict[str, ArtificialDataParameter | RealDataParameter],
        model_settings: dict[str, dict[str, list[str]]] = CONFIG_OPT["model"],
        algorithm_settings: dict[str, Any] = CONFIG_ALG,
    ) -> None:
        self.data_type = data_type
        self.data_params = data_params
        self.result = defaultdict(dict)
        self.model_settings = model_settings
        self.algorithm_settings = algorithm_settings
        self.algorithms = defaultdict(dict)

    def make_model_input(self, model_name: str) -> tuple[IndexSet, Constant]:
        """モデルで使用するデータの作成"""
        module_name = "make_input"
        module_path = str(MODEL_DIR / model_name / (module_name + ".py"))
        make_input_module = SourceFileLoader(module_name, module_path).load_module()
        make_input = getattr(make_input_module, f"make_{self.data_type}_input")
        index_set, constant = make_input(params=self.data_params[self.data_type])
        return index_set, constant

    def make_model(self, model_name: str) -> Model:
        """最適化モデルの作成"""
        index_set, constant = self.make_model_input(model_name=model_name)
        module_name = "model"
        module_path = str(MODEL_DIR / model_name / (module_name + ".py"))
        model_module = SourceFileLoader(module_name, module_path).load_module()
        Model_ = getattr(model_module, "Model")
        model = Model_(index_set=index_set, constant=constant)
        return model

    def run(self) -> None:
        for model_name in self.model_settings:
            for algorithm_name in self.model_settings[model_name]["algorithm"]:
                _algorithm = ALGORITHMS.get(algorithm_name, None)
                if _algorithm is not None:
                    model = self.make_model(model_name)
                    algorithm = _algorithm(model=model, **self.algorithm_settings[algorithm_name])
                    algorithm.run()
                    self.result[model_name][algorithm_name] = algorithm.result
                    self.algorithms[model_name][algorithm_name] = algorithm
