from __future__ import annotations

from importlib.machinery import SourceFileLoader
from typing import Any, Optional, TypeVar

from src.optimization.algorithms import ALGORITHMS
from src.optimization.model_input import ModelInput
from src.optimization.params import ArtificialDataParameter, RealDataParameter
from src.optimization.result import Result
from src.utils.paths import MODEL_DIR

IndexSet = TypeVar("IndexSet")
Constant = TypeVar("Constant")
Model = TypeVar("Model")
Algorithm = TypeVar("Algorithm")


class Simulator:
    def __init__(
        self,
        data_type: str,
        config_data: dict[str, Any],
        config_opt: dict[str, Any],
        config_algo: dict[str, Any],
        config_pred: Optional[dict[str, Any]] = None,
    ):
        self.data_type = data_type
        self.config_data = config_data
        self.config_opt = config_opt
        self.config_algo = config_algo
        self.config_prediction = config_pred
        self.results_dict: dict[tuple[str, str], list[Result]] = dict()
        self.data_params: list[ArtificialDataParameter | RealDataParameter] = self.make_data_params(
            config_data=config_data, data_type=data_type
        )

    def run(self, iteration: int = 1) -> None:
        if self.data_type == "artificial":
            self.run_artificial(iteration)
        elif self.data_type == "realworld":
            self.run_realworld()

    def run_artificial(self, iteration: int) -> None:
        """人工データによるシミュレーションを実行"""
        model_settings = self.config_opt["model"]
        algo_settings = self.config_algo
        model_algo_names = [
            (model_name, algo_name)
            for model_name in model_settings
            for algo_name in model_settings[model_name]["algorithm"]
        ]
        for model_name, algo_name in model_algo_names:
            algo_class = ALGORITHMS.get(algo_name, None)
            if algo_class is None:
                continue

            results: list[Result] = []
            for data_param in self.data_params:
                for i in range(iteration):
                    data_param.seed = i
                    model_input = self.make_model_input(
                        model_name=model_name, data_param=data_param
                    )
                    model = self.make_model(model_input)
                    algorithm = algo_class(model=model, **algo_settings[algo_name])
                    algorithm.run()
                    algorithm.result.data_param = data_param
                    results.append(algorithm.result)
            self.results_dict[(model_name, algo_name)] = results

    def run_realworld(self) -> None:
        """実データによるシミュレーションを実行"""

    @staticmethod
    def make_model_input(
        model_name: str, data_param: ArtificialDataParameter | RealDataParameter
    ) -> ModelInput:
        """モデルの入力データを作成"""
        module_name = "make_input"
        module_path = str(MODEL_DIR / model_name / (module_name + ".py"))
        make_input_module = SourceFileLoader(module_name, module_path).load_module()
        make_input = getattr(make_input_module, f"make_{data_param.data_type}_input")
        index_set, constant = make_input(params=data_param)
        model_input = ModelInput(model_name=model_name, index_set=index_set, constant=constant)
        return model_input

    @staticmethod
    def make_model(model_input: ModelInput) -> Model:
        """最適化モデルを構築"""
        module_name = "model"
        module_path = str(MODEL_DIR / model_input.model_name / (module_name + ".py"))
        model_module = SourceFileLoader(module_name, module_path).load_module()
        model_class = getattr(model_module, "Model")
        model = model_class(index_set=model_input.index_set, constant=model_input.constant)
        return model

    @staticmethod
    def make_data_params(
        config_data: dict[str, Any], data_type: str
    ) -> list[ArtificialDataParameter | RealDataParameter]:
        """シミュレーションで設定するパラメータの生成"""
        data_params = []
        if data_type == "artificial":
            param = config_data[data_type]["params"]
            for num_of_items in param["num_of_items"]:
                data_param = ArtificialDataParameter(
                    num_of_items=num_of_items,
                    num_of_prices=param["num_of_prices"],
                    num_of_other_features=param["num_of_other_features"],
                    depth_of_trees=param["depth_of_trees"],
                    base_price=param["base_price"],
                )
                data_params.append(data_param)
        elif data_type == "realworld":
            pass
        else:
            pass
        return data_params
