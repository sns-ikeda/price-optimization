from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from src.predict.predictor import Predictor


@dataclass(frozen=True)
class AlgorithmParameter:
    solver: str = "Cbc"
    TimeLimit: int = 600
    NoRelHeurTime: float = 0
    MIPFocus: int = 0
    num_multi_start: int = 0
    base_seed: int = 0


@dataclass(frozen=False)
class ArtificialDataParameter:
    num_of_items: int
    num_of_prices: int
    num_of_other_features: int
    depth_of_trees: int
    base_price: int
    seed: int = 0
    data_type: str = "artificial"

    def __eq__(self, other):
        if other is None or type(self) != type(other):
            return False
        self_dict = copy.deepcopy(self.__dict__)
        self_dict.pop("seed")
        other_dict = copy.deepcopy(other.__dict__)
        other_dict.pop("seed")
        return self_dict == other_dict

    def __ne__(self, other):
        return not self.__eq__(other)


@dataclass(frozen=True)
class RealDataParameter:
    num_of_prices: int
    item2prices: dict[str, list[float]]
    seed: int = 0
    item2predictor: dict[str, Predictor] = field(default_factory=dict)
    data_type: str = "realworld"
    g: dict[str, float] = field(default_factory=dict)


def make_data_params(
    config_data: dict[str, Any], data_type: str, **kwargs
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
    return data_params
