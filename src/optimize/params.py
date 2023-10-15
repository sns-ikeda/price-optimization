from __future__ import annotations

import copy
from dataclasses import dataclass, field

import pandas as pd

from src.configs import SyntheticConfig
from src.predict.predictor import Predictor


@dataclass(frozen=True)
class AlgorithmParameter:
    solver: str = "Cbc"
    TimeLimit: int = 3600
    NoRelHeurTime: float = 0
    MIPFocus: int = 0
    num_multi_start: int = 0
    base_seed: int = 0


@dataclass(frozen=False)
class SyntheticDataParameter:
    num_of_items: int
    num_of_prices: int
    num_of_other_features: int
    depth_of_trees: int
    base_price: int = 5
    price_min: float = 4.0
    price_max: float = 6.0
    base_quantity: int = 300
    seed: int = 0
    data_type: str = "synthetic"

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
    g: dict[tuple[str, str], float] = field(default_factory=dict)


def make_data_params(
    config: SyntheticConfig, data_type: str, **kwargs
) -> list[SyntheticDataParameter | RealDataParameter]:
    """Generate data parameters from config"""
    data_params = []
    if data_type == "synthetic":
        for num_of_items in config.num_of_items:
            data_param = SyntheticDataParameter(
                num_of_items=num_of_items,
                num_of_prices=config.num_of_prices,
                num_of_other_features=config.num_of_other_features,
                depth_of_trees=config.depth_of_trees,
                base_price=config.base_price,
            )
            data_params.append(data_param)
    return data_params


def calc_g(df: pd.DataFrame, item2predictor: dict[str, Predictor]) -> dict[tuple[str, str], float]:
    price_cols = ["PRICE" + "_" + item for item in item2predictor.keys()]
    g = dict()
    for item, predictor in item2predictor.items():
        if len(df) == 1:
            X = df[list(predictor.feature_cols)]
        else:
            X = df[list(predictor.feature_cols)].head(1)
        print("item", item)
        print("predictor.feature_cols", list(predictor.feature_cols))
        print("X", X)
        other_feature_cols = [col for col in list(predictor.feature_cols) if col not in price_cols]
        print("other_feature", other_feature_cols)
        g.update({(item, col): float(X[col]) for col in other_feature_cols})
    return g
