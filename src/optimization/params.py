import copy
from dataclasses import dataclass


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
    num_of_items: int
    num_of_prices: int
    num_of_other_features: int
    depth_of_trees: int
    base_price: int
    seed: int = 0
    data_type: str = "realworld"

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
