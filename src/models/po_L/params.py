from dataclasses import dataclass


@dataclass(frozen=True)
class Parameter:
    num_of_items: int
    num_of_prices: int
    num_of_other_features: int
    depth_of_trees: int
    base_price: int
    solver: str = "Cbc"
    TimeLimit: int = 600
    NoRelHeurTime: float = 0
    MIPFocus: int = 0
    num_multi_start: int = 0
    num_of_simulations: int = 1
    base_seed: int = 0
