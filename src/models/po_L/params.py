from dataclasses import dataclass


@dataclass(frozen=True)
class Parameter:
    num_of_items: int
    num_of_prices: int
    num_of_other_features: int
    depth_of_trees: int
    base_price: int
    num_of_simulations: int
