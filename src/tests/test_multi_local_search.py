from src.algorithm.multi_local_search import MultiLocalSearch
from src.models.po_L.params import Parameter


def test_get_neighbors():
    params = Parameter(
        num_of_items=3,
        num_of_prices=3,
        num_of_other_features=1,
        depth_of_trees=3,
        base_price=1000,
        num_of_simulations=1,
        solver="Cbc",
        TimeLimit=100,
        NoRelHeurTime=100,
        base_seed=42,
    )
    multi_local_search = MultiLocalSearch(params=params)
    x = {0: 0, 1: 0, 2: 0}
    x_neighbors = multi_local_search.get_neighbors(x)
    assert x_neighbors == [{0: 1, 1: 0, 2: 0}, {0: 0, 1: 1, 2: 0}, {0: 0, 1: 0, 2: 1}]
    x = {0: 1, 1: 1, 2: 1}
    x_neighbors = multi_local_search.get_neighbors(x)
    assert x_neighbors == [
        {0: 0, 1: 1, 2: 1},
        {0: 2, 1: 1, 2: 1},
        {0: 1, 1: 0, 2: 1},
        {0: 1, 1: 2, 2: 1},
        {0: 1, 1: 1, 2: 0},
        {0: 1, 1: 1, 2: 2},
    ]
    x = {0: 0, 1: 1, 2: 2}
    x_neighbors = multi_local_search.get_neighbors(x)
    assert x_neighbors == [
        {0: 1, 1: 1, 2: 2},
        {0: 0, 1: 0, 2: 2},
        {0: 0, 1: 2, 2: 2},
        {0: 0, 1: 1, 2: 1},
    ]
