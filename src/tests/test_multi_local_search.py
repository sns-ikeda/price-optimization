# from src.optimization.algorithms.multi_local_search import MultiLocalSearch


# def test_get_neighbors():
#     multi_local_search = MultiLocalSearch(params=params)
#     x = {0: 0, 1: 0, 2: 0}
#     x_neighbors = multi_local_search.get_neighbors(x)
#     assert x_neighbors == [{0: 1, 1: 0, 2: 0}, {0: 0, 1: 1, 2: 0}, {0: 0, 1: 0, 2: 1}]
#     x = {0: 1, 1: 1, 2: 1}
#     x_neighbors = multi_local_search.get_neighbors(x)
#     assert x_neighbors == [
#         {0: 0, 1: 1, 2: 1},
#         {0: 2, 1: 1, 2: 1},
#         {0: 1, 1: 0, 2: 1},
#         {0: 1, 1: 2, 2: 1},
#         {0: 1, 1: 1, 2: 0},
#         {0: 1, 1: 1, 2: 2},
#     ]
#     x = {0: 0, 1: 1, 2: 2}
#     x_neighbors = multi_local_search.get_neighbors(x)
#     assert x_neighbors == [
#         {0: 1, 1: 1, 2: 2},
#         {0: 0, 1: 0, 2: 2},
#         {0: 0, 1: 2, 2: 2},
#         {0: 0, 1: 1, 2: 1},
#     ]
