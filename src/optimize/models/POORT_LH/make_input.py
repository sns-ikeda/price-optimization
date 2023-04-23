from __future__ import annotations

import itertools

import numpy as np
from interpretableai import iai
from logzero import logger

from src.optimize.models.POORT_LH.model import Constant, IndexSet
from src.optimize.params import SyntheticDataParameter, RealDataParameter
from src.optimize.processing import rename_dict, rename_feature
from src.optimize.processing.binary_tree import depth2branchnodes, depth2leaves
from src.optimize.processing.binary_tree import leaf2LtRt as leaf2LtRt_

# 超平面分割の場合
# def make_artificial_input(params: ArtificialDataParameter) -> tuple[IndexSet, Constant]:
#     """人工的にモデルのパラメータを生成"""
#     # 集合を作成
#     M = [str(m) for m in range(params.num_of_items)]
#     K = list(range(params.num_of_prices))
#     _D = [max(M) + 1 + i for i in range(params.num_of_other_features)]
#     D = {m: _D for m in M}
#     TL = {m: depth2leaves(params.depth_of_trees) for m in M}
#     TB = {m: depth2branchnodes(params.depth_of_trees) for m in M}
#     L, R = dict(), dict()
#     for m in M:
#         for t in TL[m]:
#             Lt, Rt = leaf2LtRt_(leaf_node=t)
#             L[m, t] = Lt
#             R[m, t] = Rt
#     index_set = IndexSet(D=D, M=M, K=K, TL=TL, L=L, R=R)

#     # 定数を作成
#     prices = list(np.linspace(params.price_min, params.price_max, params.num_of_prices))
#     prices = [round(price, 3) for price in prices]
#     price_min, price_max = min(prices), max(prices)
#     P = {(m, k): prices[k] for m, k in itertools.product(M, K)}
#     # base_price = params.base_price
#     # unit_price = int(base_price / len(K))
#     # price_max = base_price + unit_price * max(K)

#     # def scale_price(base_price: int, price_max: int, unit_price: int, k: int) -> float:
#     #     scaled_price = (base_price + unit_price * k) / price_max
#     #     return scaled_price

#     # P = {
#     #     (m, k): round(scale_price(base_price, price_max, unit_price, k), 3)
#     #     for m, k in itertools.product(M, K)
#     # }
#     base_seed = params.seed
#     a, b, g, epsilon, beta, beta0 = dict(), dict(), dict(), dict(), dict(), dict()
#     for m in M:
#         epsilon[m] = 0.00001
#         for t in TB[m]:
#             for mp in M + D[m]:
#                 np.random.seed(base_seed + int(m) + 5 * int(mp) + t)
#                 a[m, mp, t] = round(np.random.rand(), 3)
#             a_sum = sum([a[m, mp, t] for mp in M + D[m]])
#             # depth <= 2でしか対応できない
#             # if t == 0:
#             #     price_mid = (price_min + price_max) / 2
#             # elif t % 2 == 0:
#             #     price_mid = (price_mid + price_max) / 2
#             # else:
#             #     price_mid = (price_mid + price_min) / 2
#             # b[m, t] = round(price_mid * a_sum, 3)
#             b[m, t] = round((price_min + price_max) / 2 * a_sum, 3)

#         for t in TL[m]:
#             # beta0[m, t] = round(np.random.normal(loc=100, scale=10, size=1)[0], 3)
#             beta0[m, t] = round((200 - 100) * np.random.rand() + 100, 3)
#             for mp in M + D[m]:
#                 np.random.seed(base_seed + int(m) + 10 * int(mp) + t)
#                 if m == mp:
#                     beta[m, mp, t] = round(np.random.normal(loc=-1, scale=1, size=1)[0], 3)
#                     # beta[m, mp, t] = - round((20 - 10) * np.random.rand() + 10, 3)
#                 else:
#                     beta[m, mp, t] = round(np.random.normal(loc=1, scale=1, size=1)[0], 3)
#                     # beta[m, mp, t] = round((5 - 0) * np.random.rand(), 3)
#         for d in D[m]:
#             np.random.seed(base_seed + int(m) + d)
#             g[m, d] = round(np.random.rand(), 3)
#     constant = Constant(beta=beta, beta0=beta0, epsilon=epsilon, a=a, b=b, g=g, P=P, prices=prices)
#     logger.info(f"D: {D}")
#     logger.info(f"beta: {beta}")
#     logger.info(f"beta0: {beta0}")
#     logger.info(f"g: {g}")
#     logger.info(f"a: {a}")
#     logger.info(f"b: {b}")
#     return index_set, constant

# 1変数分岐の場合
def make_synthetic_input(params: SyntheticDataParameter) -> tuple[IndexSet, Constant]:
    """人工的にモデルのパラメータを生成"""
    # 集合を作成
    M = [str(m) for m in range(params.num_of_items)]
    K = list(range(params.num_of_prices))
    _D = [max(M) + 1 + i for i in range(params.num_of_other_features)]
    D = {m: _D for m in M}
    TL = {m: depth2leaves(params.depth_of_trees) for m in M}
    TB = {m: depth2branchnodes(params.depth_of_trees) for m in M}
    L, R = dict(), dict()
    for m in M:
        for t in TL[m]:
            Lt, Rt = leaf2LtRt_(leaf_node=t)
            L[m, t] = Lt
            R[m, t] = Rt
    index_set = IndexSet(D=D, M=M, K=K, TL=TL, L=L, R=R)

    # 定数を作成
    prices = list(np.linspace(params.price_min, params.price_max, params.num_of_prices))
    prices = [round(price, 3) for price in prices]
    price_min, price_max = min(prices), max(prices)
    P = {(m, k): prices[k] for m, k in itertools.product(M, K)}

    base_seed = params.seed
    a, b, g, epsilon, beta, beta0 = dict(), dict(), dict(), dict(), dict(), dict()
    for m in M:
        epsilon[m] = 0.00001
        for t in TB[m]:
            np.random.seed(base_seed + int(m) + 50 + t)
            b[m, t] = np.random.choice(prices[1:-1])
            picked_m = np.random.choice(M)
            for mp in M + D[m]:
                if mp == picked_m:
                    a[m, mp, t] = 1
                else:
                    a[m, mp, t] = 0

        for t in TL[m]:
            # beta0[m, t] = round(np.random.normal(loc=100, scale=10, size=1)[0], 3)
            beta0[m, t] = round((200 - 100) * np.random.rand() + 100, 3)
            for mp in M + D[m]:
                np.random.seed(base_seed + int(m) + 100 * int(mp) + t)
                if m == mp:
                    beta[m, mp, t] = round(np.random.normal(loc=-1, scale=1, size=1)[0], 3)
                    # beta[m, mp, t] = - round((20 - 10) * np.random.rand() + 10, 3)
                else:
                    beta[m, mp, t] = round(np.random.normal(loc=1, scale=1, size=1)[0], 3)
                    # beta[m, mp, t] = round((5 - 0) * np.random.rand(), 3)
        for d in D[m]:
            np.random.seed(base_seed + int(m) + d)
            g[m, d] = round(np.random.rand(), 3)
    constant = Constant(beta=beta, beta0=beta0, epsilon=epsilon, a=a, b=b, g=g, P=P, prices=prices)
    logger.info(f"D: {D}")
    logger.info(f"beta: {beta}")
    logger.info(f"beta0: {beta0}")
    logger.info(f"g: {g}")
    logger.info(f"a: {a}")
    logger.info(f"b: {b}")
    return index_set, constant


# # 1商品の場合
# def make_artificial_input(params: ArtificialDataParameter) -> tuple[IndexSet, Constant]:
#     """人工的にモデルのパラメータを生成"""
#     # 集合を作成
#     M = [str(m) for m in range(params.num_of_items)]
#     K = list(range(params.num_of_prices))
#     _D = [max(M) + 1 + i for i in range(params.num_of_other_features)]
#     D = {m: _D for m in M}
#     TL = {m: depth2leaves(params.depth_of_trees) for m in M}
#     TB = {m: depth2branchnodes(params.depth_of_trees) for m in M}
#     L, R = dict(), dict()
#     for m in M:
#         for t in TL[m]:
#             Lt, Rt = leaf2LtRt_(leaf_node=t)
#             L[m, t] = Lt
#             R[m, t] = Rt
#     index_set = IndexSet(D=D, M=M, K=K, TL=TL, L=L, R=R)

#     # 定数を作成
#     prices = list(np.linspace(params.price_min, params.price_max, params.num_of_prices))
#     prices = [round(price, 3) for price in prices]
#     price_min, price_max = min(prices), max(prices)
#     P = {(m, k): prices[k] for m, k in itertools.product(M, K)}

#     base_seed = params.seed
#     a, b, g, epsilon, beta, beta0 = dict(), dict(), dict(), dict(), dict(), dict()
#     values_b = {0: 0.9, 1: 0.85, 2: 0.95}
#     t_to_x1x2 = {3: (0.8, 0.85), 4: (0.85, 0.9), 5: (0.9, 0.95), 6: (0.95, 1)}
#     mt_to_y1y2 = {
#         ("0", 3): (200, 195), ("0", 4): (195, 120), ("0", 5): (120, 105), ("0", 6): (105, 100),
#         ("1", 3): (200, 195), ("1", 4): (195, 180), ("1", 5): (180, 105), ("1", 6): (105, 100),
#         ("2", 3): (200, 195), ("2", 4): (195, 180), ("2", 5): (180, 150), ("2", 6): (150, 100),
#         ("3", 3): (200, 150), ("3", 4): (150, 120), ("3", 5): (120, 105), ("3", 6): (105, 100),
#         ("4", 3): (200, 200), ("4", 4): (200, 200), ("4", 5): (200, 100), ("4", 6): (100, 100),
#     }
#     def calc_beta(x1, y1, x2, y2):
#         return ((y1 - y2) / (x1 - x2))

#     def calc_beta0(x1, y1, x2, y2):
#         return ((x1 * y2 - x2 * y1) / (x1 - x2))

#     for m in M:
#         epsilon[m] = 0.00001
#         for mp in M + D[m]:
#             for t in TB[m]:
#                 a[m, mp, t] = 1 if m == mp else 0
#                 b[m, t] = values_b[t]

#         for t in TL[m]:
#             x1, x2 = t_to_x1x2[t]
#             y1, y2 = mt_to_y1y2[m, t]
#             beta0[m, t] = calc_beta0(x1=x1, y1=y1, x2=x2, y2=y2)
#             for mp in M + D[m]:
#                 if m == mp:
#                     beta[m, mp, t] = calc_beta(x1=x1, y1=y1, x2=x2, y2=y2)
#                 else:
#                     beta[m, mp, t] = 0
#         for d in D[m]:
#             np.random.seed(base_seed + int(m) + d)
#             g[m, d] = round(np.random.rand(), 3)
#     constant = Constant(beta=beta, beta0=beta0, epsilon=epsilon, a=a, b=b, g=g, P=P, prices=prices)
#     logger.info(f"D: {D}")
#     logger.info(f"TL: {TL}")
#     logger.info(f"TB: {TB}")
#     logger.info(f"beta: {beta}")
#     logger.info(f"beta0: {beta0}")
#     logger.info(f"g: {g}")
#     logger.info(f"a: {a}")
#     logger.info(f"b: {b}")

#     return index_set, constant


# # 2商品の場合
# def make_artificial_input(params: ArtificialDataParameter) -> tuple[IndexSet, Constant]:
#     """人工的にモデルのパラメータを生成"""
#     # 集合を作成
#     M = [str(m) for m in range(params.num_of_items)]
#     K = list(range(params.num_of_prices))
#     _D = [max(M) + 1 + i for i in range(params.num_of_other_features)]
#     D = {m: _D for m in M}
#     TL = {m: depth2leaves(params.depth_of_trees) for m in M}
#     TB = {m: depth2branchnodes(params.depth_of_trees) for m in M}
#     L, R = dict(), dict()
#     for m in M:
#         for t in TL[m]:
#             Lt, Rt = leaf2LtRt_(leaf_node=t)
#             L[m, t] = Lt
#             R[m, t] = Rt
#     index_set = IndexSet(D=D, M=M, K=K, TL=TL, L=L, R=R)

#     # 定数を作成
#     prices = list(np.linspace(params.price_min, params.price_max, params.num_of_prices))
#     prices = [round(price, 3) for price in prices]
#     price_min, price_max = min(prices), max(prices)
#     P = {(m, k): prices[k] for m, k in itertools.product(M, K)}

#     base_seed = params.seed
#     a, b, g, epsilon, beta, beta0 = dict(), dict(), dict(), dict(), dict(), dict()
#     a = {
#         ("0", "0", 0): 1, ("0", "0", 1): 0, ("0", "0", 2): 0,
#         ("0", "1", 0): 0, ("0", "1", 1): 1, ("0", "1", 2): 1,
#         ("1", "0", 0): 0, ("1", "0", 1): 1, ("1", "0", 2): 1,
#         ("1", "1", 0): 1, ("1", "1", 1): 0, ("1", "1", 2): 0,
#     }
#     b = {
#         ("0", 0): 0.95, ("0", 1): 0.9, ("0", 2): 0.9,
#         ("1", 0): 0.95, ("1", 1): 0.9, ("1", 2): 0.9
#     }
#     beta0 = {
#         ("0", 3): 505, ("0", 4): 470, ("0", 5): 220, ("0", 6): 130,
#         ("1", 3): 145, ("1", 4): 230, ("1", 5): 1340, ("1", 6): 1420
#     }
#     beta = {
#         ("0", "0", 3): -500, ("0", "0", 4): -400, ("0", "0", 5): -200, ("0", "0", 6): -200,
#         ("0", "1", 3): 100, ("0", "1", 4): 50, ("0", "1", 5): 100, ("0", "1", 6): 200,
#         ("1", "0", 3): 100, ("1", "0", 4): 50, ("1", "0", 5): 200, ("1", "0", 6): 100,
#         ("1", "1", 3): -50, ("1", "1", 4): -100, ("1", "1", 5): -1400, ("1", "1", 6): -1400,
#     }
#     for m in M:
#         epsilon[m] = 0.00001
#         for d in D[m]:
#             np.random.seed(base_seed + int(m) + d)
#             g[m, d] = round(np.random.rand(), 3)
#     constant = Constant(beta=beta, beta0=beta0, epsilon=epsilon, a=a, b=b, g=g, P=P, prices=prices)
#     logger.info(f"D: {D}")
#     logger.info(f"TL: {TL}")
#     logger.info(f"TB: {TB}")
#     logger.info(f"beta: {beta}")
#     logger.info(f"beta0: {beta0}")
#     logger.info(f"g: {g}")
#     logger.info(f"a: {a}")
#     logger.info(f"b: {b}")

#     return index_set, constant


def make_realworld_input(params: RealDataParameter) -> tuple[IndexSet, Constant]:
    """実際のデータからモデルのパラメータを生成"""
    M = list(params.item2prices.keys())
    K = list(range(params.num_of_prices))
    P = dict()
    for m, k in itertools.product(M, K):
        P[m, k] = params.item2prices[m][k]

    TL, TB, L, R = dict(), dict(), dict(), dict()
    beta, beta0, epsilon, a, b = dict(), dict(), dict(), dict(), dict()
    for m in M:
        predictor = params.item2predictor[m]
        _beta = get_beta(model=predictor.model, item=m)
        _beta0 = get_beta0(model=predictor.model, item=m)
        epsilon[m] = 0.0001
        _a = get_a(model=predictor.model, item=m)
        _b = get_b(model=predictor.model, item=m)
        TL[m] = get_leafnodes(model=predictor.model)
        TB[m] = get_branchnodes(model=predictor.model)
        _L, _R = get_LR(model=predictor.model, item=m)
        beta.update(_beta)
        beta0.update(_beta0)
        a.update(_a)
        b.update(_b)
        L.update(_L)
        R.update(_R)

    g = params.g
    D = {m: [] for m in M}
    for k in g.keys():
        m, d = k
        D[m].append(d)
    index_set = IndexSet(D=D, M=M, K=K, TL=TL, L=L, R=R)
    constant = Constant(beta=beta, beta0=beta0, epsilon=epsilon, a=a, b=b, g=g, P=P)
    logger.info(f"D: {D}")
    logger.info(f"TL: {TL}")
    logger.info(f"TB: {TB}")
    logger.info(f"beta: {beta}")
    logger.info(f"beta0: {beta0}")
    logger.info(f"g: {g}")
    logger.info(f"a: {a}")
    logger.info(f"b: {b}")
    return index_set, constant


def get_leafnodes(model: iai.OptimalTreeRegressor) -> list[int]:
    all_nodes = list(range(1, model.get_num_nodes() + 1))
    leaf_nodes = []
    for t in all_nodes:
        if model.is_leaf(t):
            leaf_nodes.append(t)
    return leaf_nodes


def get_branchnodes(model: iai.OptimalTreeRegressor) -> list[int]:
    all_nodes = list(range(1, model.get_num_nodes() + 1))
    leaf_nodes = get_leafnodes(model)
    branch_nodes = [t for t in all_nodes if t not in leaf_nodes]
    return branch_nodes


def get_beta(model: iai.OptimalTreeRegressor, item: str) -> dict[tuple[str, str, int], float]:
    leaf_nodes = get_leafnodes(model)
    beta = dict()
    for t in leaf_nodes:
        coefs = model.get_regression_weights(node_index=t)
        for _coef_dict in coefs:
            coef_dict = rename_dict(_coef_dict)
            for col, value in coef_dict.items():
                beta[(item, col, t)] = value
    return beta


def get_beta0(model: iai.OptimalTreeRegressor, item: str) -> dict[tuple[str, int], float]:
    leaf_nodes = get_leafnodes(model)
    beta0 = dict()
    for t in leaf_nodes:
        constant = model.get_regression_constant(node_index=t)
        beta0[(item, t)] = constant
    return beta0


def leaf2LtRt(model: iai.OptimalTreeRegressor, leaf_node: int) -> tuple[list[int], list[int]]:
    Lt, Rt = [], []
    counter = 0
    parent_node = None
    child_node = leaf_node
    while True:
        counter += 1
        if counter > 1000:
            raise Exception("Infinite loop error")
        if parent_node is not None:
            child_node = parent_node
        try:
            parent_node = model.get_parent(node_index=child_node)
        except ValueError:
            break
        if child_node == model.get_lower_child(parent_node):
            Lt.append(parent_node)
        if child_node == model.get_upper_child(parent_node):
            Rt.append(parent_node)
    return sorted(Lt), sorted(Rt)


def get_LR(
    model: iai.OptimalTreeRegressor, item: str
) -> tuple[dict[tuple[str, int], list[int]], dict[tuple[str, int], list[int]]]:
    L, R = dict(), dict()
    leaf_nodes = get_leafnodes(model)
    for t in leaf_nodes:
        Lt, Rt = leaf2LtRt(model=model, leaf_node=t)
        L[item, t] = Lt
        R[item, t] = Rt
    return L, R


def get_a(model: iai.OptimalTreeRegressor, item: str) -> dict[tuple[str, str, int], float]:
    a = dict()
    branch_nodes = get_branchnodes(model)
    for t in branch_nodes:
        try:
            split_weights = model.get_split_weights(node_index=t)
        except ValueError:
            split_feature = model.get_split_feature(node_index=t)
            item_ = rename_feature(split_feature)
            a[item, item_, t] = 1
            continue
        for _split_weight_dict in split_weights:
            split_weight_dict = rename_dict(_split_weight_dict)
            for item_, value in split_weight_dict.items():
                a[item, item_, t] = value
    return a


def get_b(model: iai.OptimalTreeRegressor, item: str) -> dict[tuple[str, int], float]:
    b = dict()
    branch_nodes = get_branchnodes(model)
    for t in branch_nodes:
        split_threshold = model.get_split_threshold(node_index=t)
        b[item, t] = split_threshold
    return b
