from __future__ import annotations

import itertools

from interpretableai import iai

from src.optimize.models.POORT_LH.model import Constant, IndexSet
from src.optimize.params import ArtificialDataParameter, RealDataParameter
from src.optimize.processing import rename_dict


def make_artificial_input(params: ArtificialDataParameter) -> tuple[IndexSet, Constant]:
    """人工的にモデルのパラメータを生成"""


def make_realworld_input(params: RealDataParameter) -> tuple[IndexSet, Constant]:
    """実際のデータからモデルのパラメータを生成"""
    M = list(params.item2prices.keys())
    K = list(range(params.num_of_prices))
    P = dict()
    for m, k in itertools.product(M, K):
        P[m, k] = params.item2prices[m][k]

    TL, L, R = dict(), dict(), dict()
    beta, beta0, epsilon, epsilon_max, a, b = dict(), dict(), dict(), dict(), dict(), dict()
    for m in M:
        predictor = params.item2predictor[m]
        _beta = get_beta(model=predictor.model, item=m)
        _beta0 = get_beta0(model=predictor.model, item=m)
        epsilon[m] = 0.001
        epsilon_max[m] = 100000
        _a = get_a(model=predictor.model, item=m)
        _b = get_b(model=predictor.model, item=m)
        TL[m] = get_leafnodes(model=predictor.model)
        _L, _R = get_LR(model=predictor.model, item=m)
        beta.update(_beta)
        beta0.update(_beta0)
        a.update(_a)
        b.update(_b)
        L.update(_L)
        R.update(_R)

    g = params.g
    D = list(g.keys())
    index_set = IndexSet(D=D, M=M, K=K, TL=TL, L=L, R=R)
    constant = Constant(
        beta=beta, beta0=beta0, epsilon=epsilon, epsilon_max=epsilon_max, a=a, b=b, g=g, P=P
    )
    print("beta", constant.beta)
    print("beta0", constant.beta0)
    print("a", constant.a)
    print("b", constant.b)
    print("L", index_set.L)
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
        print("item", item)
        print("coef", coefs)
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
            continue
        for _split_weight_dict in split_weights:
            split_weight_dict = rename_dict(_split_weight_dict)
            for col, value in split_weight_dict.items():
                a[item, col, t] = value
    return a


def get_b(model: iai.OptimalTreeRegressor, item: str) -> dict[tuple[str, int], float]:
    b = dict()
    branch_nodes = get_branchnodes(model)
    for t in branch_nodes:
        split_threshold = model.get_split_threshold(node_index=t)
        b[item, t] = split_threshold
    return b
