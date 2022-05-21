from __future__ import annotations

import itertools
import random

import numpy as np

from src.models.po_L.model import Constant, IndexSet
from src.models.po_L.params import Parameter
from src.processing.binary_tree import depth2branchnodes, depth2leaves, leaf2LtRt


def make_sample_input(params: Parameter) -> tuple[IndexSet, Constant]:
    """サンプルデータを作成s"""
    # 集合を作成
    M = list(range(params.num_of_items))
    K = list(range(params.num_of_prices))
    D = {
        m: list(map(lambda x: max(M) + x + 1, list(range(params.num_of_other_features)))) for m in M
    }
    D_ = {m: [max(D[m]) + 1] if D[m] else [max(M) + 1] for m in M}
    TL = {m: depth2leaves(params.depth_of_trees) for m in M}
    TB = {m: depth2branchnodes(params.depth_of_trees) for m in M}
    L, R = dict(), dict()
    for m in M:
        for t in TL[m]:
            Lt, Rt = leaf2LtRt(t)
            L[m, t] = Lt
            R[m, t] = Rt
    index_set = IndexSet(D=D, D_=D_, M=M, K=K, TL=TL, L=L, R=R)

    # 定数を作成
    base_price = params.base_price
    unit_price = int(base_price / len(K))
    price_max = base_price + unit_price * max(K)

    def scale_price(base_price: int, price_max: int, unit_price: int, k: int) -> float:
        scaled_price = (base_price + unit_price * k) / price_max
        return scaled_price

    P = {
        (m, k): round(scale_price(base_price, price_max, unit_price, k), 3)
        for m, k in itertools.product(M, K)
    }
    phi = {
        (m, mp, k): round(scale_price(base_price, price_max, unit_price, k), 3)
        for m, mp, k in itertools.product(M, M, K)
    }
    a, b, g, epsilon, epsilon_max, beta = dict(), dict(), dict(), dict(), dict(), dict()
    for m in M:
        epsilon[m] = 0.001
        epsilon_max[m] = 1.0
        for t in TB[m]:
            b[m, t] = round(np.random.rand(), 3)
            # aは特徴量の中から一つだけ1がたつ
            node_to_one = random.choice(M + D[m])
            for mp in M + D[m]:
                if mp == node_to_one:
                    a[m, mp, t] = 1
                else:
                    a[m, mp, t] = 0

        for d in D[m] + D_[m]:
            if d in D_[m]:
                g[m, d] = 1
            else:
                g[m, d] = round(np.random.rand(), 3)

        for mp in M + D[m] + D_[m]:
            for t in TL[m]:
                beta[m, mp, t] = round(np.random.randn(), 3)
    constant = Constant(
        beta=beta, phi=phi, epsilon=epsilon, epsilon_max=epsilon_max, a=a, b=b, g=g, P=P
    )
    return index_set, constant
