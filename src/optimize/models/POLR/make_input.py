from __future__ import annotations

import itertools

from src.optimize.models.POLR.model import Constant, IndexSet
from src.optimize.params import ArtificialDataParameter, RealDataParameter


def make_artificial_input(params: ArtificialDataParameter) -> tuple[IndexSet, Constant]:
    """人工的にモデルのパラメータを生成"""
    index_set, constant = None, None
    return index_set, constant


def make_realworld_input(params: RealDataParameter) -> tuple[IndexSet, Constant]:
    """実際のデータからモデルのパラメータを生成"""
    M = list(params.item2prices.keys())
    K = list(range(params.num_of_prices))
    P = dict()
    for m, k in itertools.product(M, K):
        P[m, k] = params.item2prices[m][k]
    phi = {(m, mp, k): 1 for m, mp, k in itertools.product(M, M, K)}

    price_cols = ["PRICE" + "_" + m for m in M]
    beta, g, D, D_ = dict(), dict(), dict(), dict()
    for m in M:
        predictor = params.item2predictor[m]
        for col, coef in predictor.coef_dict.items():
            beta[m, col] = coef
            g[col] = params.g[col]
        beta["intercept"] = predictor.intercept
        g["intercept"] = 1
        D[m] = [col for col in predictor.coef_dict.keys() if col not in price_cols]
        D_[m] = ["intercept"]
    index_set = IndexSet(D=D, D_=D_, M=M, K=K)
    constant = Constant(beta=beta, phi=phi, g=g, P=P)
    return index_set, constant
