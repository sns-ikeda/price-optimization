from __future__ import annotations

import itertools

from src.optimize.models.POLR.model import Constant, IndexSet
from src.optimize.params import ArtificialDataParameter, RealDataParameter
from src.optimize.processing import rename_dict


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

    beta, beta0, g, D = dict(), dict(), dict(), dict()
    for m in M:
        predictor = params.item2predictor[m]
        feature_cols = predictor.feature_cols
        model = predictor.model
        _coef_dict = {feature_cols[i]: coef for i, coef in enumerate(model.coef_)}
        coef_dict = rename_dict(_coef_dict)
        for col, coef in coef_dict.items():
            beta[m, col] = coef
        beta0[m] = model.intercept_
        D[m] = [col for col in coef_dict.keys() if col not in M]
    g = params.g
    index_set = IndexSet(D=D, M=M, K=K)
    constant = Constant(beta=beta, beta0=beta0, phi=phi, g=g, P=P)
    return index_set, constant
