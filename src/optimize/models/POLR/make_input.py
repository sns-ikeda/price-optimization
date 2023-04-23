from __future__ import annotations

import itertools

import numpy as np
from logzero import logger

from src.optimize.models.POLR.model import Constant, IndexSet
from src.optimize.params import SyntheticDataParameter, RealDataParameter
from src.optimize.processing import rename_dict


def make_synthetic_input(params: SyntheticDataParameter) -> tuple[IndexSet, Constant]:
    """人工的にモデルのパラメータを生成"""
    # 集合を作成
    M = [str(m) for m in range(params.num_of_items)]
    K = list(range(params.num_of_prices))
    _D = [str(len(M) + i) for i in range(params.num_of_other_features)]
    D = {m: _D for m in M}
    index_set = IndexSet(D=D, M=M, K=K)
    prices = list(np.linspace(params.price_min, params.price_max, params.num_of_prices))
    # 定数を作成
    P = {(m, k): prices[k] for m, k in itertools.product(M, K)}
    base_seed = params.seed
    g, beta, beta0 = dict(), dict(), dict()
    # quantity_min = params.base_quantity * 0.8
    # quantity_max = params.base_quantity * 1.2
    # coef = 3
    for m in M:
        # beta0[m] = int((quantity_max - quantity_min) * np.random.rand() + quantity_min)
        np.random.seed(base_seed + int(m))
        # beta0[m] = round(np.random.normal(loc=100, scale=10, size=1)[0], 3)
        beta0[m] = round((200 - 100) * np.random.rand() + 100, 3)
        target_m = np.random.choice([m_ for m_ in M if m_ != m])
        for mp in M + D[m]:
            np.random.seed(base_seed + int(m) + 10 * int(mp))
            # beta[m, mp] = round(np.random.normal(loc=0, scale=1, size=1)[0], 3)
            if m == mp:
                beta[m, mp] = round(np.random.normal(loc=-10, scale=1, size=1)[0], 3)
            else:
                # beta[m, mp] = round(np.random.normal(loc=1, scale=1, size=1)[0], 3)
                if mp == target_m:
                    beta[m, mp] = round(np.random.normal(loc=5, scale=1, size=1)[0], 3)
                else:
                    beta[m, mp] = 0
        for d in D[m]:
            np.random.seed(base_seed + int(d))
            # g[m, d] = round(np.random.rand(), 3)
            g[m, d] = 1.0
    constant = Constant(beta=beta, beta0=beta0, g=g, P=P, prices=prices)
    logger.info(f"D: {D}")
    logger.info(f"beta: {beta}")
    logger.info(f"beta0: {beta0}")
    logger.info(f"g: {g}")
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
        D[m] = [col.split("_")[-1] for col in coef_dict.keys() if col not in M]
    g = params.g
    index_set = IndexSet(D=D, M=M, K=K)
    constant = Constant(beta=beta, beta0=beta0, g=g, P=P)
    logger.info(f"D: {D}")
    logger.info(f"beta: {beta}")
    logger.info(f"beta0: {beta0}")
    logger.info(f"g: {g}")
    return index_set, constant
