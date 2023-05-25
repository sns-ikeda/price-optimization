from __future__ import annotations

import random
import time
from typing import Optional, TypeVar

import numpy as np
from logzero import logger

from src.optimize.algorithms.base_algorithm import BaseSearchAlgorithm
from src.optimize.processing import get_opt_prices
from src.optimize.result import OptResult
from src.utils.paths import ALGO_DIR

Model = TypeVar("Model")


def calc_z(x: dict[tuple[str, int], int], a, b, phi, g, M, K, D, TL) -> dict[tuple[str, int], int]:
    """xからzを算出"""
    z, mt_z = dict(), dict()
    # xの値が1となるmとk
    mk_x: dict[str, int] = {m: k for m in M for k in K if x[m, k] >= 0.99}
    assert len(mk_x) == len(M)

    for m in M:
        t = 0
        while True:
            linear_sum = 0
            linear_sum_m = sum(a[m, mp, t] * phi[mp, mk_x[mp]] for mp in M)
            linear_sum_d = sum(a[m, d, t] * g[m, d] for d in D[m])
            linear_sum = linear_sum_m + linear_sum_d
            if linear_sum < b[m, t]:
                # 左に分岐
                t = t * 2 + 1
            else:
                # 右に分岐
                t = t * 2 + 2
            if t in TL[m]:
                break
            if t > 1000:
                raise Exception("Infinite Loop Error")
        mt_z[m] = t
        z.update({(m, t): 1 if t == mt_z[m] else 0 for t in TL[m]})
    return z

def calc_obj(x: dict[tuple[str, int], int], z: dict[tuple[str, int], int], M, K, TL, P, beta, beta0, phi, g, D) -> float:
    """x, zから目的関数を計算"""

    # xの値が1となるmとk
    mk_x: dict[str, int] = {m: k for m in M for k in K if x[m, k] >= 0.99}

    # zの値が1となるmとt
    mt_z: dict[str, int] = {m: t for m in M for t in TL[m] if z[m, t] >= 0.99}
    assert len(mt_z) == len(mk_x) == len(M)
    assert mk_x.keys() == mt_z.keys()

    p = np.array([P[(m, k)] for m, k in mk_x.items()])
    q = np.array(
        [
            sum(beta[m, mp, t] * phi[mp, mk_x[mp]] for mp in mt_z.keys())
            + sum(beta[m, d, t] * g[m, d] for d in D[m])
            + beta0[m, t]
            for m, t in mt_z.items()
        ]
    )
    return np.dot(p, q)

def get_opt_prices(
    x: dict[tuple[str, int], float], P: dict[tuple[str, int], float]
) -> dict[str, float]:
    """解xから各商品の最適価格を算出"""
    opt_prices = dict()
    for k_tuple, v in x.items():
        if round(v, 2) == 1.0:
            item = k_tuple[0]
            opt_prices[item] = P[k_tuple]
    # 商品順にソート
    opt_prices = dict(sorted(opt_prices.items()))
    return opt_prices

def generate_x_init(M: list[str], K: list[int], seed: int = 0) -> dict[tuple[str, int], int]:
    """初期解を生成"""
    x_init = dict()
    for m in M:
        np.random.seed(int(m) + (seed + 100))
        selected_k = np.random.choice(K)
        x_init.update({(m, k): int(k == selected_k) for k in K})
    return x_init

def coordinate_descent(M, K, P, x_init: dict[tuple[str, int], int], a, b, phi, g, D, TL, beta, beta0, threshold: int = 10) -> None:
    """商品をランダムに1つ選び最適化"""
    z_init = calc_z(x=x_init, a=a, b=b, phi=phi, g=g, M=M, K=K, D=D, TL=TL)
    best_obj = calc_obj(x=x_init, z=z_init, M=M, K=K, TL=TL, P=P, beta=beta, beta0=beta0, phi=phi, g=g, D=D)
    best_x = x_init.copy()

    break_count, total_count = 0, 0
    while True:
        total_count += 1
        m = random.choice(M)
        # 商品mのKパターンの価格を試す
        x_m = np.zeros((len(K),))
        for k in K:
            x_m[k] = 1
            x = best_x.copy()
            x.update({(m, k): x_m[k] for k in K})
            z = calc_z(x=x, a=a, b=b, phi=phi, g=g, M=M, K=K, D=D, TL=TL)
            obj = calc_obj(x=x, z=z, M=M, K=K, TL=TL, P=P, beta=beta, beta0=beta0, phi=phi, g=g, D=D)
            if obj > best_obj:
                best_obj = obj
                best_x = x
                break_count = 0
        # logger.info(f"best_obj: {best_obj}, break_count: {break_count}, total_count: {total_count}")
        break_count += 1

        if break_count >= threshold:
            # logger.info(f"break_count: {break_count}, total_count: {total_count}")
            break
        # logger.info(f"product: {m}, best_obj_m: {best_obj}")
    opt_prices = get_opt_prices(x=best_x, P=P)
    return best_obj, opt_prices

class CoordinateDescent(BaseSearchAlgorithm):
    def __init__(
        self,
        model: Model,
        solver: str = "Cbc",
        TimeLimit: Optional[int] = None,
        num_iteration: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(model)
        self.solver = solver
        self.TimeLimit = TimeLimit
        if num_iteration is None:
            self.num_iteration = len(model.index_set.M) * len(model.index_set.K)
        else:
            self.num_iteration = num_iteration
        self.result = None
        self.xs = []

    def run(self, use_julia: bool = False) -> None:
        """緩和問題のxの値に基づいてランダムサンプリングして丸め込む"""
        M = self.model.index_set.M
        K = self.model.index_set.K
        P = self.model.constant.P
        a = self.model.constant.a
        b = self.model.constant.b
        phi = self.model.constant.P
        g = self.model.constant.g
        D = self.model.index_set.D
        TL = self.model.index_set.TL
        beta = self.model.constant.beta
        beta0 = self.model.constant.beta0

        best_obj = 0
        opt_prices = dict()

        start = time.time()
        for i in range(self.num_iteration):
            # ランダムに初期解を作成
            x_init = generate_x_init(M=M, K=K, seed=i)

            # 商品の価格を1つずつ最適化
            if use_julia:
                import julia
                julia.install()
                from julia import Main
                Main.include(str(ALGO_DIR / 'coordinate_descent.jl'))
                _best_obj, _opt_prices = Main.coordinate_descent(M=M, K=K, P=P, x_init=x_init, a=a, b=b, phi=phi, g=g, D=D, TL=TL, beta=beta, beta0=beta0)
            else:
                _best_obj, _opt_prices = coordinate_descent(M=M, K=K, P=P, x_init=x_init, a=a, b=b, phi=phi, g=g, D=D, TL=TL, beta=beta, beta0=beta0)
            if _best_obj > best_obj:
                best_obj = _best_obj
                opt_prices = _opt_prices

            logger.info(f"num_iteration: {i}, _best_obj: {_best_obj}, best_obj: {best_obj}")
        elapsed_time = time.time() - start

        # 最終的な結果を格納
        self.result = OptResult(
            calculation_time=elapsed_time,
            objective=best_obj,
            opt_prices=opt_prices,
            index_set=self.model.index_set,
            constant=self.model.constant,
        )
