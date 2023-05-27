from __future__ import annotations

import random
import time
from typing import Optional, TypeVar

import numpy as np
from logzero import logger

from src.optimize.algorithms.base_algorithm import BaseSearchAlgorithm
from src.optimize.processing import get_opt_prices
from src.optimize.result import OptResult

Model = TypeVar("Model")


def calc_z(
    x: dict[tuple[str, int], int],
    a: dict[tuple[str, str, int], float],
    b: dict[tuple[str, int], float],
    phi: dict[tuple[str, int], float],
    g: dict[tuple[str, str], float],
    M: list[str],
    K: list[int],
    D: dict[str, list[str]],
    TL: dict[str, list[int]],
    x_matrix: np.ndarray,
    a_matrix: np.ndarray,
    g_matrix: np.ndarray,
    phi_matrix: np.ndarray,
) -> dict[tuple[str, int], int]:
    """xからzを算出"""
    print("x", x)
    print("x_matrix", x_matrix)
    z, mt_z = dict(), dict()
    # xの値が1となるmとk
    mk_x: dict[str, int] = {m: k for m in M for k in K if x[m, k] >= 0.99}
    _mk_x: dict[str, int] = {m: k for i, m in enumerate(M) for k in K if x_matrix[i, k] >= 0.99}
    print("mk_x", mk_x)
    print("_mk_x", _mk_x)
    assert mk_x == _mk_x
    assert len(mk_x) == len(M)
    ones_k = np.ones(len(K))
    z_matrix = np.zeros((len(M), len(TL[M[0]])))

    for i, m in enumerate(M):
        t = 0
        while True:
            linear_sum_m = sum(a[m, mp, t] * phi[mp, mk_x[mp]] for mp in M)
            _linear_sum_m = np.dot(
                a_matrix[i, :, t], np.dot(np.multiply(phi_matrix, x_matrix), ones_k)
            )
            print("linear_sum_m", linear_sum_m, _linear_sum_m)
            linear_sum_d = sum(a[m, d, t] * g[m, d] for d in D[m])
            _linear_sum_d = 0
            for j, _ in enumerate(D[m]):
                _linear_sum_d += a_matrix[i, len(M) + j, t], g_matrix[i, j]
            print("linear_sum_d", linear_sum_d, _linear_sum_d)
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
        z_matrix[i, t - min(TL[m])] = 1
    return z, z_matrix, mk_x, mt_z


def calc_obj(
    M: list[str],
    P: dict[tuple[str, int], float],
    K: list[int],
    beta: dict[tuple[str, str, int], float],
    beta0: dict[tuple[str, int], float],
    phi: dict[tuple[str, int], float],
    g: dict[tuple[str, str], float],
    D: dict[str, list[str]],
    mk_x: dict[str, int],
    mt_z: dict[str, int],
    P_matrix: np.ndarray,
    x_matrix: np.ndarray,
    z_matrix: np.ndarray,
    phi_matrix: np.ndarray,
    g_matrix: np.ndarray,
    beta_matrix: np.ndarray,
    beta0_matrix: np.ndarray,
) -> float:
    """x, zから目的関数を計算"""

    assert len(mt_z) == len(mk_x) == len(M)
    assert mk_x.keys() == mt_z.keys()

    p = np.array([P[(m, k)] for m, k in mk_x.items()])
    _p = np.dot(P_matrix, x_matrix.T)[0]
    print("P_matrix", P_matrix)
    print("x_matrix.T", x_matrix.T)
    print("p", p, _p[0])
    assert (p == _p).all()
    q = np.array(
        [
            sum(beta[m, mp, t] * phi[mp, mk_x[mp]] for mp in mt_z.keys())
            + sum(beta[m, d, t] * g[m, d] for d in D[m])
            + beta0[m, t]
            for m, t in mt_z.items()
        ]
    )
    ones_k = np.ones(len(K))
    _q = np.zeros(len(M))
    for i, m in enumerate(M):
        _q[i] = np.dot(
            np.dot(beta_matrix[i, :, :], z_matrix[i, :].T),
            np.dot(np.multiply(phi_matrix, x_matrix), ones_k),
        ) + np.dot(beta0_matrix[i, :], z_matrix[i, :])
        for j, _ in enumerate(D[m]):
            _q[i] += np.dot(beta_matrix[i, len(M) + j, :], z_matrix[i, :].T) * g_matrix[i, j]
    print("q", q, _q)
    # assert (q == _q).all()
    return np.dot(p, q)


def get_opt_prices(
    x: dict[tuple[str, int], float], P: dict[tuple[str, int], float], M, K
) -> dict[str, float]:
    """解xから各商品の最適価格を算出"""
    opt_prices = dict()
    try:
        for k_tuple, v in x.items():
            if round(v, 2) == 1.0:
                item = k_tuple[0]
                opt_prices[item] = P[k_tuple]
    except Exception:
        for i, m in enumerate(M):
            for k in K:
                if x[i, k] >= 0.99:
                    opt_prices[m] = P[m, k]
    # 商品順にソート
    opt_prices = dict(sorted(opt_prices.items()))
    return opt_prices


def generate_x_init(M: list[str], K: list[int], seed: int = 0) -> np.ndarray:
    """初期解を生成"""
    x_init_matrix = np.zeros((len(M), len(K)))
    x_init = dict()
    for i, m in enumerate(M):
        np.random.seed(int(i) + (seed + 100))
        selected_k = np.random.choice(K)
        x_init_matrix[i, selected_k] = 1
        x_init.update({(m, k): 1 if k == selected_k else 0 for k in K})
    return x_init, x_init_matrix


def coordinate_descent(
    M: list[str],
    K: list[int],
    P: dict[tuple[str, int], float],
    x_init: dict[tuple[str, int], int],
    a: dict[tuple[str, str, int], float],
    b: dict[tuple[str, int], float],
    phi: dict[tuple[str, int], float],
    g: dict[tuple[str, str], float],
    D: dict[str, list[str]],
    TL: dict[str, list[int]],
    beta: dict[tuple[str, str, int], float],
    beta0: dict[tuple[str, int], float],
    x_init_matrix: np.ndarray,
    a_matrix: np.ndarray,
    phi_matrix: np.ndarray,
    g_matrix: np.ndarray,
    P_matrix: np.ndarray,
    beta_matrix: np.ndarray,
    beta0_matrix: np.ndarray,
    threshold: int = 10,
    randomized: bool = True,
    num_iter: int = 1,
) -> None:
    """商品をランダムに1つ選び最適化"""
    z_init, z_init_matrix, mk_x_init, mt_z_init = calc_z(
        x=x_init,
        a=a,
        b=b,
        phi=phi,
        g=g,
        M=M,
        K=K,
        D=D,
        TL=TL,
        x_matrix=x_init_matrix,
        a_matrix=a_matrix,
        g_matrix=g_matrix,
        phi_matrix=phi_matrix,
    )
    best_obj = calc_obj(
        M=M,
        P=P,
        K=K,
        beta=beta,
        beta0=beta0,
        phi=phi,
        g=g,
        D=D,
        mk_x=mk_x_init,
        mt_z=mt_z_init,
        P_matrix=P_matrix,
        x_matrix=x_init_matrix,
        z_matrix=z_init_matrix,
        g_matrix=g_matrix,
        phi_matrix=phi_matrix,
        beta_matrix=beta_matrix,
        beta0_matrix=beta0_matrix,
    )
    best_x = x_init.copy()
    best_x_matrix = x_init_matrix.copy()
    if randomized:
        break_count, total_count = 0, 0
        while True:
            total_count += 1
            m = random.choice(M)
            # 商品mのKパターンの価格を試す
            for k in K:
                x_m = np.zeros((len(K),))
                x_m[k] = 1
                x = best_x.copy()
                x.update({(m, k): x_m[k] for k in K})
                x_matrix = best_x_matrix.copy()
                x_matrix[M.index(m), :] = x_m
                z, z_matrix, mk_x, mt_z = calc_z(
                    x=x,
                    a=a,
                    b=b,
                    phi=phi,
                    g=g,
                    M=M,
                    K=K,
                    D=D,
                    TL=TL,
                    x_matrix=x_matrix,
                    a_matrix=a_matrix,
                    g_matrix=g_matrix,
                    phi_matrix=phi_matrix,
                )
                obj = calc_obj(
                    M=M,
                    P=P,
                    K=K,
                    beta=beta,
                    beta0=beta0,
                    phi=phi,
                    g=g,
                    D=D,
                    mk_x=mk_x,
                    mt_z=mt_z,
                    P_matrix=P_matrix,
                    x_matrix=x_matrix,
                    z_matrix=z_matrix,
                    phi_matrix=phi_matrix,
                    g_matrix=g_matrix,
                    beta_matrix=beta_matrix,
                    beta0_matrix=beta0_matrix,
                )
                if obj > best_obj:
                    best_obj = obj
                    best_x = x
                    best_x_matrix = x_matrix
                    break_count = 0
            # logger.info(f"best_obj: {best_obj}, break_count: {break_count}, total_count: {total_count}")
            break_count += 1

            if break_count >= threshold:
                # logger.info(f"break_count: {break_count}, total_count: {total_count}")
                break
            # logger.info(f"product: {m}, best_obj_m: {best_obj}")
    else:
        for i in range(num_iter):
            np.random.seed(i)
            np.random.shuffle(M)
            for m in M:
                # 商品mのKパターンの価格を試す
                for i, _ in enumerate(K):
                    x_m = np.zeros((len(K),))
                    x_m[k] = 1
                    x = best_x.copy()
                    x.update({(m, k): x_m[k] for k in K})
                    x_matrix = best_x_matrix.copy()
                    x_matrix[M.index(m), :] = x_m
                    z, mk_x, mt_z = calc_z(
                        x=x,
                        a=a,
                        b=b,
                        phi=phi,
                        g=g,
                        M=M,
                        K=K,
                        D=D,
                        TL=TL,
                        x_matrix=x_matrix,
                        a_matrix=a_matrix,
                        g_matrix=g_matrix,
                        phi_matrix=phi_matrix,
                    )
                    obj = calc_obj(
                        M=M,
                        P=P,
                        K=K,
                        beta=beta,
                        beta0=beta0,
                        phi=phi,
                        g=g,
                        D=D,
                        mk_x=mk_x,
                        mt_z=mt_z,
                        P_matrix=P_matrix,
                        x_matrix=x_matrix,
                        z_matrix=z_matrix,
                        phi_matrix=phi_matrix,
                        g_matrix=g_matrix,
                        beta_matrix=beta_matrix,
                        beta0_matrix=beta0_matrix,
                    )
                    if obj > best_obj:
                        best_obj = obj
                        best_x = x
                        best_x_matrix = x_matrix

    opt_prices = get_opt_prices(x=best_x, P=P, M=M, K=K)
    _opt_prices = get_opt_prices(x=best_x_matrix, P=P, M=M, K=K)
    assert opt_prices == _opt_prices
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

    def run(self) -> None:
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

        TB = [t for t in range(min(TL[M[0]]))]
        a_matrix = np.zeros((len(M), len(M), len(TB)))
        g_matrix = np.zeros((len(D[M[0]]), len(D[M[0]])))
        phi_matrix = np.zeros((len(M), len(K)))
        P_matrix = np.zeros((len(M), len(K)))
        beta_matrix = np.zeros((len(M), len(M), len(TL[M[0]])))
        beta0_matrix = np.zeros((len(M), len(TL[M[0]])))

        for i, m in enumerate(M):
            for j, mp in enumerate(M):
                for t in TB:
                    a_matrix[i, j, t] = a[m, mp, t]
                for kt, t_ in enumerate(TL[m]):
                    beta_matrix[i, j, kt] = beta[m, mp, t_]
                    beta0_matrix[i, kt] = beta0[m, t_]

            for k in K:
                phi_matrix[i, k] = phi[m, k]
                P_matrix[i, k] = P[m, k]

        start = time.time()
        for i in range(self.num_iteration):
            # ランダムに初期解を作成
            x_init, x_init_matrix = generate_x_init(M=M, K=K, seed=i)
            _best_obj, _opt_prices = coordinate_descent(
                M=M,
                K=K,
                P=P,
                x_init=x_init,
                a=a,
                b=b,
                phi=phi,
                g=g,
                D=D,
                TL=TL,
                beta=beta,
                beta0=beta0,
                x_init_matrix=x_init_matrix,
                a_matrix=a_matrix,
                g_matrix=g_matrix,
                phi_matrix=phi_matrix,
                P_matrix=P_matrix,
                beta_matrix=beta_matrix,
                beta0_matrix=beta0_matrix,
            )
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
