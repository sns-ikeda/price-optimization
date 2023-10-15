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
    b: dict[tuple[str, int], float],
    M: list[str],
    K: list[int],
    D: dict[str, list[str]],
    TL: dict[str, list[int]],
    x_matrix: np.ndarray,
    a_matrix: np.ndarray,
    g_matrix: np.ndarray,
    phi_matrix: np.ndarray,
) -> dict[tuple[str, int], int]:
    """Calculate z from x"""
    ones_k = np.ones(len(K))
    z_matrix = np.zeros((len(M), len(TL[M[0]])))

    for i, m in enumerate(M):
        t = 0
        if len(TL[m]) > 1:
            while True:
                linear_sum_m = np.dot(
                    a_matrix[i, :, t], np.dot(np.multiply(phi_matrix, x_matrix), ones_k)
                )
                linear_sum_d = 0
                for j, _ in enumerate(D[m]):
                    linear_sum_d += a_matrix[i, len(M) + j, t], g_matrix[i, j]
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
        z_matrix[i, t - min(TL[m])] = 1
    return z_matrix


def calc_obj(
    M: list[str],
    K: list[int],
    D: dict[str, list[str]],
    P_matrix: np.ndarray,
    x_matrix: np.ndarray,
    z_matrix: np.ndarray,
    phi_matrix: np.ndarray,
    g_matrix: np.ndarray,
    beta_matrix: np.ndarray,
    beta0_matrix: np.ndarray,
) -> float:
    """Calculate objective function value from x and z"""
    p = np.dot(P_matrix, x_matrix.T)[0]
    ones_k = np.ones(len(K))
    q = np.zeros(len(M))

    for i, m in enumerate(M):
        q[i] = np.dot(
            np.dot(beta_matrix[i, :, :], z_matrix[i, :].T),
            np.dot(np.multiply(phi_matrix, x_matrix), ones_k),
        ) + np.dot(beta0_matrix[i, :], z_matrix[i, :])

        for j, _ in enumerate(D[m]):
            q[i] += np.dot(beta_matrix[i, len(M) + j, :], z_matrix[i, :].T) * g_matrix[i, j]

    return np.dot(p, q)


def get_opt_prices(
    x_matrix: np.ndarray, P: dict[tuple[str, int], float], M: list[str], K: list[int]
) -> dict[str, float]:
    """Calculate optimal prices from x"""
    opt_prices = dict()
    for i, m in enumerate(M):
        for k in K:
            if x_matrix[i, k] >= 0.99:
                opt_prices[m] = P[m, k]
    # 商品順にソート
    opt_prices = dict(sorted(opt_prices.items()))
    return opt_prices


def generate_x_init(M: list[str], K: list[int], seed: int = 0) -> np.ndarray:
    """generate x_init_matrix"""
    x_init_matrix = np.zeros((len(M), len(K)))
    for i, m in enumerate(M):
        np.random.seed(int(i) + (seed + 100))
        selected_k = np.random.choice(K)
        x_init_matrix[i, selected_k] = 1
    return x_init_matrix


def coordinate_descent(
    M: list[str],
    K: list[int],
    P: dict[tuple[str, int], float],
    b: dict[tuple[str, int], float],
    D: dict[str, list[str]],
    TL: dict[str, list[int]],
    x_init_matrix: np.ndarray,
    a_matrix: np.ndarray,
    phi_matrix: np.ndarray,
    g_matrix: np.ndarray,
    P_matrix: np.ndarray,
    beta_matrix: np.ndarray,
    beta0_matrix: np.ndarray,
    threshold: int = 10,
    randomized: bool = False,
) -> None:
    """Coordinate descent algorithm"""
    z_init_matrix = calc_z(
        b=b,
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
        K=K,
        D=D,
        P_matrix=P_matrix,
        x_matrix=x_init_matrix,
        z_matrix=z_init_matrix,
        g_matrix=g_matrix,
        phi_matrix=phi_matrix,
        beta_matrix=beta_matrix,
        beta0_matrix=beta0_matrix,
    )
    best_x_matrix = x_init_matrix.copy()

    # randomized coordinate descent
    break_count, total_count = 0, 0
    if randomized:
        while True:
            total_count += 1
            m = random.choice(M)
            # Try the K-pattern price of item m
            for k in K:
                x_m = np.zeros((len(K),))
                x_m[k] = 1
                x_matrix = best_x_matrix.copy()
                x_matrix[M.index(m), :] = x_m
                z_matrix = calc_z(
                    b=b,
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
                    K=K,
                    D=D,
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
                    best_x_matrix = x_matrix
                    break_count = 0
            # logger.info(f"best_obj: {best_obj}, break_count: {break_count}, total_count: {total_count}")
            break_count += 1

            if break_count >= threshold:
                # logger.info(f"break_count: {break_count}, total_count: {total_count}")
                break
            # logger.info(f"product: {m}, best_obj_m: {best_obj}")

    # cyclic coordinate descent
    else:
        while True:
            total_count += 1
            M_shuffled = random.sample(M, len(M))
            for m in M_shuffled:
                # 商品mのKパターンの価格を試す
                for k in K:
                    x_m = np.zeros((len(K),))
                    x_m[k] = 1
                    x_matrix = best_x_matrix.copy()
                    x_matrix[M.index(m), :] = x_m
                    z_matrix = calc_z(
                        b=b,
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
                        K=K,
                        D=D,
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
                        best_x_matrix = x_matrix
                        break_count = 0
                break_count += 1
                if break_count >= threshold:
                    break
            if break_count >= threshold:
                break

    opt_prices = get_opt_prices(x_matrix=best_x_matrix, P=P, M=M, K=K)
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
        """Run coordinate descent algorithm"""
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
        g_matrix = np.zeros((len(M), len(D[M[0]])))
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
            for kd, d in enumerate(D[m]):
                g_matrix[i, kd] = g[m, d]
            for k in K:
                phi_matrix[i, k] = phi[m, k]
                P_matrix[i, k] = P[m, k]

        start = time.time()
        for i in range(self.num_iteration):
            # generate x_init_matrix
            x_init_matrix = generate_x_init(M=M, K=K, seed=i)
            _best_obj, _opt_prices = coordinate_descent(
                M=M,
                K=K,
                P=P,
                b=b,
                D=D,
                TL=TL,
                x_init_matrix=x_init_matrix,
                a_matrix=a_matrix,
                phi_matrix=phi_matrix,
                g_matrix=g_matrix,
                P_matrix=P_matrix,
                beta_matrix=beta_matrix,
                beta0_matrix=beta0_matrix,
            )
            if _best_obj > best_obj:
                best_obj = _best_obj
                opt_prices = _opt_prices

            logger.info(f"num_iteration: {i}, _best_obj: {_best_obj}, best_obj: {best_obj}")
        elapsed_time = time.time() - start

        # store results
        self.result = OptResult(
            calculation_time=elapsed_time,
            objective=best_obj,
            opt_prices=opt_prices,
            index_set=self.model.index_set,
            constant=self.model.constant,
        )
