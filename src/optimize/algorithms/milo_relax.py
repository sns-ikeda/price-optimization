from __future__ import annotations

import random
import time
from typing import Optional, TypeVar

import numpy as np
from logzero import logger

from src.optimize.algorithms.base_algorithm import BaseAlgorithm
from src.optimize.processing import get_opt_prices
from src.optimize.result import OptResult

Model = TypeVar("Model")


class MiloRelax(BaseAlgorithm):
    def __init__(
        self,
        model: Model,
        solver: str = "Cbc",
        TimeLimit: Optional[int] = None,
        num_iteration: int = 10,
        **kwargs,
    ):
        super().__init__(model)
        self.solver = solver
        self.TimeLimit = TimeLimit
        self.num_iteration = num_iteration
        self._result = None

    def run(self) -> None:
        self.model.relax = True
        self.model.solve(solver=self.solver, TimeLimit=self.TimeLimit)
        self._result = self.model.result
        self.randomized_search_rounding()

    def randomized_search_rounding(self) -> None:
        """緩和問題のxの値に基づいてランダムサンプリングして丸め込む"""
        M = self.model.index_set.M
        K = self.model.index_set.K
        best_obj = 0
        start = time.time()
        for i in range(self.num_iteration):
            selected_m_k = []  # 選択されたmとkのタプルを保持
            for m in M:
                probabilities = [self._result.variable.x[m, k] for k in K]
                selected_k = random.choices(K, weights=probabilities, k=1)[0]
                selected_m_k.append((m, selected_k))
            # 解を作成
            x = dict()
            for m in M:
                for k in K:
                    if (m, k) in selected_m_k:
                        x[m, k] = 1
                    else:
                        x[m, k] = 0
            z = self.calc_z(x=x)

            # 目的関数を計算し，最も良ければ更新
            obj = self.calc_obj(x=x, z=z)
            if obj > best_obj:
                best_obj = obj
                opt_prices = get_opt_prices(x=x, P=self.model.constant.P)
                self.result = OptResult(
                    calculation_time=0,
                    objective=best_obj,
                    opt_prices=opt_prices,
                )
            logger.info(f"num_iteration: {i}, obj: {obj}, best_obj: {best_obj}")
        elapsed_time = time.time() - start

        # 最終的な計算時間などを格納
        self.result.calculation_time = elapsed_time + self._result.calculation_time
        self.result.index_set = self.model.index_set
        self.result.constant = self.model.constant

    def calc_z(self, x: dict[tuple[str, int], int]) -> dict[tuple[str, int], int]:
        """xからzを算出"""
        z, mt_z = dict(), dict()
        a = self.model.constant.a
        b = self.model.constant.b
        phi = self.model.constant.P
        g = self.model.constant.g
        M = self.model.index_set.M
        D = self.model.index_set.D
        TL = self.model.index_set.TL

        # xの値が1となるmとk
        mk_x: dict[str, int] = {mk[0]: mk[1] for mk, value in x.items() if value >= 0.99}
        assert len(mk_x) == len(M)

        for m in M:
            t = 0
            while True:
                linear_sum = 0
                for mp in M:
                    k = mk_x[mp]
                    linear_sum += a[m, mp, t] * phi[mp, k]
                for d in D[m]:
                    linear_sum += a[m, d, t] * g[m, d]
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
            for t in TL[m]:
                if t == mt_z[m]:
                    z[m, t] = 1
                else:
                    z[m, t] = 0
        return z

    def calc_obj(self, x: dict[tuple[str, int], int], z: dict[tuple[str, int], int]) -> float:
        """x, zから目的関数を計算"""
        # xの値が1となるmとk
        mk_x = {mk[0]: mk[1] for mk, value in x.items() if value >= 0.99}

        # zの値が1となるmとt
        print("z", z)
        mt_z = {mt[0]: mt[1] for mt, value in z.items() if value >= 0.99}
        assert len(mt_z) == len(mk_x)

        P = self.model.constant.P
        beta = self.model.constant.beta
        beta0 = self.model.constant.beta0
        phi = self.model.constant.P
        g = self.model.constant.g
        D = self.model.index_set.D

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
