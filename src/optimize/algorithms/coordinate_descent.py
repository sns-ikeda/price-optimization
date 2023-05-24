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
            self.num_iteration = (
                len(model.index_set.M) * len(model.index_set.K)
            )
        else:
            self.num_iteration = num_iteration
        self.result = None
        self.xs = []

    def run(self) -> None:
        """緩和問題のxの値に基づいてランダムサンプリングして丸め込む"""
        M = self.model.index_set.M
        K = self.model.index_set.K
        best_obj = 0
        opt_prices = dict()

        start = time.time()
        for i in range(self.num_iteration):
            # ランダムに初期解を作成
            x_init = self.generate_x_init(M=M, K=K, seed=i)

            # 商品の価格を1つずつ最適化
            _best_obj, _opt_prices = self.coordinate_descent(x_init)
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

    @staticmethod
    def generate_x_init(M: list[str], K: list[int], seed: int = 0) -> dict[tuple[str, int], int]:
        """初期解を生成"""
        x_init = dict()
        for m in M:
            np.random.seed(int(m) + (seed + 100))
            selected_k = np.random.choice(K)
            x_init.update({(m, k): int(k == selected_k) for k in K})
        return x_init

    def coordinate_descent(self, x_init: dict[tuple[str, int], int], threshold: int = 10) -> None:
        """商品をランダムに1つ選び最適化"""
        M = self.model.index_set.M
        K = self.model.index_set.K

        z_init = self.calc_z(x=x_init)
        best_obj = self.calc_obj(x=x_init, z=z_init)
        best_x = x_init.copy()

        break_count, total_count = 0, 0
        while True:
            total_count += 1
            m = random.choice(M)
            # 商品mのKパターンの価格を試す
            x_m = np.zeros((len(K),))
            objs = []
            for k in K:
                x_m[k] = 1
                x = best_x.copy()
                x.update({(m, k): x_m[k] for k in K})
                z = self.calc_z(x=x)
                obj = self.calc_obj(x=x, z=z)
                objs.append(obj)
                if obj > best_obj:
                    best_obj = obj
                    best_x = x
                    break_count = 0
            # logger.info(f"best_obj: {best_obj}, objs: {objs}")
            if best_obj >= max(objs):
                break_count += 1

            if break_count >= threshold:
                # logger.info(f"break_count: {break_count}, total_count: {total_count}")
                break
            logger.info(f"product: {m}, best_obj_m: {best_obj}")
        opt_prices = get_opt_prices(x=best_x, P=self.model.constant.P)
        return best_obj, opt_prices

