from __future__ import annotations

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
        num_iteration: int = 10,
        **kwargs,
    ):
        super().__init__(model)
        self.solver = solver
        self.TimeLimit = TimeLimit
        self.num_iteration = num_iteration
        self.result = None
        self.xs = []

    def run(self) -> None:
        """緩和問題のxの値に基づいてランダムサンプリングして丸め込む"""
        M = self.model.index_set.M
        K = self.model.index_set.K

        start = time.time()
        for i in range(self.num_iteration):
            # ランダムに初期解を作成
            x_init = dict()
            for m in M:
                np.random.seed(int(m) + (i + 100))
                selected_k = np.random.choice(K)
                for k in K:
                    if k == selected_k:
                        x_init[m, k] = 1
                    else:
                        x_init[m, k] = 0

            # 商品の価格を1つずつ最適化
            self.coordinate_descent(x_init)

            logger.info(f"num_iteration: {i}, best_obj: {self.result.objective}")
        elapsed_time = time.time() - start

        # 最終的な計算時間などを格納
        self.result.calculation_time = elapsed_time
        self.result.index_set = self.model.index_set
        self.result.constant = self.model.constant

    def coordinate_descent(self, x_init: dict[tuple[str, int], int]) -> None:
        """初期解から商品価格を1つずつ最適化"""
        M = self.model.index_set.M
        K = self.model.index_set.K

        z_init = self.calc_z(x=x_init)
        best_obj = self.calc_obj(x=x_init, z=z_init)
        best_x = x_init.copy()
        for m in M:
            # 商品mのKパターンの価格を試す
            for i, _ in enumerate(K):
                x_m = dict()
                for k in K:
                    if k == i:
                        x_m[m, k] = 1
                    else:
                        x_m[m, k] = 0
                assert sum(x_m.values()) == 1
                x = best_x.copy()
                x.update(x_m)
                z = self.calc_z(x=x)
                obj = self.calc_obj(x=x, z=z)
                if obj > best_obj:
                    best_obj = obj
                    best_x = x
            logger.info(f"product: {m}, obj: {obj}, best_obj: {best_obj}, x: {x}")
        opt_prices = get_opt_prices(x=best_x, P=self.model.constant.P)
        self.result = OptResult(
            calculation_time=0,
            objective=best_obj,
            opt_prices=opt_prices,
        )
