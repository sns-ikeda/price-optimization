from __future__ import annotations

import random
import time
from typing import Optional, TypeVar

from logzero import logger

from src.optimize.algorithms.base_algorithm import BaseSearchAlgorithm
from src.optimize.processing import get_opt_prices
from src.optimize.result import OptResult

Model = TypeVar("Model")


class MiloRelax(BaseSearchAlgorithm):
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
        self._result = None

    def run(self) -> None:
        self.model.relax = True
        self.model.solve(solver=self.solver, TimeLimit=self.TimeLimit)
        self._result = self.model.result
        self.randomized_search_rounding()

    def randomized_search_rounding(self) -> None:
        """Random sampling and rounding based on the value of x in the relaxation problem"""
        M = self.model.index_set.M
        K = self.model.index_set.K
        best_obj = 0
        start = time.time()
        for i in range(self.num_iteration):
            selected_m_k = []  # Keep selected m and k tuples
            for m in M:
                probabilities = [self._result.variable.x[m, k] for k in K]
                selected_k = random.choices(K, weights=probabilities, k=1)[0]
                selected_m_k.append((m, selected_k))
            # make x from selected_m_k
            x = dict()
            for m in M:
                for k in K:
                    if (m, k) in selected_m_k:
                        x[m, k] = 1
                    else:
                        x[m, k] = 0
            z = self.calc_z(x=x)

            # compute the objective function and update if best
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

        # store results
        self.result.calculation_time = elapsed_time + self._result.calculation_time
        self.result.index_set = self.model.index_set
        self.result.constant = self.model.constant
