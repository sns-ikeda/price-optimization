from typing import Optional

from src.optimize.algorithms.base_algorithm import BaseAlgorithm


class SolverNaive(BaseAlgorithm):
    def __init__(self, model, solver: str = "Cbc", TimeLimit: Optional[int] = None):
        super().__init__(model)
        self.solver = solver
        self.TimeLimit = TimeLimit

    def run(self) -> None:
        self.model.solve(solver=self.solver, TimeLimit=self.TimeLimit)
        self.result = self.model.result
