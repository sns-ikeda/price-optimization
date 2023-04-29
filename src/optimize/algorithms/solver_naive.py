from typing import Optional, TypeVar

from src.optimize.algorithms.base_algorithm import BaseAlgorithm

Model = TypeVar("Model")


class SolverNaive(BaseAlgorithm):
    def __init__(self, model: Model, solver: str = "Cbc", TimeLimit: Optional[int] = None):
        super().__init__(model)
        self.solver = solver
        self.TimeLimit = TimeLimit

    def run(self) -> None:
        self.model.solve(solver=self.solver, TimeLimit=self.TimeLimit)
        self.result = self.model.result
