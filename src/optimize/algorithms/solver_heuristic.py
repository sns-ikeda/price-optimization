from typing import Optional

from src.optimize.algorithms.base_algorithm import BaseAlgorithm


class SolverHeuristics(BaseAlgorithm):
    def __init__(
        self,
        model,
        solver: str = "Cbc",
        TimeLimit: Optional[int] = None,
        NoRelHeurTime: int = 0,
        MIPFocus: int = 0,
    ):
        super().__init__(model)
        self.solver = solver
        self.TimeLimit = TimeLimit
        self.NoRelHeurTime = NoRelHeurTime
        self.MIPFocus = MIPFocus

    def run(self) -> None:
        self.model.solve(
            solver=self.solver,
            TimeLimit=self.TimeLimit,
            NoRelHeurTime=self.NoRelHeurTime,
            MIPFocus=self.MIPFocus,
        )
        self.result = self.model.result
