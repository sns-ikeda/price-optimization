from src.algorithm.base_algorithm import BaseAlgorithm
from src.models.po_L.make_input import make_sample_input
from src.models.po_L.model import Model


class SolverNaive(BaseAlgorithm):
    def run(self) -> None:
        index_set, constant = make_sample_input(params=self.params)
        model = Model(index_set=index_set, constant=constant)
        model.solve(solver=self.params.solver, TimeLimit=self.params.TimeLimit)
        self.model = model
        self.index_set = index_set
        self.constant = constant
        self.result = model.result
