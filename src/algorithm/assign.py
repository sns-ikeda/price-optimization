from src.algorithm.multi_local_search import MultiLocalSearch
from src.algorithm.solver_heuristic import SolverHeuristic
from src.algorithm.solver_naive import SolverNaive
from src.models.po_L.params import Parameter


class AlgorithmAssigner:
    def __init__(self, params: Parameter, method: str):
        self.params = params
        self.method = method
        self.algorithm = None
        self.result = None
        self._assign(method=self.method)

    def _assign(self, method: str) -> None:
        if method == "solver_naive":
            self.algorithm = SolverNaive(self.params)
        elif method == "solver_heuristic":
            self.algorithm = SolverHeuristic(self.params)
        elif method == "multi_start_local_search":
            self.algorithm = MultiLocalSearch(self.params)

    def run(self) -> None:
        if self.algorithm is not None:
            self.algorithm.run()
            self.result = self.algorithm.result
