from src.optimize.algorithms.multi_local_search import MultiLocalSearch
from src.optimize.algorithms.solver_heuristic import SolverHeuristics
from src.optimize.algorithms.solver_naive import SolverNaive

ALGORITHMS = {
    "solver_naive": SolverNaive,
    "solver_heuristics": SolverHeuristics,
    "multi_start_local_search": MultiLocalSearch,
}
