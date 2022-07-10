from src.optimization.algorithms.multi_local_search import MultiLocalSearch
from src.optimization.algorithms.solver_heuristic import SolverHeuristics
from src.optimization.algorithms.solver_naive import SolverNaive

ALGORITHMS = {
    "solver_naive": SolverNaive,
    "solver_heuristics": SolverHeuristics,
    "multi_start_local_search": MultiLocalSearch,
}
