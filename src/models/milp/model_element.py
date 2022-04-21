from dataclasses import dataclass
import itertools
from typing import Dict, List, Tuple

import pulp


@dataclass(frozen=True)
class IndexSet:
    M: List[int]
    K: List[int]

    @property
    def N(self) -> List[int]:
        _N = list(itertools.product(range(len(self.K)), repeat=len(self.M)))
        return [i for i, _ in enumerate(_N)]


@dataclass(frozen=True)
class Constant:
    P: Dict[Tuple[int, int], int]
    Q: Dict[Tuple[int, int], float]
    c: Dict[int, int]


class Variable:
    def __init__(self, index_set: IndexSet, constant: Constant):
        self.index_set = index_set
        self.constant = constant

        self.q: Dict[int, pulp.LpVariable] = dict()
        self.r: Dict[int, pulp.LpVariable] = dict()
        self.x: Dict[int, pulp.LpVariable] = dict()

        self._set_variables()

    def _set_variables(self):
        for m in self.index_set.M:
            self.q[m] = pulp.LpVariable(f"q[{m}]", 0)
            self.r[m] = pulp.LpVariable(f"r[{m}]", 0)

        for n in self.index_set.N:
            self.x[n] = pulp.LpVariable(f"x[{n}]", 0, 1, "Binary")


class ObjectiveFunctionMixin:
    index_set: IndexSet
    constant: Constant
    variable: Variable
    problem: pulp.LpProblem

    def set_objective_function(self) -> None:
        objective_function = 0
        for m in self.index_set.M:
            objective_function += self.variable.r[m] - self.constant.c[m] * self.variable.q[m]
        self.problem += objective_function, "Objective Function"


class ConstraintsMixin:
    index_set: IndexSet
    constant: Constant
    variable: Variable
    problem: pulp.LpProblem

    def set_constraints(self) -> None:
        self._set_quantity_constraints()
        self._set_revenue_constraints()
        self._set_assignment_constraints()

    def _set_quantity_constraints(self) -> None:
        for m in self.index_set.M:
            self.problem += self.variable.q[m] == pulp.lpSum([self.constant.Q[m, n] * self.variable.x[n] for n in self.index_set.N])

    def _set_revenue_constraints(self) -> None:
        for m in self.index_set.M:
            self.problem += self.variable.r[m] == pulp.lpSum([self.constant.P[m, n] * self.constant.P[m, n] * self.variable.x[n] for n in self.index_set.N])

    def _set_assignment_constraints(self) -> None:
        self.problem += pulp.lpSum([self.variable.x[n] for n in self.index_set.N]) == 1
