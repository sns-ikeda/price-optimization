import itertools
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pulp

from src.utils.paths import DATA_DIR


@dataclass(frozen=True)
class IndexSet:
    D: Dict[int, List[int]]
    D_: Dict[int, List[int]]
    M: List[int]
    K: List[int]
    TL: Dict[int, List[int]]
    L: Dict[Tuple[int, int], List[int]]
    R: Dict[Tuple[int, int], List[int]]


@dataclass(frozen=True)
class Constant:
    beta: Dict[Tuple[int, int, int], float]
    phi: Dict[Tuple[int, int, int], float]
    epsilon: Dict[int, float]
    epsilon_max: Dict[int, float]
    a: Dict[Tuple[int, int, int], float]
    b: Dict[Tuple[int, int], float]
    g: Dict[Tuple[int, int], float]
    P: Dict[Tuple[int, int], float]


class Variable:
    def __init__(self, index_set: IndexSet, constant: Constant):
        self.index_set = index_set
        self.constant = constant
        self.u: Dict[Tuple[int, int, int, int], pulp.LpVariable] = dict()
        self.v: Dict[Tuple[int, int, int, int], pulp.LpVariable] = dict()
        self.x: Dict[Tuple[int, int], pulp.LpVariable] = dict()
        self.z: Dict[Tuple[int, int], pulp.LpVariable] = dict()

        self._set_variables()

    def _set_variables(self) -> None:
        for m in self.index_set.M:
            for mp, k, kp, t in itertools.product(
                self.index_set.M, self.index_set.K, self.index_set.K, self.index_set.TL[m]
            ):
                self.u[m, mp, k, kp, t] = pulp.LpVariable(f"u[{m}_{mp}_{k}_{kp}_{t}]", cat="Binary")

            for mp, k, t in itertools.product(
                self.index_set.M, self.index_set.K, self.index_set.TL[m]
            ):
                self.v[m, mp, k, t] = pulp.LpVariable(f"v[{m}_{mp}_{k}_{t}]", cat="Binary")

            for k in self.index_set.K:
                self.x[m, k] = pulp.LpVariable(f"x[{m}_{k}]", cat="Binary")

            for t in self.index_set.TL[m]:
                self.z[m, t] = pulp.LpVariable(f"z[{m}_{t}]", cat="Binary")

    def to_value(self):
        pass


class ObjectiveFunctionMixin:
    index_set: IndexSet
    constant: Constant
    variable: Variable
    problem: pulp.LpProblem

    def set_objective_function(self) -> None:
        objective_function = 0
        for m in self.index_set.M:
            for mp, k, kp, t in itertools.product(
                self.index_set.M, self.index_set.K, self.index_set.K, self.index_set.TL[m]
            ):
                objective_function += (
                    self.constant.beta[m, mp, t]
                    * self.constant.phi[m, mp, k]
                    * self.constant.P[m, k]
                    * self.variable.u[m, mp, k, kp, t]
                )

            for k, t, d in itertools.product(
                self.index_set.K, self.index_set.TL[m], self.index_set.D[m] + self.index_set.D_[m]
            ):
                objective_function += (
                    self.constant.beta[m, d, t]
                    * self.constant.g[m, d]
                    * self.variable.v[m, m, k, t]
                )

        self.problem += objective_function, "Objective Function"


class ConstraintsMixin:
    index_set: IndexSet
    constant: Constant
    variable: Variable
    problem: pulp.LpProblem

    def set_constraints(self) -> None:
        self._set_branch_constraints()
        self._set_leafnode_constraints()
        self._set_price_constraints()
        self._set_xxz2u_constraints()
        self._set_xz2v_constraints()

    def _set_branch_constraints(self) -> None:
        for m in self.index_set.M:
            for t in self.index_set.TL[m]:
                for tp in self.index_set.R[m, t]:
                    self.problem += (
                        pulp.lpSum(
                            self.constant.a[m, mp, tp]
                            * self.constant.phi[m, mp, k]
                            * self.variable.x[mp, k]
                            for mp, k in itertools.product(self.index_set.M, self.index_set.K)
                        )
                        + pulp.lpSum(
                            self.constant.a[m, d, tp] * self.constant.g[m, d]
                            for d in self.index_set.D[m]
                        )
                        >= self.constant.b[m, tp] + self.variable.z[m, t] - 1
                    )
                for tp in self.index_set.L[m, t]:
                    self.problem += (
                        pulp.lpSum(
                            self.constant.a[m, mp, tp]
                            * self.constant.phi[m, mp, k]
                            * self.variable.x[mp, k]
                            for mp, k in itertools.product(self.index_set.M, self.index_set.K)
                        )
                        + pulp.lpSum(
                            self.constant.a[m, d, tp] * self.constant.g[m, d]
                            for d in self.index_set.D[m]
                        )
                        <= self.constant.b[m, tp]
                        - (1 + self.constant.epsilon_max[m]) * self.variable.z[m, t]
                        + 1
                        + self.constant.epsilon_max[m]
                        - self.constant.epsilon[m]
                    )

    def _set_leafnode_constraints(self) -> None:
        for m in self.index_set.M:
            self.problem += pulp.lpSum(self.variable.z[m, t] for t in self.index_set.TL[m]) == 1

    def _set_price_constraints(self) -> None:
        for m in self.index_set.M:
            self.problem += pulp.lpSum(self.variable.x[m, k] for k in self.index_set.K) == 1

    def _set_xxz2u_constraints(self) -> None:
        for m in self.index_set.M:
            for mp, k, kp, t in itertools.product(
                self.index_set.M, self.index_set.K, self.index_set.K, self.index_set.TL[m]
            ):
                self.problem += (
                    self.variable.x[m, k] + self.variable.x[mp, kp] + self.variable.z[m, t]
                    <= self.variable.u[m, mp, k, kp, t] + 2
                )
                self.problem += self.variable.u[m, mp, k, kp, t] <= self.variable.x[m, k]
                self.problem += self.variable.u[m, mp, k, kp, t] <= self.variable.z[m, t]

    def _set_xz2v_constraints(self) -> None:
        for m in self.index_set.M:
            for mp, k, t in itertools.product(
                self.index_set.M, self.index_set.K, self.index_set.TL[m]
            ):
                self.problem += (
                    self.variable.x[mp, k] + self.variable.z[m, t]
                    <= self.variable.v[m, mp, k, t] + 1
                )
                self.problem += self.variable.v[m, mp, k, t] <= self.variable.x[mp, k]
                self.problem += self.variable.v[m, mp, k, t] <= self.variable.z[m, t]


class Model(ObjectiveFunctionMixin, ConstraintsMixin):
    index_set: IndexSet
    constant: Constant
    variable: Variable
    lp_problem: pulp.LpProblem

    def __init__(self, index_set: IndexSet, constant: Constant):
        self.index_set = index_set
        self.constant = constant
        self.name = "POORT-L"
        self.calculation_time = None

    def _set_model(self) -> None:
        self.problem = pulp.LpProblem(name=self.name, sense=pulp.LpMaximize)
        self.variable = Variable(self.index_set, self.constant)
        self.set_objective_function()
        self.set_constraints()

    def write_lp(self) -> None:
        self.problem.writeLP(DATA_DIR / "output" / "lpfile" / f"{self.name}.lp")

    def _solve_gurobi(self) -> None:
        pass

    def solve(self, solver: Optional[str] = None) -> None:
        self._set_model()
        # 処理開始直前の時間
        start = time.time()
        if solver == "gurobi":
            self._solve_gurobi()
        else:
            self.problem.solve()
        elapsed_time = time.time() - start
        self.calculation_time = elapsed_time
