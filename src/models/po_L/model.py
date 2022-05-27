from __future__ import annotations

import itertools
import time
from dataclasses import dataclass
from typing import Optional

import pulp
from gurobipy import GurobiError

from src.utils.paths import DATA_DIR


@dataclass(frozen=True)
class IndexSet:
    D: dict[int, list[int]]
    D_: dict[int, list[int]]
    M: list[int]
    K: list[int]
    TL: dict[int, list[int]]
    L: dict[tuple[int, int], list[int]]
    R: dict[tuple[int, int], list[int]]


@dataclass(frozen=True)
class Constant:
    beta: dict[tuple[int, int, int], float]
    phi: dict[tuple[int, int, int], float]
    epsilon: dict[int, float]
    epsilon_max: dict[int, float]
    a: dict[tuple[int, int, int], float]
    b: dict[tuple[int, int], float]
    g: dict[tuple[int, int], float]
    P: dict[tuple[int, int], float]


class Variable:
    def __init__(self, index_set: IndexSet, constant: Constant):
        self.index_set = index_set
        self.constant = constant
        self.u: dict[tuple[int, int, int, int], pulp.LpVariable] = dict()
        self.v: dict[tuple[int, int, int, int], pulp.LpVariable] = dict()
        self.x: dict[tuple[int, int], pulp.LpVariable] = dict()
        self.z: dict[tuple[int, int], pulp.LpVariable] = dict()

        self._set_variables()

    def _set_variables(self) -> None:
        for m in self.index_set.M:
            for mp, k, kp, t in itertools.product(
                self.index_set.M, self.index_set.K, self.index_set.K, self.index_set.TL[m]
            ):
                self.u[m, mp, k, kp, t] = pulp.LpVariable(
                    f"u[{m}_{mp}_{k}_{kp}_{t}]", cat=pulp.LpBinary
                )

            for mp, k, t in itertools.product(
                self.index_set.M, self.index_set.K, self.index_set.TL[m]
            ):
                self.v[m, mp, k, t] = pulp.LpVariable(f"v[{m}_{mp}_{k}_{t}]", cat=pulp.LpBinary)

            for k in self.index_set.K:
                self.x[m, k] = pulp.LpVariable(f"x[{m}_{k}]", cat=pulp.LpBinary)

            for t in self.index_set.TL[m]:
                self.z[m, t] = pulp.LpVariable(f"z[{m}_{t}]", cat=pulp.LpBinary)

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
    problem: pulp.LpProblem

    def __init__(self, index_set: IndexSet, constant: Constant):
        self.index_set = index_set
        self.constant = constant
        self.name = "POORT-L"
        self.calculation_time = None
        self.objective = None

    def _set_model(self) -> None:
        self.problem = pulp.LpProblem(name=self.name, sense=pulp.LpMaximize)
        self.variable = Variable(self.index_set, self.constant)
        self.set_objective_function()
        self.set_constraints()

    def write_lp(self) -> None:
        self.problem.writeLP(DATA_DIR / "output" / "lpfile" / f"{self.name}.lp")

    def solve(self, solver: Optional[str] = None, time_limit: int = 100) -> None:
        self._set_model()
        # 求解
        start = time.time()
        if solver in ["gurobi", "GUROBI", "Gurobi"]:
            try:
                solver = pulp.GUROBI(timeLimit=time_limit)
                return
            except GurobiError:
                raise Exception("Set the solver to Cbc because Gurobi is not installed.")
        else:
            solver = pulp.PULP_CBC_CMD(timeLimit=time_limit)
        self.problem.solve(solver=solver)
        elapsed_time = time.time() - start
        self.calculation_time = elapsed_time
        self.objective = self.problem.objective.value()
