from __future__ import annotations

import itertools
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pulp
from gurobipy import GurobiError

from src.optimize.result import Result
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
    big_M: int = 100000


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
                    f"u[{m}_{mp}_{k}_{kp}_{t}]", cat=pulp.LpContinuous, lowBound=0, upBound=1
                )
            for k, t in itertools.product(self.index_set.K, self.index_set.TL[m]):
                self.v[m, k, t] = pulp.LpVariable(
                    f"v[{m}_{k}_{t}]", cat=pulp.LpContinuous, lowBound=0, upBound=1
                )

            for k in self.index_set.K:
                self.x[m, k] = pulp.LpVariable(f"x[{m}_{k}]", cat=pulp.LpBinary)

            for t in self.index_set.TL[m]:
                self.z[m, t] = pulp.LpVariable(f"z[{m}_{t}]", cat=pulp.LpBinary)

    def to_value(self) -> None:
        for attr in self.__dict__.keys():
            if attr not in ["index_set", "constant"]:
                for k, v in self.__dict__[attr].items():
                    self.__dict__[attr][k] = v.value()


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
                    * self.constant.phi[m, mp, kp]
                    * self.constant.P[m, k]
                    * self.variable.u[m, mp, k, kp, t]
                )

            for k, t, d in itertools.product(
                self.index_set.K, self.index_set.TL[m], self.index_set.D[m] + self.index_set.D_[m]
            ):
                objective_function += (
                    self.constant.beta[m, d, t]
                    * self.constant.P[m, k]
                    * self.constant.g[m, d]
                    * self.variable.v[m, k, t]
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
                self.problem += self.variable.u[m, mp, k, kp, t] <= self.variable.x[mp, kp]
                self.problem += self.variable.u[m, mp, k, kp, t] <= self.variable.x[m, k]
                self.problem += self.variable.u[m, mp, k, kp, t] <= self.variable.z[m, t]

    def _set_xz2v_constraints(self) -> None:
        for m in self.index_set.M:
            for k, t in itertools.product(self.index_set.K, self.index_set.TL[m]):
                self.problem += (
                    self.variable.x[m, k] + self.variable.z[m, t] <= self.variable.v[m, k, t] + 1
                )
                self.problem += self.variable.v[m, k, t] <= self.variable.x[m, k]
                self.problem += self.variable.v[m, k, t] <= self.variable.z[m, t]


class Model(ObjectiveFunctionMixin, ConstraintsMixin):
    index_set: IndexSet
    constant: Constant
    variable: Variable
    problem: pulp.LpProblem

    def __init__(self, index_set: IndexSet, constant: Constant):
        self.index_set = index_set
        self.constant = constant
        self.name = "POORT_L"
        self.result = None
        self.calculation_time = None
        self.objective = None

    def _set_model(self) -> None:
        self.problem = pulp.LpProblem(name=self.name, sense=pulp.LpMaximize)
        self.variable = Variable(self.index_set, self.constant)
        self.set_objective_function()
        self.set_constraints()

    def write_lp(self, dir_path: Optional[Path] = DATA_DIR / "results" / "lpfile") -> None:
        self.problem.writeLP(str(dir_path / f"{self.name}.lp"))

    def solve(
        self,
        solver: Optional[str] = None,
        TimeLimit: int = 100,
        NoRelHeurTime: float = 0,
        MIPFocus: int = 0,
    ) -> None:
        # モデルを構築
        self._set_model()
        # solverの設定
        if solver in ["gurobi", "GUROBI", "Gurobi", "grb", "GRB", "Grb"]:
            try:
                solver = pulp.GUROBI(
                    timeLimit=TimeLimit, NoRelHeurTime=NoRelHeurTime, MIPFocus=MIPFocus
                )
            except GurobiError:
                raise Exception("Set the solver to Cbc because Gurobi is not installed.")
        else:
            solver = pulp.PULP_CBC_CMD(timeLimit=TimeLimit)
        # 求解
        start = time.time()
        self.problem.solve(solver=solver)
        elapsed_time = time.time() - start
        # 結果の格納
        self.objective = self.problem.objective.value()
        self.variable.to_value()
        self.result = Result(calculation_time=elapsed_time, objective=self.objective)
        self.write_lp()
