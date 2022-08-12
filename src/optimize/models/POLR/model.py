from __future__ import annotations

import itertools
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pulp
from gurobipy import GurobiError

from src.optimize.processing import get_opt_prices
from src.optimize.result import Result
from src.utils.paths import DATA_DIR


@dataclass(frozen=True)
class IndexSet:
    D: dict[str, list[int]]
    M: list[str]
    K: list[str]


@dataclass(frozen=True)
class Constant:
    beta: dict[tuple[str, str], float]
    beta0: dict[str, float]
    phi: dict[tuple[str, str, int], float]
    g: dict[str, float]
    P: dict[tuple[str, int], float]
    big_M: int = 1000000


class Variable:
    def __init__(self, index_set: IndexSet, constant: Constant):
        self.index_set = index_set
        self.constant = constant
        self.p: dict[tuple[str], pulp.LpVariable] = dict()
        self.q: dict[tuple[str], pulp.LpVariable] = dict()
        self.u: dict[tuple[str, int], pulp.LpVariable] = dict()
        self.x: dict[tuple[str, int], pulp.LpVariable] = dict()

        self._set_variables()

    def _set_variables(self) -> None:
        for m in self.index_set.M:
            self.p[m] = pulp.LpVariable(f"p[{m}]", cat=pulp.LpContinuous)
            self.q[m] = pulp.LpVariable(f"q[{m}]", cat=pulp.LpContinuous)
            for k in self.index_set.K:
                self.x[m, k] = pulp.LpVariable(f"x[{m}_{k}]", cat=pulp.LpBinary)
                self.u[m, k] = pulp.LpVariable(f"u[{m}_{k}]", cat=pulp.LpContinuous)

    def to_value(self):
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
        for m, k in itertools.product(self.index_set.M, self.index_set.K):
            objective_function += self.constant.P[m, k] * self.variable.u[m, k]
        self.problem += objective_function, "Objective Function"


class ConstraintsMixin:
    index_set: IndexSet
    constant: Constant
    variable: Variable
    problem: pulp.LpProblem

    def set_constraints(self) -> None:
        self._set_q_constraints()
        self._set_price_constraints()
        self._set_p_constraints()
        self._set_u_constraints()

    def _set_q_constraints(self) -> None:
        for m in self.index_set.M:
            self.problem += (
                self.variable.q[m]
                == pulp.lpSum(
                    self.constant.beta[m, mp] * self.variable.p[mp] for mp in self.index_set.M
                )
                + pulp.lpSum(
                    self.constant.beta[m, d] * self.constant.g[d] for d in self.index_set.D[m]
                )
                + self.constant.beta0[m]
            )

    def _set_price_constraints(self) -> None:
        for m in self.index_set.M:
            self.problem += pulp.lpSum(self.variable.x[m, k] for k in self.index_set.K) == 1

    def _set_p_constraints(self) -> None:
        for m in self.index_set.M:
            self.problem += self.variable.p[m] == pulp.lpSum(
                self.constant.P[m, k] * self.variable.x[m, k] for k in self.index_set.K
            )

    def _set_u_constraints(self) -> None:
        for m, k in itertools.product(self.index_set.M, self.index_set.K):
            self.problem += -self.constant.big_M * self.variable.x[m, k] <= self.variable.u[m, k]
            self.problem += self.variable.u[m, k] <= self.constant.big_M * self.variable.x[m, k]
            self.problem += (
                self.variable.q[m] - self.constant.big_M * (1 - self.variable.x[m, k])
                <= self.variable.u[m, k]
            )
            self.problem += self.variable.u[m, k] <= self.variable.q[m] + self.constant.big_M * (
                1 - self.variable.x[m, k]
            )


class Model(ObjectiveFunctionMixin, ConstraintsMixin):
    index_set: IndexSet
    constant: Constant
    variable: Variable
    problem: pulp.LpProblem

    def __init__(self, index_set: IndexSet, constant: Constant):
        self.index_set = index_set
        self.constant = constant
        self.name = "POLR"
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
        write_lp: bool = False,
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
        opt_prices = get_opt_prices(x=self.variable.x, P=self.constant.P)
        self.result = Result(
            calculation_time=elapsed_time,
            objective=self.objective,
            opt_prices=opt_prices,
            problem=self.problem,
            index_set=self.index_set,
            constant=self.constant,
            variable=self.variable,
        )
        if write_lp:
            self.write_lp()
