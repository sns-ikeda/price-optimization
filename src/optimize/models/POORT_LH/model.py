from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pulp
from gurobipy import GurobiError

from src.optimize.processing import get_opt_prices
from src.optimize.result import OptResult
from src.utils.paths import RESULT_DIR


@dataclass(frozen=False)
class IndexSet:
    D: dict[str, list[str]]
    M: list[str]
    K: list[int]
    TL: dict[str, list[int]]
    L: dict[tuple[str, int], list[int]]
    R: dict[tuple[str, int], list[int]]


@dataclass(frozen=False)
class Constant:
    beta: dict[tuple[str, str, int], float]
    beta0: dict[tuple[str, int], float]
    epsilon: dict[str, float]
    a: dict[tuple[str, str, int], float]
    b: dict[tuple[str, int], float]
    g: dict[tuple[str, str], float]
    P: dict[tuple[str, int], float]
    prices: list[float] = field(default_factory=list)
    big_M: int = 10000


class Variable:
    def __init__(self, index_set: IndexSet, constant: Constant, relax: bool):
        self.index_set = index_set
        self.constant = constant
        self.relax = relax
        self.p: dict[str, pulp.LpVariable] = dict()
        self.q: dict[str, pulp.LpVariable] = dict()
        self.u: dict[tuple[str, int], pulp.LpVariable] = dict()
        self.x: dict[tuple[str, int], pulp.LpVariable] = dict()
        self.z: dict[tuple[str, int], pulp.LpVariable] = dict()

        self._set_variables()

    def _set_variables(self) -> None:
        for m in self.index_set.M:
            self.p[m] = pulp.LpVariable(f"p[{m}]", cat=pulp.LpContinuous)
            self.q[m] = pulp.LpVariable(f"q[{m}]", cat=pulp.LpContinuous)
            for k in self.index_set.K:
                if self.relax:
                    self.x[m, k] = pulp.LpVariable(
                        f"x[{m}_{k}]", lowBound=0, upBound=1, cat=pulp.LpContinuous
                    )
                else:
                    self.x[m, k] = pulp.LpVariable(f"x[{m}_{k}]", cat=pulp.LpBinary)
                self.u[m, k] = pulp.LpVariable(f"u[{m}_{k}]", cat=pulp.LpContinuous)
            for t in self.index_set.TL[m]:
                if self.relax:
                    self.z[m, t] = pulp.LpVariable(
                        f"z[{m}_{t}]", lowBound=0, upBound=1, cat=pulp.LpContinuous
                    )
                else:
                    self.z[m, t] = pulp.LpVariable(f"z[{m}_{t}]", cat=pulp.LpBinary)

    def to_value(self):
        for attr in self.__dict__.keys():
            if attr not in ["index_set", "constant", "relax"]:
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
    x: Optional[dict[str, int]]

    def set_constraints(self) -> None:
        self._set_q_constraints()
        self._set_branch_constraints()
        self._set_leafnode_constraints()
        self._set_price_constraints()
        self._set_p_constraints()
        self._set_u_constraints()
        if self.x is not None:
            self._set_x_constraints()

    def _set_x_constraints(self) -> None:
        for m, k in self.x.items():
            self.problem += self.variable.x[m, k] == 1

    def _set_q_constraints(self) -> None:
        for m in self.index_set.M:
            for t in self.index_set.TL[m]:
                self.problem += self.variable.q[m] - pulp.lpSum(
                    self.constant.beta.get((m, mp, t), 0) * self.variable.p[mp]
                    for mp in self.index_set.M
                ) - pulp.lpSum(
                    self.constant.beta.get((m, d, t), 0) * self.constant.g[m, d]
                    for d in self.index_set.D[m]
                ) - self.constant.beta0[
                    m, t
                ] >= -self.constant.big_M * (
                    1 - self.variable.z[m, t]
                )

                self.problem += self.variable.q[m] - pulp.lpSum(
                    self.constant.beta.get((m, mp, t), 0) * self.variable.p[mp]
                    for mp in self.index_set.M
                ) - pulp.lpSum(
                    self.constant.beta.get((m, d, t), 0) * self.constant.g[m, d]
                    for d in self.index_set.D[m]
                ) - self.constant.beta0[
                    m, t
                ] <= self.constant.big_M * (
                    1 - self.variable.z[m, t]
                )

    def _set_branch_constraints(self) -> None:
        for m in self.index_set.M:
            for t in self.index_set.TL[m]:
                for tp in self.index_set.R[m, t]:
                    self.problem += (
                        pulp.lpSum(
                            self.constant.a.get((m, mp, tp), 0) * self.variable.p[mp]
                            for mp in self.index_set.M
                        )
                        + pulp.lpSum(
                            self.constant.a.get((m, d, tp), 0) * self.constant.g[m, d]
                            for d in self.index_set.D[m]
                        )
                        >= self.constant.b[m, tp]
                        + self.constant.big_M * self.variable.z[m, t]
                        - self.constant.big_M
                    )
                for tp in self.index_set.L[m, t]:
                    self.problem += (
                        pulp.lpSum(
                            self.constant.a.get((m, mp, tp), 0) * self.variable.p[mp]
                            for mp in self.index_set.M
                        )
                        + pulp.lpSum(
                            self.constant.a.get((m, d, tp), 0) * self.constant.g[m, d]
                            for d in self.index_set.D[m]
                        )
                        + self.constant.epsilon
                        <= self.constant.b[m, tp]
                        - self.constant.big_M * self.variable.z[m, t]
                        + self.constant.big_M
                    )

    def _set_leafnode_constraints(self) -> None:
        for m in self.index_set.M:
            self.problem += pulp.lpSum(self.variable.z[m, t] for t in self.index_set.TL[m]) == 1

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

    def __init__(
        self,
        index_set: IndexSet,
        constant: Constant,
        x: Optional[dict[str, int]] = None,
        relax: bool = False,
    ):
        self.index_set = index_set
        self.constant = constant
        self.x = x
        self.name = "POORT_LH"
        self.result = None
        self.calculation_time = None
        self.objective = None
        self.relax = relax

    def _set_model(self) -> None:
        self.problem = pulp.LpProblem(name=self.name, sense=pulp.LpMaximize)
        self.variable = Variable(self.index_set, self.constant, self.relax)
        self.set_objective_function()
        self.set_constraints()

    def write_lp(self, dir_path: Optional[Path] = RESULT_DIR / "lpfile") -> None:
        self.problem.writeLP(str(dir_path / f"{self.name}.lp"))

    def solve(
        self,
        solver: Optional[str] = None,
        TimeLimit: int = 100,
        NoRelHeurTime: float = 0,
        MIPFocus: int = 0,
        write_lp: bool = True,
    ) -> None:
        # モデルを構築
        self._set_model()
        # solverの設定
        if solver in ["gurobi", "GUROBI", "Gurobi", "grb", "GRB", "Grb"]:
            try:
                solver = pulp.GUROBI(
                    timeLimit=TimeLimit, NoRelHeurTime=NoRelHeurTime, MIPFocus=MIPFocus, threads=1
                )
            except GurobiError:
                raise Exception("Set the solver to Cbc because Gurobi is not installed.")
        else:
            solver = pulp.PULP_CBC_CMD(timeLimit=TimeLimit)

        if write_lp:
            self.write_lp()

        # 求解
        start = time.time()
        self.problem.solve(solver=solver)
        elapsed_time = time.time() - start

        # 結果の格納
        self.objective = self.problem.objective.value()
        self.variable.to_value()
        opt_prices = get_opt_prices(x=self.variable.x, P=self.constant.P)
        self.result = OptResult(
            calculation_time=elapsed_time,
            objective=self.objective,
            opt_prices=opt_prices,
            problem=self.problem,
            index_set=self.index_set,
            constant=self.constant,
            variable=self.variable,
        )
