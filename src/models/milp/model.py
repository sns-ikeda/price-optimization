from typing import Optional

import pulp

from .model_element import Constant, ConstraintsMixin, IndexSet, ObjectiveFunctionMixin, Variable


class Model(ObjectiveFunctionMixin, ConstraintsMixin):
    index_set: IndexSet
    constant: Constant
    variable: Variable
    lp_problem: pulp.LpProblem

    def __init__(self, index_set: IndexSet, constant: Constant):
        self.index_set = index_set
        self.constant = constant

    def __set_model(self) -> None:
        self.problem = pulp.LpProblem()
        self.variable = Variable(self.index_set, self.constant)
        self.set_objective_function()
        self.set_constraints()

    def generate_lp_file(self) -> None:
        pass

    def __solve_gurobi(self) -> None:
        pass

    def solve(self, solver: Optional[str] = None) -> None:
        self.__set_model()
        if solver == "gurobi":
            self.__solve_gurobi()
        else:
            self.problem.solve()
