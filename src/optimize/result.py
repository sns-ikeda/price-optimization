from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TypeVar

import pulp

from src.optimize.params import RealDataParameter, SyntheticDataParameter

IndexSet = TypeVar("IndexSet")
Constant = TypeVar("Constant")
Variable = TypeVar("Variable")


@dataclass(frozen=False)
class OptResult:
    calculation_time: float
    objective: float
    opt_prices: dict[str, float]
    data_param: Optional[SyntheticDataParameter | RealDataParameter] = None
    problem: Optional[pulp.LpProblem] = None
    index_set: Optional[IndexSet] = None
    constant: Optional[Constant] = None
    variable: Optional[Variable] = None

    def to_dict(self):
        return {**self.__dict__, **self.data_param.__dict__}
