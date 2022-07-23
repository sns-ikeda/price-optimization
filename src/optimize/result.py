from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.optimize.params import ArtificialDataParameter, RealDataParameter


@dataclass(frozen=False)
class Result:
    calculation_time: float
    objective: float
    data_param: Optional[ArtificialDataParameter | RealDataParameter] = None

    def to_dict(self):
        return {**self.__dict__, **self.data_param.__dict__}
