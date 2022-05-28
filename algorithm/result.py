from dataclasses import dataclass


@dataclass(frozen=True)
class Result:
    calculation_time: float
    objective: float
