from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

IndexSet = TypeVar("IndexSet")
Constant = TypeVar("Constant")


@dataclass
class ModelInput:
    model_name: str
    index_set: IndexSet
    constant: Constant
