from abc import ABCMeta, abstractmethod
from typing import Optional, TypeVar

from src.optimize.result import OptResult

Model = TypeVar("Model")


class BaseAlgorithm(metaclass=ABCMeta):
    """アルゴリズムの抽象基底クラス"""

    def __init__(self, model: Model):
        self.model: Model = model
        self.result: Optional[OptResult] = None

    @abstractmethod
    def run(self) -> None:
        raise NotImplementedError()
