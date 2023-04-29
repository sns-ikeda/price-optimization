from abc import ABCMeta, abstractmethod
from typing import Optional

from src.optimize.result import OptResult


class BaseAlgorithm(metaclass=ABCMeta):
    """アルゴリズムの抽象基底クラス"""

    def __init__(self, model):
        self.model = model
        self.result: Optional[OptResult] = None

    @abstractmethod
    def run(self) -> None:
        raise NotImplementedError()
