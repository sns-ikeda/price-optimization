from abc import ABCMeta, abstractmethod

from src.models.po_L.params import Parameter


class BaseAlgorithm(metaclass=ABCMeta):
    """アルゴリズムの抽象基底クラス"""

    def __init__(self, params: Parameter):
        self.params = params
        self.result = None

    @abstractmethod
    def run(self) -> None:
        raise NotImplementedError()
