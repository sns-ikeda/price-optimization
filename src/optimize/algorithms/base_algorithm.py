from abc import ABCMeta, abstractmethod


class BaseAlgorithm(metaclass=ABCMeta):
    """アルゴリズムの抽象基底クラス"""

    def __init__(self, model):
        self.model = model
        self.result = None

    @abstractmethod
    def run(self) -> None:
        raise NotImplementedError()
