from abc import ABCMeta, abstractmethod
from typing import Optional, TypeVar

import numpy as np

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


class BaseSearchAlgorithm(BaseAlgorithm):
    """探索系アルゴリズムの抽象基底クラス"""

    def __init__(self, model: Model):
        super().__init__(model)

    def calc_z(self, x: dict[tuple[str, int], int]) -> dict[tuple[str, int], int]:
        """xからzを算出"""
        z, mt_z = dict(), dict()
        a = self.model.constant.a
        b = self.model.constant.b
        phi = self.model.constant.P
        g = self.model.constant.g
        M = self.model.index_set.M
        K = self.model.index_set.K
        D = self.model.index_set.D
        TL = self.model.index_set.TL

        # xの値が1となるmとk
        mk_x: dict[str, int] = {m: k for m in M for k in K if x[m, k] >= 0.99}
        assert len(mk_x) == len(M)

        for m in M:
            t = 0
            while True:
                linear_sum = 0
                linear_sum_m = sum(a[m, mp, t] * phi[mp, mk_x[mp]] for mp in M)
                linear_sum_d = sum(a[m, d, t] * g[m, d] for d in D[m])
                linear_sum = linear_sum_m + linear_sum_d
                if linear_sum < b[m, t]:
                    # 左に分岐
                    t = t * 2 + 1
                else:
                    # 右に分岐
                    t = t * 2 + 2
                if t in TL[m]:
                    break
                if t > 1000:
                    raise Exception("Infinite Loop Error")
            mt_z[m] = t
            z.update({(m, t): 1 if t == mt_z[m] else 0 for t in TL[m]})
        return z

    def calc_obj(self, x: dict[tuple[str, int], int], z: dict[tuple[str, int], int]) -> float:
        """x, zから目的関数を計算"""
        M = self.model.index_set.M
        K = self.model.index_set.K
        TL = self.model.index_set.TL

        # xの値が1となるmとk
        mk_x: dict[str, int] = {m: k for m in M for k in K if x[m, k] >= 0.99}

        # zの値が1となるmとt
        mt_z: dict[str, int] = {m: t for m in M for t in TL[m] if z[m, t] >= 0.99}
        assert len(mt_z) == len(mk_x) == len(M)
        assert mk_x.keys() == mt_z.keys()

        P = self.model.constant.P
        beta = self.model.constant.beta
        beta0 = self.model.constant.beta0
        phi = self.model.constant.P
        g = self.model.constant.g
        D = self.model.index_set.D

        p = np.array([P[(m, k)] for m, k in mk_x.items()])
        q = np.array(
            [
                sum(beta[m, mp, t] * phi[mp, mk_x[mp]] for mp in mt_z.keys())
                + sum(beta[m, d, t] * g[m, d] for d in D[m])
                + beta0[m, t]
                for m, t in mt_z.items()
            ]
        )

        return np.dot(p, q)
