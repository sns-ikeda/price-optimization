from __future__ import annotations

import copy
import time

import numpy as np
from logzero import logger
from tqdm import tqdm

from src.optimize.algorithms.base_algorithm import BaseAlgorithm
from src.optimize.result import Result


class MultiLocalSearch(BaseAlgorithm):
    def __init__(self, model, num_multi_start: int = 1, seed: int = 0):
        super().__init__(model)
        self.num_multi_start = num_multi_start
        self.seed = seed
        self.index_set, self.constant = self.model.index_set, self.model.constant
        self.xs = []
        self.zs = []
        self.objectives = []

    def run(self) -> None:
        start = time.time()
        for i in tqdm(range(self.num_multi_start)):
            logger.info(f"num of multi start: {i}")
            self.local_search(seed=self.seed + i)
        elapsed_time = time.time() - start
        self.result = Result(
            calculation_time=elapsed_time, objective=max(self.objectives, default=None)
        )

    def local_search(self, seed: int = 0) -> None:
        """局所探索を1度実行"""
        # 初期解を生成
        current_x, current_z = self.get_initial_solution(seed)
        current_objective = self.evaluate(current_x, current_z)

        counter = 0
        while True:
            self.xs.append(current_x)
            self.zs.append(current_z)
            self.objectives.append(current_objective)

            # 近傍を取得
            x_neighbors = self.get_neighbors(x=current_x, num_of_price=len(self.index_set.K))
            best_x_neighbor, best_z_neighbor, best_objective = self.get_best_neighbor(
                x_neighbors=x_neighbors
            )
            logger.info(f"current_objective: {current_objective}")
            # 近傍解で今よりも良い解があれば移動する
            if best_objective > current_objective:
                current_objective = best_objective
                current_x = best_x_neighbor
                current_z = best_z_neighbor
            else:
                break

            counter += 1
            if counter > 100000:
                raise Exception("Infinite Loop Error")

    def get_initial_solution(self, seed: int = 0) -> tuple[dict[int, int], dict[int, int]]:
        """初期の状態を生成"""
        x = dict()
        for m in self.index_set.M:
            np.random.seed(100 * seed + m)
            x[m] = np.random.choice(self.index_set.K)
        z = self.calculate_z(x=x)
        return x, z

    def evaluate(self, x: dict[int, int], z: dict[int, int]) -> float:
        """x, zから目的関数を計算"""
        p = np.array([self.constant.P[(m, k)] for m, k in x.items()])
        q = []
        for m, t in z.items():
            q_m = 0
            for mp in z.keys():
                k = x[mp]
                q_m += self.constant.beta[m, mp, t] * self.constant.phi[m, mp, k]
            for d in self.index_set.D[m] + self.index_set.D_[m]:
                q_m += self.constant.beta[m, d, t] * self.constant.g[m, d]
            q.append(q_m)
        q = np.array(q)
        return np.dot(p, q)

    @staticmethod
    def get_neighbors(x: dict[int, int], num_of_price: int) -> list[dict[int, int]]:
        """近傍解を取得"""
        x_neighbors = []
        for m, k in x.items():
            k_neighbors = []
            if k > 0:
                k_neighbors.append(k - 1)
            if k < num_of_price - 1:
                k_neighbors.append(k + 1)
            for k_neighbor in k_neighbors:
                x_neighbor = copy.deepcopy(x)
                x_neighbor[m] = k_neighbor
                x_neighbors.append(x_neighbor)
        return x_neighbors

    def calculate_z(self, x: dict[int, int]) -> dict[int, int]:
        """xからzを算出"""
        z = dict()
        for m in self.index_set.M:
            t = 0
            while True:
                linear_sum = 0
                for mp in self.index_set.M:
                    k = x[mp]
                    linear_sum += self.constant.a[m, mp, t] * self.constant.phi[m, mp, k]
                for d in self.index_set.D[m]:
                    linear_sum += self.constant.a[m, d, t] * self.constant.g[m, d]
                if linear_sum < self.constant.b[m, t]:
                    # 左に分岐
                    t = t * 2 + 1
                else:
                    # 右に分岐
                    t = t * 2 + 2
                if t in self.index_set.TL[m]:
                    break
                if t > 1000:
                    raise Exception("Infinite Loop Error")
            z[m] = t
        return z

    def get_best_neighbor(self, x_neighbors: list[dict[int, int]]) -> tuple[dict[int, int], float]:
        """近傍の中から最も良い解とそのときの目的関数の値を取得"""
        best_x_neighbor, best_z_neighbor, best_objective = None, None, -np.inf
        for x_neighbor in x_neighbors:
            z_neighbor = self.calculate_z(x=x_neighbor)
            _objctive = self.evaluate(x=x_neighbor, z=z_neighbor)
            if _objctive > best_objective:
                best_objective = _objctive
                best_x_neighbor = x_neighbor
                best_z_neighbor = z_neighbor
        return best_x_neighbor, best_z_neighbor, best_objective