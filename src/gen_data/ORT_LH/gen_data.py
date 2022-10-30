from __future__ import annotations

import numpy as np
import pandas as pd

from src.optimize.models.POLR.model import Constant, IndexSet


def calculate_z(x: dict[str, int], constant, index_set) -> dict[str, int]:
    """xからzを算出"""
    z = dict()
    for m in index_set.M:
        t = 0
        while True:
            linear_sum = 0
            for mp in index_set.M:
                k = x[mp]
                linear_sum += constant.a[m, mp, t] * constant.P[mp, k]
            for d in index_set.D[m]:
                linear_sum += constant.a[m, d, t] * constant.g[m, d]
            if linear_sum < constant.b[m, t]:
                # 左に分岐
                t = t * 2 + 1
            else:
                # 右に分岐
                t = t * 2 + 2
            if t in index_set.TL[m]:
                break
            if t > 1000:
                raise Exception("Infinite Loop Error")
        z[m] = t
    return z


def calculate_q(x: dict[str, int], z: dict[str, int], constant, index_set) -> dict[str, float]:
    """x, zから目的関数を計算"""
    np.array([constant.P[(m, k)] for m, k in x.items()])
    q = dict()
    for m, t in z.items():
        q_m = constant.beta0[m, t]
        for mp in z.keys():
            k = x[mp]
            q_m += constant.beta[m, mp, t] * constant.P[mp, k]
        for d in index_set.D[m]:
            q_m += constant.beta[m, d, t] * constant.g[m, d]
        q[m] = q_m
    return q


def generate_data(
    price_candidates: list[list[float]],
    index_set: IndexSet,
    constant: Constant,
    noise_variance: float,
) -> dict[str, pd.DataFrame]:
    df_dict = dict()
    M = index_set.M
    for m in M:
        # 販売数のデータ作成
        qs = []
        for prices in price_candidates:
            x = dict()
            for price, m_ in zip(prices, M):
                for k in index_set.K:
                    if round(price, 3) == round(constant.P[m_, k], 3):
                        x[m_] = k
            assert len(x) == len(M), f"length of x is {len(x)} and length of M is {len(M)}"
            z = calculate_z(x, constant, index_set)
            q = calculate_q(x, z, constant, index_set)
            qs.append(round(q[m], 3))

        # 販売数にノイズ追加
        q_avg = np.mean(qs)
        qs_noise = []
        for q in qs:
            qs_noise.append(
                q
                + round(
                    float(np.random.normal(loc=0, scale=noise_variance * abs(q_avg), size=1)), 3
                )
            )
        price_cols = ["PRICE_" + m for m in M]
        price_df = pd.DataFrame(price_candidates, columns=price_cols)
        unit_col = ["UNITS_" + m]
        unit_df = pd.DataFrame(qs_noise, columns=unit_col)
        df = pd.concat([price_df, unit_df], axis=1)
        df_dict[m] = df
    return df_dict
