from __future__ import annotations

from itertools import zip_longest

import numpy as np
import pandas as pd

from src.optimize.models.POLR.model import Constant, IndexSet


def generate_data(
    price_candidates: list[list[float]],
    index_set: IndexSet,
    constant: Constant,
    noise_variance: float,
) -> tuple[dict[str, pd.DataFrame], dict[str, float]]:
    df_dict, q_avg_dict = dict(), dict()
    M = list(constant.beta0.keys())
    for m in M:
        # generate external variables
        external_df = pd.DataFrame()
        external_vals = []
        if constant.g:
            num_external_val = len(list(index_set.D.values())[0])
            for _ in range(len(price_candidates)):
                _external_vals = []
                for _ in range(num_external_val):
                    _external_vals.append(
                        round(float(np.random.normal(loc=0, scale=noise_variance, size=1)), 3)
                    )
                external_vals.append(_external_vals)
            external_cols = [f"G_{m}_{d + len(M)}" for d in range(num_external_val)]
            external_df = pd.DataFrame(external_vals, columns=external_cols)

        # generate sales
        qs = []
        for prices, external_vals_ in zip_longest(price_candidates, external_vals):
            q = constant.beta0[m]
            for i, m_ in enumerate(M):
                q += constant.beta[m, m_] * prices[i]
            try:
                for j, d in enumerate(index_set.D[m]):
                    q += constant.beta[m, d] * external_vals_[j]
            except KeyError:
                pass
            qs.append(round(q, 3))

        # add noise
        q_avg = np.mean(qs)
        q_avg_dict[m] = q_avg
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
        df = pd.concat([price_df, external_df, unit_df], axis=1)
        df_dict[m] = df
    return df_dict, q_avg_dict
