from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.optimize.models.POLR.model import Constant


def generate_data(
    price_candidates: list[list[float]], constant: Constant, noise_variance: float = 0.1
) -> dict[str, pd.DataFrame]:
    df_dict = dict()
    M = list(constant.beta0.keys())
    for m in tqdm(M):
        qs = []
        for prices in price_candidates:
            q = constant.beta0[m]
            for i, m_ in enumerate(M):
                q += constant.beta[m, m_] * prices[i]
            q = float(np.random.normal(loc=q, scale=noise_variance * q, size=1))
            qs.append(q)
        price_cols = ["PRICE_" + m for m in M]
        unit_col = ["UNITS_" + m]
        price_df = pd.DataFrame(price_candidates, columns=price_cols)
        unit_df = pd.DataFrame(qs, columns=unit_col)
        df = pd.concat([price_df, unit_df], axis=1)
        df_dict[m] = df
    return df_dict
