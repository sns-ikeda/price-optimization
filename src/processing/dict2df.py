from __future__ import annotations

import pandas as pd


def dict2df(dict_: dict[int, float], value_name: str) -> pd.DataFrame:
    result_df = (
        pd.DataFrame.from_dict(dict_, orient="index")
        .reset_index()
        .rename(columns={"index": "num_of_items", 0: value_name})
    )
    return result_df.round(3)
