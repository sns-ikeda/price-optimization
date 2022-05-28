from __future__ import annotations

from typing import Any

import pandas as pd


def dict2df(
    dict_: dict[Any, Any], value_name: str, index_name: str = "num_of_items"
) -> pd.DataFrame:
    result_df = (
        pd.DataFrame.from_dict(dict_, orient="index")
        .reset_index()
        .rename(columns={"index": index_name, 0: value_name})
    )
    return result_df.round(3)
