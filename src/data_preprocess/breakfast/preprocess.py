from __future__ import annotations

from typing import Optional

import pandas as pd


def filter_df(df: pd.DataFrame, category: Optional[str] = None, store_num: Optional[int] = None):
    df_ = df.copy()
    if category is not None:
        df_ = df_.query("CATEGORY == @category").reset_index(drop=True)
    if store_num is not None:
        df_ = df_.query("STORE_NUM == @store_num").reset_index(drop=True)
    return df_


def preprocess(
    df: pd.DataFrame,
    base_cols: list[str],
    master_cols: list[str],
    category: Optional[str] = None,
    store_num: Optional[int] = None,
    **kwargs,
) -> pd.DataFrame:
    _df = df.copy()
    weekend_dates = _df["WEEK_END_DATE"].tolist()
    years, months, days = [], [], []
    for weekend_date in weekend_dates:
        _list = weekend_date.split("-")
        year = "20" + _list[2]
        month = _list[1].strip("月")
        day = _list[0]
        years.append(year)
        months.append(month)
        days.append(day)
    _df = pd.concat(
        [
            _df,
            pd.Series(years, name="YEAR"),
            pd.Series(months, name="MONTH"),
            pd.Series(days, name="DAY"),
        ],
        axis=1,
    )
    # 対象の店舗・商品カテゴリに絞ってデータを抽出
    filtered_df = filter_df(df=_df, category=category, store_num=store_num)

    # 学習用のデータに整形
    value_cols = ["UNITS", "PRICE"]
    base_df = filtered_df[base_cols]
    df = pd.pivot_table(
        base_df, index=master_cols, columns=["DESCRIPTION"], values=value_cols
    ).reset_index()
    df.columns = [
        "_".join(col) if len(set(col).intersection(value_cols)) > 0 else "".join(col)
        for col in df.columns.values
    ]
    df[master_cols] = df[master_cols].astype(int)
    df = df.dropna(how="all", axis=1).sort_values(by=master_cols).reset_index(drop=True)
    return df
