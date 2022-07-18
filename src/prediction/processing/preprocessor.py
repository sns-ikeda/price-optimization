from __future__ import annotations

from typing import Optional

import pandas as pd
from tqdm import tqdm

from src.configs import CONFIG_DATA
from src.prediction.processing.data_loader import data_loader


def filter_df(df: pd.DataFrame, category: Optional[str] = None, store_num: Optional[int] = None):
    df_ = df.copy()
    if category is not None:
        df_ = df_.query("CATEGORY == @category").reset_index(drop=True)
    if store_num is not None:
        df_ = df_.query("STORE_NUM == @store_num").reset_index(drop=True)
    return df_


def load_preprocess(dataset: str = "breakfast"):
    _df = data_loader(dataset=dataset)
    config = CONFIG_DATA["realworld"][dataset]
    if dataset == "breakfast":
        # WEEKEND_DATEを年・月・日に変換
        weekend_dates = _df["WEEK_END_DATE"].tolist()
        years, months, days = [], [], []
        for weekend_date in tqdm(weekend_dates):
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
        category = config["category"]
        store_num = config["store_num"]
        filtered_df = filter_df(df=_df, category=category, store_num=store_num)

        # 学習用のデータに整形
        base_cols = config["base_cols"]
        master_cols = config["master_cols"]
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


def make_target_data(
    df: pd.DataFrame, target_col: str, feature_cols: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _df = df.copy()
    _df = _df.dropna()
    X = _df[feature_cols]
    y = _df[target_col]
    return X, y
