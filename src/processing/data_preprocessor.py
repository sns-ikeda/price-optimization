from __future__ import annotations

import pandas as pd
from tqdm import tqdm

from src.utils.paths import DATA_DIR

TARGET_STORE_NUM = 25027
TARGET_CATEGORY = "COLD CEREAL"

# データの読み込み
input_path = DATA_DIR / "input" / "breakfast"
_transaction_df = pd.read_csv(str(input_path / "transaction.csv")).dropna(how="all", axis=1)
product_df = pd.read_csv(str(input_path / "product.csv"))
_transaction_product_df = _transaction_df.merge(product_df, on="UPC", how="left")

# WEEKEND_DATEを年・月・日に変換
weekend_dates = _transaction_product_df["WEEK_END_DATE"].tolist()
years, months, days = [], [], []
for weekend_date in tqdm(weekend_dates):
    _list = weekend_date.split("-")
    year = "20" + _list[2]
    month = _list[1].strip("月")
    day = _list[0]
    years.append(year)
    months.append(month)
    days.append(day)
_transaction_product_df = pd.concat(
    [
        _transaction_product_df,
        pd.Series(years, name="YEAR"),
        pd.Series(months, name="MONTH"),
        pd.Series(days, name="DAY"),
    ],
    axis=1,
)

# 対象の店舗・商品カテゴリに絞ってデータを抽出
transaction_product_df = _transaction_product_df.query(
    "STORE_NUM == @TARGET_STORE_NUM and CATEGORY == @TARGET_CATEGORY"
).reset_index(drop=True)
products = transaction_product_df["DESCRIPTION"].unique().tolist()

del _transaction_df, _transaction_product_df

# 学習用のデータに整形
base_columns = [
    "YEAR",
    "MONTH",
    "DAY",
    "PRICE",
    "DESCRIPTION",
    "UNITS",
]
master_columns = [
    "YEAR",
    "MONTH",
    "DAY",
]
value_columns = ["UNITS", "PRICE"]
base_df = transaction_product_df[base_columns]
df = pd.pivot_table(
    base_df, index=master_columns, columns=["DESCRIPTION"], values=value_columns
).reset_index()
df.columns = [
    "_".join(col) if len(set(col).intersection(value_columns)) > 0 else "".join(col)
    for col in df.columns.values
]
