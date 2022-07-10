from __future__ import annotations

import numpy as np
import pandas as pd
from logzero import logger
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
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
df[master_columns] = df[master_columns].astype(int)
df = df.sort_values(by=["YEAR", "MONTH", "DAY"]).reset_index()


def make_target_data(
    df: pd.DataFrame, product: str, feature_cols: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _df = df.copy()
    _df = _df.dropna()
    target_col = "UNITS" + "_" + product
    X = _df[feature_cols]
    y = _df[target_col]
    return X, y


feature_cols = ["PRICE" + "_" + product for product in products] + master_columns
for product in tqdm(products):
    logger.info(product)
    X, y = make_target_data(df=df, product=product, feature_cols=feature_cols)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.7, test_size=0.3, shuffle=False
    )  # データを学習用と検証用に分割
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_test)  # 検証データを用いて目的変数を予測

    from sklearn.metrics import mean_squared_error

    y_train_pred = lr.predict(X_train)  # 学習データに対する目的変数を予測
    print(
        "RMSE train data: ", np.sqrt(mean_squared_error(y_train, y_train_pred))
    )  # 学習データを用いたときの平均二乗誤差を出力
    print("RMSE test data: ", np.sqrt(mean_squared_error(y_test, y_pred)))  # 検証データを用いたときの平均二乗誤差を出力

    import matplotlib.pyplot as plt

    x_col = "PRICE" + "_" + product
    plt.scatter(X[x_col], y, color="blue", label="actual")  # 説明変数と目的変数のデータ点の散布図をプロット
    plt.scatter(X[x_col], lr.predict(X), color="red", label="predicted")  # 回帰直線をプロット

    plt.title(f"Regression: {product}")  # 図のタイトル
    plt.xlabel("Price[$]")  # x軸のラベル
    plt.ylabel("#Units")  # y軸のラベル
    plt.grid()  # グリッド線を表示
    plt.legend()

    plt.show()
