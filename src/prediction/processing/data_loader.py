from __future__ import annotations

import pandas as pd

from src.utils.paths import DATA_DIR


# データの読み込み
def data_loader(dataset: str = "breakfast") -> pd.DataFrame:
    data_path = DATA_DIR / "dataset" / dataset
    if dataset == "breakfast":
        transaction_df = pd.read_csv(str(data_path / "transaction.csv"))
        product_df = pd.read_csv(str(data_path / "product.csv"))
        transaction_product_df = transaction_df.merge(product_df, on="UPC", how="left")
        return transaction_product_df
