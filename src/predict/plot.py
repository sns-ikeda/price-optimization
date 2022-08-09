from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import pandas as pd


def plot(
    X: pd.DataFrame,
    y: pd.DataFrame,
    y_pred: pd.DataFrame,
    predictor_name: str,
    target_item: str,
    dir_path: Union[str, Path],
    suffix: Optional[str] = None,
):
    # 結果をプロット
    x_col = "PRICE" + "_" + target_item
    plt.scatter(X[x_col], y, color="blue", label="actual")  # 説明変数と目的変数のデータ点の散布図をプロット
    plt.scatter(X[x_col], y_pred, color="red", label="predicted")  # 回帰直線をプロット
    if suffix is None:
        plt.title(f"{predictor_name}: {target_item}")  # 図のタイトル
        save_name = f"{predictor_name}_{target_item}.png"
    else:
        plt.title(f"{predictor_name}: {target_item} [{suffix}]")  # 図のタイトル
        save_name = f"{predictor_name}_{target_item}_{suffix}.png"

    plt.xlabel("Price[$]")  # x軸のラベル
    plt.ylabel("# of Units")  # y軸のラベル
    plt.grid()  # グリッド線を表示
    plt.legend()
    # plt.show()
    plt.savefig(Path(dir_path) / save_name, format="png", dpi=300)
    plt.clf()
