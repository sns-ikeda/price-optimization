from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot(
    y: pd.DataFrame,
    y_pred: pd.DataFrame,
    predictor_name: str,
    target_item: str,
    dir_path: Path,
    suffix: Optional[str] = None,
):
    # 結果をプロット
    plt.scatter(y, y_pred, color="blue")  # 横軸に予測値，縦軸に実測値
    if suffix is None:
        plt.title(f"{predictor_name}: {target_item}")  # 図のタイトル
        save_name = f"{predictor_name}_{target_item}.png"
    else:
        plt.title(f"{predictor_name}: {target_item} [{suffix}]")  # 図のタイトル
        save_name = f"{predictor_name}_{target_item}_{suffix}.png"

    plt.xlabel("actual units")  # x軸のラベル
    plt.ylabel("predicted units")  # y軸のラベル
    plt.xlim(0, max(max(np.array(y))[0], max(y_pred)) * 1.1)
    plt.ylim(0, max(max(np.array(y))[0], max(y_pred)) * 1.1)
    plt.grid()  # グリッド線を表示
    plt.savefig(dir_path / save_name, format="png", dpi=300)
    plt.clf()
