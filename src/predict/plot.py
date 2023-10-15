from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot(
    y: pd.DataFrame,
    y_pred: pd.DataFrame,
    predictor_name: str,
    target_item: str,
    dir_path: Path,
    suffix: Optional[str] = None,
):
    # plot results
    plt.scatter(y_pred, y, color="blue")  # horizontal: predicted, vertical: actual
    if suffix is None:
        plt.title(f"{predictor_name}: {target_item}")
        save_name = f"{predictor_name}_{target_item}.png"
    else:
        plt.title(f"{predictor_name}: {target_item} [{suffix}]")
        save_name = f"{predictor_name}_{target_item}_{suffix}.png"

    plt.xlabel("predicted units")
    plt.ylabel("actual units")
    plt.xlim(0, max(max(np.array(y))[0], max(y_pred)) * 1.1)
    plt.ylim(0, max(max(np.array(y))[0], max(y_pred)) * 1.1)
    plt.grid()
    plt.savefig(dir_path / save_name, format="png", dpi=300)
    plt.clf()
