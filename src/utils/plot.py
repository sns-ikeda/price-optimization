from __future__ import annotations

from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import seaborn as sns


def save_image(
    avg_results_dict: dict[str, dict[int, float]],
    dir_path: Union[str, Path],
    image_name: str,
    y_label: str,
    x_label: str = "num of items",
) -> None:

    plt.style.use("default")
    sns.set()
    sns.set_style("whitegrid")
    sns.set_palette("gray")
    sns.set_palette("Set1")
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    y_max = 0
    for method, avg_results_dict_method in avg_results_dict.items():
        x = list(avg_results_dict_method.keys())
        y = list(avg_results_dict_method.values())
        ax.plot(x, y, marker="o", label=method)
        if max(y) > y_max:
            y_max = max(y)

    ax.legend()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_ylim(0, y_max * 1.1)
    fig.savefig(Path(dir_path) / f"{image_name}.png", format="png", dpi=300)
