from __future__ import annotations

from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import seaborn as sns


def save_image(
    calculation_time_dict: dict[int, float],
    dir_path: Union[str, Path],
    image_name: str,
    time_limit: int,
) -> None:
    x = list(calculation_time_dict.keys())
    y = list(calculation_time_dict.values())
    plt.style.use("default")
    sns.set()
    sns.set_style("whitegrid")
    sns.set_palette("gray")
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(x, y, marker="o")

    ax.set_xlabel("num of items")
    ax.set_ylabel("calculation time [sec]")
    ax.set_ylim(0, time_limit)
    fig.savefig(Path(dir_path) / f"{image_name}.png", format="png", dpi=300)
