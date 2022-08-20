from __future__ import annotations

from src.configs import CONFIG_ALGO, CONFIG_DATA, CONFIG_OPT, CONFIG_PRED
from src.simulator import Simulator


def tune():
    simulator = Simulator(
        data_type="realworld",
        config_data=CONFIG_DATA,
        config_opt=CONFIG_OPT,
        config_algo=CONFIG_ALGO,
        config_pred=CONFIG_PRED,
    )
    simulator.tune_params()


if __name__ == "__main__":
    tune()
