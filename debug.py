from src.configs import CONFIG_ALGO, CONFIG_DATA, CONFIG_OPT, CONFIG_PRED, CONFIG_SIM
from src.simulator import Simulator

data_type = "realworld"
num_iteration = CONFIG_SIM["num_iteration"]
simulator = Simulator(
    data_type=data_type,
    config_data=CONFIG_DATA,
    config_opt=CONFIG_OPT,
    config_algo=CONFIG_ALGO,
    config_pred=CONFIG_PRED,
)
simulator.run(num_iteration)
