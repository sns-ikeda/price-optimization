from typing import Dict, List

import yaml
from tqdm import tqdm

from src.models.po_L.make_input import make_sample_input
from src.models.po_L.model import Model
from src.models.po_L.params import Parameter
from src.processing.models2time import models2avg_cal_time
from src.utils.paths import DATA_DIR
from src.utils.plot import save_image


def main():
    # configファイルからシミュレーションの設定を取得
    with open("simulation_config.yaml") as file:
        config = yaml.safe_load(file.read())

    num_of_items: List[int] = config["params"]["num_of_items"]
    num_of_prices: int = config["params"]["num_of_prices"]
    num_of_other_features: int = config["params"]["num_of_other_features"]
    depth_of_trees: int = config["params"]["depth_of_trees"]
    base_price: int = config["params"]["base_price"]
    num_of_simulations: int = config["params"]["num_of_simulations"]
    time_limit: int = config["params"]["time_limit"]

    # シミュレーションを実行
    models_dict = dict()
    for i in tqdm(num_of_items):
        models = []
        for _ in range(num_of_simulations):
            # モデルで使用するパラメータ
            params = Parameter(
                num_of_items=i,
                num_of_prices=num_of_prices,
                num_of_other_features=num_of_other_features,
                depth_of_trees=depth_of_trees,
                base_price=base_price,
                num_of_simulations=num_of_simulations,
            )
            # パラメータからモデルの入力を作成
            index_set, constant = make_sample_input(params=params)
            model = Model(index_set=index_set, constant=constant)
            model.solve()
            models.append(model)
        models_dict[i] = models

    # 計算時間の後処理
    calculation_time_dict: Dict[int, float] = models2avg_cal_time(models_dict=models_dict)
    image_name = "calculation_time"
    save_image(
        calculation_time_dict=calculation_time_dict,
        dir_path=DATA_DIR / "output" / "result",
        image_name=image_name,
        time_limit=time_limit,
    )


if __name__ == "__main__":
    main()
