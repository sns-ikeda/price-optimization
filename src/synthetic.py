from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.configs import ALGO_CONFIG
from src.data_preprocess.preprocessor import get_label_from_item
from src.evaluate.evaluator import Evaluator
from src.optimize.models.POLR.make_input import make_artificial_input
from src.optimize.optimizer import Optimizer
from src.optimize.params import ArtificialDataParameter, RealDataParameter
from src.optimize.predictor2model import predictor2model
from src.predict.predictor import PredictorMaker
from src.simulator import postproceess_pred_result
from src.utils.module_handler import get_object_from_module
from src.utils.paths import GEN_DATA_DIR


def get_price_candidates(
    data_size: int, num_of_items: int, price_min: int, price_max: int
) -> list[list[float]]:
    price_candidates = []
    for _ in range(data_size):
        prices = [
            round((price_max - price_min) * np.random.rand() + price_min, 2)
            for _ in range(num_of_items)
        ]
        price_candidates.append(prices)
    return price_candidates


if __name__ == "__main__":
    predictor_name = "linear_regression"
    # predictor_name = "ORT_LH"
    _predictor_name = "linear_regression"
    algo_name = "solver_naive"
    num_iteration = 1
    data_size = 1000
    train_size = 0.5
    test_size = round(1 - train_size, 3)

    results = []
    for i in range(num_iteration):
        # データ生成
        params = ArtificialDataParameter(
            num_of_items=5,
            num_of_prices=5,
            num_of_other_features=0,
            depth_of_trees=3,
            base_price=5,
            price_min=4,
            price_max=6,
            base_quantity=300,
            seed=i,
            data_type="artificial",
        )
        index_set, constant = make_artificial_input(params=params)
        price_candidates = get_price_candidates(
            data_size=data_size,
            num_of_items=params.num_of_items,
            price_min=int(params.base_price * 0.5),
            price_max=int(params.base_price * 1.5),
        )
        module_path = GEN_DATA_DIR / _predictor_name / "gen_data.py"
        generate_data = get_object_from_module(module_path, "generate_data")
        df_dict = generate_data(price_candidates, constant, noise_variance=0.05)

        # 予測
        item2prices = {m: constant.prices for m in index_set.M}
        item2predictor_train, item2predictor_test = dict(), dict()
        _pred_result_train, _pred_result_test = dict(), dict()
        result_dict = dict()
        for item, df in df_dict.items():
            train_df, test_df = train_test_split(
                df, train_size=train_size, test_size=test_size, shuffle=False
            )
            target_col = get_label_from_item(item=item)
            if item == "0":
                test_df_ = test_df
            else:
                col = "UNITS_" + item
                test_df_ = pd.concat([test_df_, test_df[col]], axis=1)

            # 訓練データに対する予測モデルを構築
            pm_train = PredictorMaker(
                predictor_name=predictor_name,
                train_df=train_df,
                test_df=test_df,
                target_col=target_col,
                data_type="synthetic",
            )
            predictor_train = pm_train.run(train_or_test="train", suffix=str(train_size))
            item2predictor_train[item] = predictor_train
            _pred_result_train[item] = pm_train.result

            # 検証データに対する予測モデルを構築
            pm_test = PredictorMaker(
                predictor_name=_predictor_name,
                train_df=test_df,
                target_col=target_col,
                data_type="synthetic",
            )
            predictor_test = pm_test.run(train_or_test="test", suffix=str(train_size))
            item2predictor_test[item] = predictor_test
            _pred_result_test[item] = pm_test.result

            pred_result_train = postproceess_pred_result(_pred_result_train)
            pred_result_test = postproceess_pred_result(_pred_result_test)
            result_dict["train"] = pred_result_train
            result_dict["test"] = pred_result_test

        # 最適化
        # 訓練データに対して計算
        data_param_train = RealDataParameter(
            num_of_prices=params.num_of_prices,
            item2predictor=item2predictor_train,
            item2prices=item2prices,
            g={},
        )
        model_name = predictor2model[predictor_name]
        optimizer_train = Optimizer(model_name=model_name, algo_name=algo_name, data_param=data_param_train)
        optimizer_train.run(**ALGO_CONFIG[algo_name])
        # テストデータに対して計算
        data_param_test = RealDataParameter(
            num_of_prices=params.num_of_prices,
            item2predictor=item2predictor_test,
            item2prices=item2prices,
            g={},
        )
        model_name = predictor2model[predictor_name]
        optimizer_test = Optimizer(model_name=model_name, algo_name=algo_name, data_param=data_param_test)
        optimizer_test.run(**ALGO_CONFIG[algo_name])

        # 評価
        # 訓練データから算出した最適価格の評価
        evaluator_train = Evaluator(
            test_df=test_df_,
            item2predictor=item2predictor_test,
            opt_prices=optimizer_train.result.opt_prices,
            item2prices=item2prices,
        )
        evaluator_train.run()
        # テストデータから算出した最適価格の評価
        evaluator_test = Evaluator(
            test_df=test_df_,
            item2predictor=item2predictor_test,
            opt_prices=optimizer_test.result.opt_prices,
            item2prices=item2prices,
        )
        evaluator_test.run()
        result_dict["opt_train"] = evaluator_train.opt_prices
        result_dict["opt_test"] = evaluator_test.opt_prices
        result_dict["eval"] = evaluator_train.result
        result_dict["eval"]["theoretical_sales"] = evaluator_test.result["pred_sales_at_opt_price"]
        results.append(result_dict)
