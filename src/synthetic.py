from __future__ import annotations

import numpy as np
from logzero import logger
from tqdm import tqdm

from src.configs import ALGO_CONFIG
from src.data_preprocess.preprocessor import get_label_from_item
from src.optimize.algorithms import ALGORITHMS
from src.optimize.optimizer import Optimizer
from src.optimize.params import ArtificialDataParameter, RealDataParameter
from src.optimize.predictor2model import predictor2model
from src.predict.predictor import PredictorMaker
from src.simulator import postproceess_pred_result
from src.utils.module_handler import get_object_from_module
from src.utils.paths import GEN_DATA_DIR, OPT_MODEL_DIR


def get_price_candidates(
    data_size: int, num_of_items: int, price_min: float, price_max: float, num_of_prices: int
) -> list[list[float]]:
    price_candidates = []
    for i in range(data_size):
        # prices = [
        #     round((price_max - price_min) * np.random.rand() + price_min, 3)
        #     for _ in range(num_of_items)
        # ]
        np.random.seed(i)
        prices = [
            np.random.choice(np.linspace(price_min, price_max, num_of_prices))
            for _ in range(num_of_items)
        ]
        price_candidates.append(prices)
    return price_candidates


def calc_obj(
    x: dict[str, int], constant, index_set, predictor_name: str = "linear_regression"
) -> float:
    x_1 = {k[0]: k[1] for k, v in x.items() if round(v, 2) == 1.0}  # x_mk = 1となるm: k
    assert len(x_1) == len(index_set.M), "The length of optimal prices is not enough."

    if predictor_name == "linear_regression":
        p = np.array([constant.P[(m, k)] for m, k in x_1.items()])
        q = []
        for m in index_set.M:
            q_m = constant.beta0[m]
            for mp, k in x_1.items():
                q_m += constant.beta[m, mp] * constant.P[mp, k]
            for d in index_set.D[m]:
                q_m += constant.beta[m, d] * constant.g[m, d]
            q.append(q_m)
        q = np.array(q)
        obj = np.dot(p, q)
    elif predictor_name == "ORT_LH":
        optimizer = Optimizer(model_name=use_model_name, algo_name=algo_name, data_param=data_param)
        optimizer.run(**ALGO_CONFIG[algo_name])
        obj = optimizer.result.objective
    return obj


if __name__ == "__main__":
    # use_predictor_name = "linear_regression"
    use_predictor_name = "ORT_LH"
    true_predictor_name = "linear_regression"
    # true_predictor_name = "ORT_LH"
    use_model_name = predictor2model[use_predictor_name]
    true_model_name = predictor2model[true_predictor_name]
    algo_name = "solver_naive"

    num_iteration = 10
    data_size = 500
    noise_variance = 0.2
    num_of_items = 5
    num_of_prices = 5
    depth_of_trees = 2
    base_price = 5
    price_min = 0.8
    price_max = 1.0
    num_of_g = 2 * num_of_items * 0
    tune = True

    results = []
    for i in tqdm(range(num_iteration)):
        # データ生成
        params = ArtificialDataParameter(
            num_of_items=num_of_items,
            num_of_prices=num_of_prices,
            num_of_other_features=num_of_g,
            depth_of_trees=depth_of_trees,
            base_price=base_price,
            price_min=price_min,
            price_max=price_max,
            base_quantity=300,
            seed=i,
            data_type="artificial",
        )
        price_candidates = get_price_candidates(
            data_size=data_size,
            num_of_items=params.num_of_items,
            price_min=params.price_min,
            price_max=params.price_max,
            num_of_prices=num_of_prices,
        )
        logger.info("真のモデルのパラメータ")
        model_input = Optimizer.make_model_input(model_name=true_model_name, data_param=params)
        index_set, constant = model_input.index_set, model_input.constant

        module_path = GEN_DATA_DIR / true_predictor_name / "gen_data.py"
        generate_data = get_object_from_module(module_path, "generate_data")
        df_dict = generate_data(
            price_candidates, index_set, constant, noise_variance=noise_variance
        )
        print(list(df_dict.values())[0].head(5))

        # 真のモデルに対する最適価格の計算
        module_path = OPT_MODEL_DIR / true_model_name / "model.py"
        model_class = get_object_from_module(module_path, "Model")
        model = model_class(index_set=index_set, constant=constant)
        algo_class = ALGORITHMS.get(algo_name, None)
        algorithm = algo_class(model=model, **ALGO_CONFIG[algo_name])
        algorithm.run()

        # 予測
        item2prices = {m: constant.prices for m in index_set.M}
        item2predictor = dict()
        _pred_result, result_dict = dict(), dict()
        try:
            for item, df in df_dict.items():
                target_col = get_label_from_item(item=item)
                # 訓練データに対する予測モデルを構築
                pm = PredictorMaker(
                    predictor_name=use_predictor_name,
                    train_df=df,
                    target_col=target_col,
                    data_type="synthetic",
                )
                predictor = pm.run(train_or_test="train", tune=tune)
                item2predictor[item] = predictor
                _pred_result[item] = pm.result
        except RuntimeError:
            logger.error("predictor error")
            continue
        pred_result = postproceess_pred_result(_pred_result)
        result_dict["pred"] = pred_result

        # 最適化
        # 訓練データに対して計算
        data_param = RealDataParameter(
            num_of_prices=params.num_of_prices,
            item2predictor=item2predictor,
            item2prices=item2prices,
            g=constant.g,
        )
        logger.info("予測モデルの最適化価格計算")
        optimizer = Optimizer(model_name=use_model_name, algo_name=algo_name, data_param=data_param)
        optimizer.run(**ALGO_CONFIG[algo_name])
        result_dict["obj"] = {
            "obj_true_price_true": algorithm.result.objective,
            "obj_hat_price_hat": optimizer.result.objective,
            "obj_true_price_hat": calc_obj(
                x=optimizer.result.variable.x,
                constant=constant,
                index_set=index_set,
                predictor_name=true_predictor_name,
            ),
        }
        result_dict["opt_price"] = {
            "price_true": algorithm.result.opt_prices,
            "price_hat": optimizer.result.opt_prices,
        }
        results.append(result_dict)

    # f^(z^) / f*(z*), f*(z^) / f*(z*) の計算結果
    f_ratio_uppers, f_ratio_lowers = [], []
    for result in results:
        obj_true_price_true = result["obj"]["obj_true_price_true"]
        obj_hat_price_hat = result["obj"]["obj_hat_price_hat"]
        obj_true_price_hat = result["obj"]["obj_true_price_hat"]
        if obj_true_price_true > 0 and obj_hat_price_hat > 0 and obj_true_price_hat > 0:
            print(result["obj"])
            ratio_upper = obj_hat_price_hat / obj_true_price_true
            f_ratio_uppers.append(ratio_upper)
            ratio_lower = obj_true_price_hat / obj_true_price_true
            f_ratio_lowers.append(ratio_lower)

    # 計算結果の出力
    print("mean (upper)", np.mean(f_ratio_uppers))
    print("std (upper)", np.std(f_ratio_uppers))

    print("mean (lower)", np.mean(f_ratio_lowers))
    print("std (lower)", np.std(f_ratio_lowers))