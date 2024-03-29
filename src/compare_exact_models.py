from __future__ import annotations

import json

import numpy as np
from logzero import logger
from tqdm import tqdm

from src.configs import ALGO_CONFIG
from src.data_preprocess.preprocessor import get_label_from_item
from src.optimize.algorithms import ALGORITHMS
from src.optimize.optimizer import Optimizer
from src.optimize.params import RealDataParameter, SyntheticDataParameter
from src.optimize.predictor2model import predictor2model
from src.predict.predictor import PredictorMaker
from src.utils.module_handler import get_object_from_module
from src.utils.paths import GEN_DATA_DIR, OPT_MODEL_DIR, RESULT_DIR


def get_price_candidates(
    data_size: int,
    num_of_items: int,
    price_min: float,
    price_max: float,
    num_of_prices: int,
    seed: int = 0,
) -> list[list[float]]:
    """Generate price candidates for each item randomly"""
    price_candidates = []
    for i in range(data_size):
        # prices = [
        #     round((price_max - price_min) * np.random.rand() + price_min, 3)
        #     for _ in range(num_of_items)
        # ]
        np.random.seed(i + 10000 * seed)
        prices = [
            round(np.random.choice(np.linspace(price_min, price_max, num_of_prices)), 3)
            for _ in range(num_of_items)
        ]
        price_candidates.append(prices)
    return price_candidates


def calc_obj(
    x: dict[str, int],
    constant,
    index_set,
    predictor_name: str,
    algo_name: str = "solver_naive",
    data_param=dict(),
) -> float:
    """Calculate objective value of optimization problem with given prices"""
    x_1: dict[str, int] = {
        k[0]: k[1] for k, v in x.items() if round(v, 2) == 1.0
    }
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
        optimizer = Optimizer(
            model_name="POORT_LH", algo_name=algo_name, data_param=data_param, x=x_1
        )
        optimizer.run(**ALGO_CONFIG[algo_name])
        obj = optimizer.result.objective
    return obj


def average_pred_result(pred_result_dict: dict[str, dict[str, dict[str, float]]]):
    """Calculate average of prediction results"""
    d = dict()
    for item, dict_ in pred_result_dict.items():
        d.setdefault("rmse", dict()).setdefault("train", dict())[item] = dict_["rmse"]["train"]
        d.setdefault("mape", dict()).setdefault("train", dict())[item] = dict_["mape"]["train"]
        d.setdefault("r2", dict()).setdefault("train", dict())[item] = dict_["r2"]["train"]
        try:
            d.setdefault("rmse", dict()).setdefault("test", dict())[item] = dict_["rmse"]["test"]
            d.setdefault("mape", dict()).setdefault("test", dict())[item] = dict_["mape"]["test"]
            d.setdefault("r2", dict()).setdefault("test", dict())[item] = dict_["r2"]["test"]
        except KeyError:
            pass
    d["rmse"]["train"]["mean"] = round(np.mean(list(d["rmse"]["train"].values())), 3)
    d["mape"]["train"]["mean"] = round(np.mean(list(d["mape"]["train"].values())), 3)
    d["r2"]["train"]["mean"] = round(np.mean(list(d["r2"]["train"].values())), 3)
    try:
        d["rmse"]["test"]["mean"] = round(np.mean(list(d["rmse"]["test"].values())), 3)
        d["mape"]["test"]["mean"] = round(np.mean(list(d["mape"]["test"].values())), 3)
        d["r2"]["test"]["mean"] = round(np.mean(list(d["r2"]["test"].values())), 3)
    except KeyError:
        pass
    return d


if __name__ == "__main__":
    # use_predictor_name = "linear_regression"
    use_predictor_name = "ORT_LH"
    # true_predictor_name = "linear_regression"
    true_predictor_name = "ORT_LH"
    use_model_name = predictor2model[use_predictor_name]
    true_model_name = predictor2model[true_predictor_name]
    algo_name = "solver_naive"

    num_iteration = 10
    data_size = 100
    test_data_size = 3000
    noise_variances = [0.2]
    num_of_items_list = [20]
    num_of_prices = 5
    depth_of_trees = 2
    price_min = 0.8
    price_max = 1.0
    num_of_g = 2 * len(num_of_items_list) * 0
    tune = False
    calc_time_only = False

    result_output = dict()
    for noise_variance in noise_variances:
        for num_of_items in num_of_items_list:
            results = []
            for i in tqdm(range(num_iteration)):
                params = SyntheticDataParameter(
                    num_of_items=num_of_items,
                    num_of_prices=num_of_prices,
                    num_of_other_features=num_of_g,
                    depth_of_trees=depth_of_trees,
                    price_min=price_min,
                    price_max=price_max,
                    base_quantity=300,
                    seed=i * 100,
                )
                # Generate parameters of true model
                result_dict = dict()
                logger.info("Generate parameters of true model")
                model_input = Optimizer.make_model_input(
                    model_name=true_model_name, data_param=params
                )
                index_set, constant = model_input.index_set, model_input.constant

                # Calculate optimal prices based on true model
                module_path = OPT_MODEL_DIR / true_model_name / "model.py"
                model_class = get_object_from_module(module_path, "Model")
                model = model_class(index_set=index_set, constant=constant)
                algo_class = ALGORITHMS.get(algo_name, None)
                algorithm = algo_class(model=model, **ALGO_CONFIG[algo_name])
                algorithm.run()
                result_dict["calculation_time"] = algorithm.result.calculation_time

                # Generate data from true model
                logger.info("Generate data from true model")
                if calc_time_only:
                    results.append(result_dict)
                    continue
                module_path = GEN_DATA_DIR / true_predictor_name / "gen_data.py"
                generate_data = get_object_from_module(module_path, "generate_data")

                # Generate training data
                price_candidates_train = get_price_candidates(
                    data_size=data_size,
                    num_of_items=params.num_of_items,
                    price_min=params.price_min,
                    price_max=params.price_max,
                    num_of_prices=num_of_prices,
                    seed=i,
                )
                train_df_dict, q_avg_train = generate_data(
                    price_candidates_train, index_set, constant, noise_variance=noise_variance
                )
                # Generate testing data
                price_candidates_test = get_price_candidates(
                    data_size=test_data_size,
                    num_of_items=params.num_of_items,
                    price_min=params.price_min,
                    price_max=params.price_max,
                    num_of_prices=num_of_prices,
                    seed=i + 10000,
                )
                test_df_dict, q_avg_test = generate_data(
                    price_candidates_test, index_set, constant, noise_variance=noise_variance
                )
                # Bulid demand prediction model for each item
                item2prices = {m: constant.prices for m in index_set.M}
                item2predictor, _pred_result = dict(), dict()
                try:
                    for item, df in train_df_dict.items():
                        target_col = get_label_from_item(item=item)
                        # Build demand prediction model using training data
                        pm = PredictorMaker(
                            predictor_name=use_predictor_name,
                            train_df=df,
                            test_df=test_df_dict[item],
                            target_col=target_col,
                            data_type="synthetic",
                        )
                        predictor = pm.run(train_or_test="train", tune=tune)
                        item2predictor[item] = predictor
                        _pred_result[item] = pm.result
                except RuntimeError:
                    logger.error("predictor error")
                    continue

                # Store prediction results
                pred_result = average_pred_result(_pred_result)
                result_dict["pred"] = pred_result
                result_dict["q_avg_train"] = q_avg_train
                result_dict["q_avg_test"] = q_avg_test

                # Price optimization using demand prediction model
                # Build data parameter
                data_param = RealDataParameter(
                    num_of_prices=params.num_of_prices,
                    item2predictor=item2predictor,
                    item2prices=item2prices,
                    g=constant.g,
                )
                logger.info("Optimized price calculation using the constructed prediction model")
                optimizer = Optimizer(
                    model_name=use_model_name, algo_name=algo_name, data_param=data_param
                )
                optimizer.run(**ALGO_CONFIG[algo_name])

                # Store optimization results
                result_dict["obj"] = {
                    "obj_true_price_true": algorithm.result.objective,
                    "obj_hat_price_hat": optimizer.result.objective,
                    "obj_true_price_hat": calc_obj(
                        x=optimizer.result.variable.x,
                        constant=constant,
                        index_set=index_set,
                        predictor_name=true_predictor_name,
                        data_param=params,
                    ),
                }
                result_dict["opt_price"] = {
                    "price_true": algorithm.result.opt_prices,
                    "price_hat": optimizer.result.opt_prices,
                }
                results.append(result_dict)

                # Evaluation of f^(z^) / f*(z*) and f*(z^) / f*(z*)
                f_ratio_uppers, f_ratio_lowers = [], []
                for result in results:
                    obj_true_price_true = result["obj"]["obj_true_price_true"]
                    obj_hat_price_hat = result["obj"]["obj_hat_price_hat"]
                    obj_true_price_hat = result["obj"]["obj_true_price_hat"]
                    if obj_true_price_true > 0 and obj_hat_price_hat > 0 and obj_true_price_hat > 0:
                        ratio_upper = obj_hat_price_hat / obj_true_price_true
                        f_ratio_uppers.append(ratio_upper)
                        ratio_lower = obj_true_price_hat / obj_true_price_true
                        f_ratio_lowers.append(ratio_lower)

            # Output results
            result_summary = dict()
            result_summary["mean (calculation_time)"] = np.mean(
                [r["calculation_time"] for r in results]
            )
            result_summary["std (calculation_time)"] = np.std(
                [r["calculation_time"] for r in results]
            )
            if calc_time_only:
                json_name = f"./result_d{depth_of_trees}_n{num_of_items}_{use_predictor_name}.json"
                with open(json_name, "w") as fp:
                    json.dump(result_summary, fp)
                result_output[num_of_items] = result_summary
            try:
                print("mean (upper)", np.mean(f_ratio_uppers))
                print("std (upper)", np.std(f_ratio_uppers))
                print("mean (lower)", np.mean(f_ratio_lowers))
                print("std (lower)", np.std(f_ratio_lowers))
                result_summary["mean (upper)"] = np.mean(f_ratio_uppers)
                result_summary["std (upper)"] = np.std(f_ratio_uppers)
                result_summary["mean (lower)"] = np.mean(f_ratio_lowers)
                result_summary["std (lower)"] = np.std(f_ratio_lowers)
                result_summary["mean (mape train)"] = np.mean(
                    [r["pred"]["mape"]["train"]["mean"] for r in results]
                )
                result_summary["std (mape train)"] = np.std(
                    [r["pred"]["mape"]["train"]["mean"] for r in results]
                )
                result_summary["mean (mape test)"] = np.mean(
                    [r["pred"]["mape"]["test"]["mean"] for r in results]
                )
                result_summary["std (mape test)"] = np.std(
                    [r["pred"]["mape"]["test"]["mean"] for r in results]
                )
                # Save results as json
                json_name = (
                    RESULT_DIR / "synthetic" / f"result_{noise_variance}_{use_predictor_name}.json"
                )
                with open(json_name, "w") as fp:
                    json.dump(result_summary, fp)
                result_output[noise_variance] = result_summary
            except NameError:
                pass

    # Save results as csv
    import pandas as pd

    result_df = pd.DataFrame.from_dict(result_output, orient="index").T
    result_df.to_csv(RESULT_DIR / "synthetic" / f"result_{use_predictor_name}_{num_of_items}.json")
    print(result_df)
