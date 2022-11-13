# from src.evaluate.evaluator import Evaluator
# from src.simulator import Simulator


# def test_ORT_LH():
#     data_type = "realworld"
#     train_size = 0.5
#     config_data = {
#         "realworld": {
#             "breakfast": {
#                 "store_num": 25027,
#                 "category": "COLD CEREAL",
#                 "manufacturer": "POST FOODS",
#                 "target_col": "UNITS",
#                 "base_cols": ["YEAR", "MONTH", "DAY", "PRICE", "DESCRIPTION", "UNITS"],
#                 "master_cols": ["YEAR", "MONTH", "DAY"],
#                 "num_of_prices": 5,
#             }
#         }
#     }
#     config_opt = {"model": {"POORT_LH": {"algorithm": ["solver_naive"], "prediction": "ORT_LH"}}}
#     config_algo = {"solver_naive": {"solver": "Cbc", "TimeLimit": 600}}
#     config_pred = {}
#     simulator = Simulator(
#         data_type=data_type,
#         config_data=config_data,
#         config_opt=config_opt,
#         config_algo=config_algo,
#         config_pred=config_pred,
#     )
#     simulator.run(train_size=train_size)
#     objective1 = simulator.optimizers[("breakfast", "POORT_LH", "solver_naive")].result.objective

#     evaluator = Evaluator(
#         test_df=simulator.train_df.head(1),
#         item2predictor=simulator.train_predictors["breakfast", "ORT_LH"].item2predictor,
#         opt_prices=simulator.evaluators[("breakfast", "POORT_LH", "solver_naive")].opt_prices,
#         avg_prices={},
#         item2prices=simulator.evaluators[("breakfast", "POORT_LH", "solver_naive")].item2prices,
#     )
#     evaluator.run()
#     objective2 = evaluator.result["pred_sales_at_opt_price"]
#     objective3 = evaluator.result["theoretical_sales"]
#     assert int(objective1) == int(objective2) == int(objective3)
