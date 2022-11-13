from src.optimize.params import ArtificialDataParameter
from src.optimize.optimizer import Optimizer
from src.gen_data.ORT_LH.gen_data import generate_data


def test_generate_data():
    params = ArtificialDataParameter(
        num_of_items=3,
        num_of_prices=3,
        num_of_other_features=0,
        depth_of_trees=2,
        base_price=5,
        price_min=1,
        price_max=3,
        base_quantity=300,
        seed=0,
        data_type="artificial",
    )
    price_candidates = [
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 2.0],
        [1.0, 1.0, 3.0],
        [1.0, 2.0, 1.0],
        [1.0, 2.0, 2.0],
        [1.0, 2.0, 3.0],
        [1.0, 3.0, 1.0],
        [1.0, 3.0, 2.0],
        [1.0, 3.0, 3.0],
        [2, 1, 1],
        [2, 1, 2],
        [2, 1, 3],
        [2, 2, 1],
        [2, 2, 2],
        [2, 2, 3],
        [2, 3, 1],
        [2, 3, 2],
        [2, 3, 3],
        [3, 1, 1],
        [3, 1, 2],
        [3, 1, 3],
        [3, 2, 1],
        [3, 2, 2],
        [3, 2, 3],
        [3, 3, 1],
        [3, 3, 2],
        # [3, 3, 3],
    ]
    model_input = Optimizer.make_model_input(model_name="POORT_LH", data_param=params)
    index_set, constant = model_input.index_set, model_input.constant
    a = {
        ("0", "0", 0): 1.0,
        ("0", "1", 0): 2.0,
        ("0", "2", 0): 3.0,
        ("0", "0", 1): 2.0,
        ("0", "1", 1): 1.0,
        ("0", "2", 1): 3.0,
        ("0", "0", 2): 3.0,
        ("0", "1", 2): 2.0,
        ("0", "2", 2): 1.0,
    }
    b = {
        ("0", 0): 12.0,
        ("0", 1): 12.0,
        ("0", 2): 12.0,
    }
    beta = {
        ("0", "0", 3): -1,
        ("0", "1", 3): 1,
        ("0", "2", 3): 2,
        ("0", "0", 4): -10,
        ("0", "1", 4): 10,
        ("0", "2", 4): 20,
        ("0", "0", 5): -100,
        ("0", "1", 5): 100,
        ("0", "2", 5): 200,
        ("0", "0", 6): -1000,
        ("0", "1", 6): 1000,
        ("0", "2", 6): 2000,
    }
    beta0 = {
        ("0", 3): 0,
        ("0", 4): 0,
        ("0", 5): 0,
        ("0", 6): 0,
    }
    q_0 = [
        2,
        4,
        600,
        3,
        5,
        700,
        4,
        600,
        8000,
        1,
        3,
        500,
        2,
        4000,
        6000,
        3,
        5000,
        7000,
        0,
        20,
        4000,
        1,
        3000,
        5000,
        2000,
        4000,
        6000
    ]
    constant.a.update(a)
    constant.b.update(b)
    constant.a.update(a)
    constant.beta.update(beta)
    constant.beta0.update(beta0)

    df_dict = generate_data(
        price_candidates, index_set, constant, noise_variance=0
    )
    q_0_hat = df_dict["0"]["UNITS_0"].to_list()
    for q, q_hat in zip(q_0, q_0_hat):
        assert q == q_hat
