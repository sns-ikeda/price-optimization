from __future__ import annotations


def get_opt_prices(
    x: dict[tuple[str, int], float], P: dict[tuple[str, int], float]
) -> dict[str, float]:
    opt_prices = dict()
    for k_tuple, v in x.items():
        if round(v, 2) == 1.0:
            item = k_tuple[0]
            opt_prices[item] = P[k_tuple]
    return opt_prices
