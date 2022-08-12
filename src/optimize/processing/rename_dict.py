from __future__ import annotations


def rename_dict(target_dict: dict[str, float], prefix: str = "PRICE") -> dict[str, float]:
    renamed_dict = dict()
    for k, v in target_dict.items():
        if prefix in k:
            renamed_k = k.split("_")[-1]
            renamed_dict[renamed_k] = v
        else:
            renamed_dict[k] = v
    return renamed_dict
