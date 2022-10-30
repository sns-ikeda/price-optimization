from __future__ import annotations


def rename_feature(feature: str, prefixes: list[str] = ["PRICE", "G"]) -> str:
    for prefix in prefixes:
        if prefix in feature:
            renamed_feature = feature.split("_")[-1]
            return renamed_feature
    return feature


def rename_dict(
    target_dict: dict[str, float], prefixes: list[str] = ["PRICE", "G"]
) -> dict[str, float]:
    renamed_dict = dict()
    for k, v in target_dict.items():
        renamed_k = rename_feature(k, prefixes=prefixes)
        renamed_dict[renamed_k] = v
    return renamed_dict
