from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.configs import DATA_CONFIG


@dataclass(frozen=True)
class Config:
    target_col: str
    base_cols: list[str]
    master_cols: list[str]
    store_num: Optional[int] = None
    category: Optional[str] = None
    manufacturer: Optional[str] = None


config = Config(**DATA_CONFIG["breakfast"])
