from __future__ import annotations

from src.optimize.models.POORT_L.make_input import make_artificial_input as _make_artificial_input
from src.optimize.models.POORT_L_alpha.model import Constant, IndexSet
from src.optimize.params import ArtificialDataParameter, RealDataParameter


def make_artificial_input(params: ArtificialDataParameter) -> tuple[IndexSet, Constant]:
    """人工的にモデルのパラメータを生成"""
    index_set, constant = _make_artificial_input(params=params)
    return index_set, constant


def make_real_input(params: RealDataParameter) -> tuple[IndexSet, Constant]:
    """実際のデータからモデルのパラメータを生成"""
