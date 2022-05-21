from __future__ import annotations

from typing import Optional


def node2parent(node: int) -> Optional[int]:
    """親ノードを取得"""
    if node == 0:
        parent = None
    # ノードが奇数であれば1引いて2で割ると親ノードになる
    elif node % 2 == 1:
        parent = int((node - 1) / 2)
    # ノードが偶数であれば2引いて2で割ると親ノードになる
    elif node % 2 == 0:
        parent = int((node - 2) / 2)
    return parent


def leaf2root(leaf_node: int) -> list[int]:
    """葉ノードから根ノードまでのパスを抽出"""
    node_path = [leaf_node]
    node = leaf_node
    while node > 0:
        node = node2parent(node)
        node_path.append(node)
    node_path.sort()
    return node_path


def leaf2LtRt(leaf_node: int) -> tuple[list[int], list[int]]:
    """葉ノードに至るまでに根ノードから左分岐と右分岐したノードを抽出"""
    Lt, Rt = [], []
    nodes = leaf2root(leaf_node)
    for node in nodes:
        parent = node2parent(node)
        if node == 0:
            continue
        # 奇数のノードには左に分岐して至る
        elif node % 2 == 1:
            Lt.append(parent)
        # 偶数のノードには右に分岐して至る
        elif node % 2 == 0:
            Rt.append(parent)
    Lt.sort()
    Rt.sort()
    return Lt, Rt


def depth2leaves(depth: int) -> list[int]:
    """木の深さから葉ノードを抽出"""
    leaf_nodes = list(range(2 ** (depth) - 1, 2 ** (depth + 1) - 1))
    return leaf_nodes


def depth2allnodes(depth: int) -> list[int]:
    """木の深さから全ノードを抽出"""
    all_nodes = list(range(2 ** (depth + 1) - 1))  # 等比数列の和の公式
    return all_nodes


def depth2branchnodes(depth: int) -> list[int]:
    """木の深さから枝ノードを抽出"""
    all_nodes = depth2allnodes(depth=depth)
    leaf_nodes = depth2leaves(depth=depth)
    branch_nodes = list(set(all_nodes) - set(leaf_nodes))
    if all_nodes == [0] and leaf_nodes == [0]:
        branch_nodes = [0]
    branch_nodes.sort()
    return branch_nodes
