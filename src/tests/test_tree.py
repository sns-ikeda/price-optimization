from src.processing.binary_tree import (
    depth2allnodes,
    depth2branchnodes,
    depth2leaves,
    leaf2LtRt,
    leaf2root,
    node2parent,
)


def test_node2parent():
    parent = node2parent(node=0)
    assert parent is None
    parent = node2parent(node=1)
    assert parent == 0
    parent = node2parent(node=2)
    assert parent == 0
    parent = node2parent(node=3)
    assert parent == 1
    parent = node2parent(node=4)
    assert parent == 1
    parent = node2parent(node=5)
    assert parent == 2
    parent = node2parent(node=6)
    assert parent == 2


def test_leaf2root():
    node_path = leaf2root(leaf_node=0)
    assert node_path == [0]
    node_path = leaf2root(leaf_node=1)
    assert node_path == [0, 1]
    node_path = leaf2root(leaf_node=2)
    assert node_path == [0, 2]
    node_path = leaf2root(leaf_node=3)
    assert node_path == [0, 1, 3]
    node_path = leaf2root(leaf_node=4)
    assert node_path == [0, 1, 4]
    node_path = leaf2root(leaf_node=5)
    assert node_path == [0, 2, 5]
    node_path = leaf2root(leaf_node=6)
    assert node_path == [0, 2, 6]


def test_leaf2LtRt():
    L_t, R_t = leaf2LtRt(leaf_node=0)
    assert L_t == [] and R_t == []
    L_t, R_t = leaf2LtRt(leaf_node=1)
    assert L_t == [0] and R_t == []
    L_t, R_t = leaf2LtRt(leaf_node=2)
    assert L_t == [] and R_t == [0]
    L_t, R_t = leaf2LtRt(leaf_node=3)
    assert L_t == [0, 1] and R_t == []
    L_t, R_t = leaf2LtRt(leaf_node=4)
    assert L_t == [0] and R_t == [1]
    L_t, R_t = leaf2LtRt(leaf_node=5)
    assert L_t == [2] and R_t == [0]
    L_t, R_t = leaf2LtRt(leaf_node=6)
    assert L_t == [] and R_t == [0, 2]


def test_depth2leaves():
    leaf_nodes = depth2leaves(depth=0)
    assert leaf_nodes == [0]
    leaf_nodes = depth2leaves(depth=1)
    assert leaf_nodes == [1, 2]
    leaf_nodes = depth2leaves(depth=2)
    assert leaf_nodes == [3, 4, 5, 6]


def test_depth2allnodes():
    all_nodes = depth2allnodes(depth=0)
    assert all_nodes == [0]
    all_nodes = depth2allnodes(depth=1)
    assert all_nodes == [0, 1, 2]
    all_nodes = depth2allnodes(depth=2)
    assert all_nodes == [0, 1, 2, 3, 4, 5, 6]


def test_depth2branchnodes():
    all_nodes = depth2branchnodes(depth=0)
    assert all_nodes == [0]
    all_nodes = depth2branchnodes(depth=1)
    assert all_nodes == [0]
    all_nodes = depth2branchnodes(depth=2)
    assert all_nodes == [0, 1, 2]
    all_nodes = depth2branchnodes(depth=3)
    assert all_nodes == [0, 1, 2, 3, 4, 5, 6]
