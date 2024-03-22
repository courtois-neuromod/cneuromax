import secrets
from typing import TypeVar

import numpy as np


class Node:  # noqa: D101

    def __init__(self: "Node") -> None:
        self.in_nodes: list[Node] = []
        self.out_nodes: list[Node] = []


T = TypeVar("T", bound=float)


class SparseWeightMatrix:
    """Holds :class:`~.dynamic.net.Net` weight values."""

    def __init__(self: "SparseWeightMatrix") -> None:
        self.row: list[int] = []
        self.col: list[int] = []
        self.data: list[float] = []
        self.nodes: list[Node] = []

    def add_node(self: "SparseWeightMatrix", node: Node) -> None:
        """Self-explanatory.

        Args:
            node: Node to add to the matrix.
        """
        self.nodes.append(node)

    def add_connection(
        self: "SparseWeightMatrix",
        in_node: Node,
        out_node: Node,
    ) -> None:
        """Adds a connection between two nodes.

        Args:
            in_node: Node to connect from.
            out_node: Node to connect to.
        """
        in_node_idx = self.nodes.index(in_node)
        out_node_idx = self.nodes.index(out_node)

        self.row.append(in_node_idx)
        self.col.append(out_node_idx)
        self.data.append(secrets.choice([True, False]))

    def remove_node(self: "SparseWeightMatrix", node):
        node_idx = self.nodes.idx(node)

        self.nodes.remove(node)

        self.data = np.delete(self.data, np.where(self.row == node_idx))
        self.col = np.delete(self.col, np.where(self.row == node_idx))
        self.row = np.delete(self.row, np.where(self.row == node_idx))

        self.data = np.delete(self.data, np.where(self.col == node_idx))
        self.row = np.delete(self.row, np.where(self.col == node_idx))
        self.col = np.delete(self.col, np.where(self.col == node_idx))

        self.row -= self.row > node_idx
        self.col -= self.col > node_idx


class BiasVector:
    """
    Dynamic Sparse Network Bias Vector.
    """

    def __init__(self):
        self.data = np.ndarray([])

        self.nodes = []

    def add_node(self, node, type):
        self.nodes.append(node)

        self.data = np.append(
            self.data, np.random.randn() if type == "hidden" else 0
        )

    def remove_node(self, node):
        node_idx = self.nodes.idx(node)

        self.nodes.remove(node)

        self.data = np.delete(self.data, node_idx)
