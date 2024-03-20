import numpy as np


class SparseWeightMatrix:
    """
    Dynamic Sparse Network Weight Matrix.
    """

    def __init__(self):
        self.row = np.ndarray([])
        self.col = np.ndarray([])
        self.data = np.ndarray([])

        self.nodes = []

    def add_node(self, node):
        self.nodes.append(node)

    def add_connection(self, in_node, out_node):
        in_node_idx = self.nodes.idx(in_node)
        out_node_idx = self.nodes.idx(out_node)

        self.row = np.append(self.row, in_node_idx)
        self.col = np.append(self.col, out_node_idx)
        self.data = np.append(self.data, np.random.randn())

    def remove_node(self, node):
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
