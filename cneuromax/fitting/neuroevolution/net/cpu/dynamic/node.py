""":class:`NodeList` & :class:`Node`."""

import random
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Annotated as An
from typing import Any

import numpy as np
from jaxtyping import Float
from numpy.typing import NDArray

from cneuromax.utils.beartype import ge


@dataclass
class NodeList:
    """Holds :class:`.neuroevolution.net.dynamic.net.Net` nodes.

    Args:
        all: All existing/current network nodes.
        input: There are as many input nodes as there are input\
            signals. Each input node is assigned an input value and\
            forwards it to nodes that it connects to. Input nodes are\
            non-parametric.
        hidden: Hidden nodes are parametric nodes that receive/emits\
            signal(s) from/to any number of nodes.
        output: Same properties as hidden nodes, but also emit a\
            signal outside the network.
        receiving: List of nodes that are receiving information from a\
            source. Nodes appear in this list once per source.
        emitting: List of nodes that are emitting information to a\
            target. Nodes appear in this list once per target.
        being_pruned: List of nodes currently being pruned. As a\
            pruning operation can kicksart a series of other pruning\
            operations, this list is used to prevent infinite loops.
        layered: List of lists of nodes, where each list is indexed by\
            the layer it belongs to. Input nodes are in the first\
            layer, output nodes are in the last layer, and hidden\
            nodes are in between.
    """

    all: list["Node"] = field(default_factory=list)
    input: list["Node"] = field(default_factory=list)
    hidden: list["Node"] = field(default_factory=list)
    output: list["Node"] = field(default_factory=list)
    receiving: list["Node"] = field(default_factory=list)
    emitting: list["Node"] = field(default_factory=list)
    being_pruned: list["Node"] = field(default_factory=list)
    layered: list[list["Node"]] = field(default_factory=list)

    def __iter__(
        self: "NodeList",
    ) -> Iterator[list["Node"] | list[list["Node"]]]:
        """See return.

        Returns:
            An iterator over all lists of nodes.
        """
        return iter(
            [
                self.all,
                self.input,
                self.hidden,
                self.output,
                self.receiving,
                self.emitting,
                self.being_pruned,
                self.layered,
            ],
        )


class Node:
    """Node with full functionality.

    Used when weights & biases are not vectorized and thus stored\
    in the node itself.

    Args:

    """

    def __init__(self: "Node", type: str, id: int):
        """Constructor"""
        self.id: int = id
        self.type: str = type
        self.in_nodes: list[Node] = []
        self.out_nodes: list[Node] = []
        self.output: Float[np.ndarray, " num_out_nodes"] = np.ndarray([0])
        if self.type != "input":
            self.initialize_parameters()

    def __repr__(self: "Node") -> str:
        in_node_ids: tuple[int, ...] = tuple(
            [node.id for node in self.in_nodes]
        )
        out_node_ids: tuple[int, ...] = tuple(
            [node.id for node in self.out_nodes]
        )

        if self.type == "input":
            return str(("x",)) + "->" + str(self.id) + "->" + str(out_node_ids)

        elif self.type == "hidden":
            return (
                str(in_node_ids)
                + "->"
                + str(self.id)
                + "->"
                + str(out_node_ids)
            )

        else:  # self.type == 'output':
            return (
                str(in_node_ids)
                + "->"
                + str(self.id)
                + "->"
                + str(("y",) + out_node_ids)
            )

    def initialize_parameters(self: "Node"):
        self.weights: Float[np.ndarray, " num_in_nodes"] = np.empty(0)
        self.bias: Float[np.ndarray, " 1"] = (
            np.random.randn(1) if self.type == "hidden" else np.zeros(1)
        )

    def connect_to(self: "Node", node: "Node") -> None:
        new_weight: Float[np.ndarray, " 1"] = np.random.randn(1)
        node.weights: Float[np.ndarray, " node_num_in_nodes+1"] = (
            np.concatenate((node.weights, new_weight))
        )

        self.out_nodes.append(node)
        node.in_nodes.append(self)

    def disconnect_from(self: "Node", node: "Node") -> None:
        if self not in node.in_nodes:
            return True

        i: int = node.in_nodes.index(self)
        node.weights = np.concatenate(
            (node.weights[:i], node.weights[i + 1 :])
        )

        self.out_nodes.remove(node)
        node.in_nodes.remove(self)

        return False

    def compute(self: "Node") -> None:
        x = np.ndarray([node.output for node in self.in_nodes]).squeeze()

        x = np.dot(x, self.weights) + self.bias

        x = np.clip(x, 0, 2**31 - 1)

        self.cached_output = x

    def update(self: "Node") -> None:
        self.output = self.cached_output
        self.cached_output = None
