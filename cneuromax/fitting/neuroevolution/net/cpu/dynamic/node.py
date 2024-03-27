""":class:`NodeList` & :class:`Node`."""

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Annotated as An

import numpy as np

from cneuromax.utils.beartype import one_of


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
    """Node (Neuron) for :class:`~.dynamic.net.Net`.

    Args:
        role: Node function in the network.
        id_: Node unique identifier.
    """

    def __init__(
        self: "Node",
        role: An[str, one_of("input", "hidden", "output")],
        id_: int,
    ) -> None:
        self.id_: int = id_
        self.role: str = role
        self.in_nodes: list[Node] = []
        self.out_nodes: list[Node] = []
        self.output: float = 0
        if self.role != "input":
            self.initialize_parameters()

    def __repr__(self: "Node") -> str:
        """.

        Returns:
            A string representation of the node.
        """
        node_inputs: tuple[int, str, ...] = tuple(
            "x" if self.role == "input" else node.id_ for node in self.in_nodes
        )
        node_outputs: tuple[int, str, ...] = tuple(
            node.id_ for node in self.out_nodes
        )
        if self.role == "output":
            node_outputs = ("y", *node_outputs)
        return (
            str(node_inputs)
            + "->"
            + str(self.id_)
            + "->"
            + str(("y", *node_outputs))
        )

    def initialize_parameters(self: "Node") -> None:
        """Initialize weights and biases."""
        self.weights: list[float] = []
        self.bias: float = (
            float(np.random.randn()) if self.role == "hidden" else 0
        )

    def connect_to(self: "Node", node: "Node") -> None:
        """Connect to another node.

        Args:
            node: Node to connect to.
        """
        new_weight: float = float(np.random.randn())
        node.weights.append(new_weight)
        self.out_nodes.append(node)
        node.in_nodes.append(self)

    def disconnect_from(self: "Node", node: "Node") -> None:
        """Disconnect from another node.

        Args:
            node: Node to disconnect from.
        """
        if self not in node.in_nodes:
            return
        i = node.in_nodes.index(self)
        node.weights.pop(i)
        self.out_nodes.remove(node)
        node.in_nodes.remove(self)

    def compute(self: "Node") -> None:
        """Runs the node's computation using its `in_nodes` outputs."""
        x = [node.output for node in self.in_nodes]
        x = np.dot(x, self.weights) + self.bias
        x = np.clip(x, 0, 2**31 - 1)
        self.cached_output = float(x)

    def update(self: "Node") -> None:
        """Updates the node's output."""
        self.output = self.cached_output
        self.cached_output = None
