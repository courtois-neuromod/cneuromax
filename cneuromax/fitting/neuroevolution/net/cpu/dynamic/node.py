""":class:`NodeList` & :class:`Node`."""

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Annotated as An
from typing import Any

import numpy as np

from cneuromax.utils.beartype import one_of


@dataclass
class NodeList:
    """Holds :class:`Node` instances.

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

    def __post_init__(self: "NodeList") -> None:
        """Initializes :paramref:`layered`."""
        self.layered = [[], []]

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
    """Node for use in :class:`.DynamicNet`.

    Args:
        role: Node function in :class:`.DynamicNet`
        uid: Self-explanatory

    Attributes:
        role: See :paramref:`role`.
        uid: See :paramref:`uid`.
        in_nodes: List of nodes that send information to this node.
        out_nodes: List of nodes that receive information from this\
            node.
        output: Value emitted from the node.
        weights: Weights to apply to received values emitted by\
            :attr:`in_nodes`. Is of length 3, as a node can have at\
            most 3 incoming connections.
        num_in_nodes: Number of incoming connections.
        bias: Bias to apply to the node's output. Only present in\
            hidden nodes.
        cached_output: See :meth:`compute_and_cache_output`.
    """

    def __init__(
        self: "Node",
        role: An[str, one_of("input", "hidden", "output")],
        uid: int,
    ) -> None:
        self.role = role
        self.uid = uid
        self.in_nodes: list[Node] = []
        self.out_nodes: list[Node] = []
        self.output: float = 0
        if self.role != "input":
            self.weights: list[float] = [0, 0, 0]
            self.num_in_nodes = 0
            if self.role == "hidden":
                self.bias = np.random.randn()

    def __repr__(self: "Node") -> str:  # noqa: D105
        node_inputs: tuple[Any, ...] = tuple(
            "x" if self.role == "input" else node.uid for node in self.in_nodes
        )
        node_outputs: tuple[Any, ...] = tuple(
            node.uid for node in self.out_nodes
        )
        if self.role == "output":
            node_outputs = ("y", *node_outputs)
        return (
            str(node_inputs)
            + "->"
            + str(self.uid)
            + "->"
            + str(("y", *node_outputs))
        )

    def connect_to(self: "Node", node: "Node") -> None:  # noqa: D102
        new_weight: float = float(np.random.randn())
        node.weights[self.num_in_nodes] = new_weight
        self.num_in_nodes += 1
        self.out_nodes.append(node)
        node.in_nodes.append(self)

    def disconnect_from(self: "Node", node: "Node") -> None:  # noqa: D102
        i = node.in_nodes.index(self)
        if i == 0:
            node.weights[0] = node.weights[1]
        if i in (0, 1):
            node.weights[1] = node.weights[2]
        node.weights[2] = 0
        self.out_nodes.remove(node)
        node.in_nodes.remove(self)

    def compute_and_cache_output(self: "Node") -> None:
        """Computes the node's output from its :attr:`in_nodes`.

        Non-linear transformation of :attr:`in_nodes` :attr:`output`
        values by :attr:`weights` ``+`` :attr:`bias`, followed by a
        ReLU activation function. The output is cached as all node
        outputs are updated simultaneously at the end of the
        network's "forward" pass.
        """
        x = [node.output for node in self.in_nodes]
        x = np.dot(x, self.weights)
        if self.role == "hidden":
            x += self.bias
        x = np.clip(x, 0, 2**31 - 1)
        self.cached_output = float(x)

    def update_output(self: "Node") -> None:  # noqa: D102
        self.output = self.cached_output
