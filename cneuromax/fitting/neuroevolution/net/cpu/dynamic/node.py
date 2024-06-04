""":class:`NodeList`, :class:`Node`, and :class:`ComputingNode`."""

import random
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Annotated as An
from typing import Any

import numpy as np
from ordered_set import OrderedSet

from cneuromax.utils.beartype import one_of


@dataclass
class NodeList:
    """Holds :class:`Node` instances.

    Args:
        all: Contains all :class:`Node` instances currently in the\
            network.
        indices: Contains the indices of all :class:`Node` instances\
            currently in the network.
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
    """

    all: list["Node"] = field(default_factory=list)
    input: list["Node"] = field(default_factory=list)
    hidden: list["Node"] = field(default_factory=list)
    output: list["Node"] = field(default_factory=list)
    receiving: list["Node"] = field(default_factory=list)
    emitting: list["Node"] = field(default_factory=list)
    being_pruned: list["Node"] = field(default_factory=list)

    def __iter__(
        self: "NodeList",
    ) -> Iterator[list["Node"] | list[list["Node"]]]:
        """Iterator over all lists of nodes."""
        return iter(
            [
                self.all,
                self.input,
                self.hidden,
                self.output,
                self.receiving,
                self.emitting,
                self.being_pruned,
            ],
        )


class Node:
    """Node (Neuron) for use in :class:`.DynamicNet`.

    Args:
        role: Node function in :class:`.DynamicNet`
        uid: Self-explanatory

    Attributes:
        role: See :paramref:`role`.
        uid: See :paramref:`uid`.
        in_nodes: List of nodes that send information to this node.
        out_nodes: List of nodes that receive information from this\
            node.
        weights: Weights to apply to received values emitted by\
            :attr:`in_nodes`. Is of length 3, as a node can have at\
            most 3 incoming connections.
        num_in_nodes: Number of incoming connections.
        output: Value emitted from the node.

        TODO: move :attr:`output` to :class:`ComputingNode` and\
        figure out how to type-hint ``self.in_nodes`` as ``list[self]``.
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
        if self.role != "input":
            self.weights: list[float] = [0, 0, 0]
            self.num_in_nodes = 0
        self.output: float = 0

    def __repr__(self: "Node") -> str:  # noqa: D105
        node_inputs: tuple[Any, ...] = tuple(
            (
                "x"
                if self.role == "input"
                else (node.uid for node in self.in_nodes)
            ),
        )
        node_outputs: tuple[Any, ...] = tuple(
            node.uid for node in self.out_nodes
        )
        if self.role == "output":
            node_outputs = ("y", *node_outputs)
        return (
            str(node_inputs) + "->" + str(self.uid) + "->" + str(node_outputs)
        )

    def find_nearby_node(
        self: "Node",
        nodes_considered: OrderedSet["Node"],
        connectivity_temperature: float,
        purpose: An[str, one_of("connect with", "connect to")],
    ) -> "Node":
        """Finds a nearby node to connect to/from.

        With ``i`` starting at ``1``, return a random node within
        distance ``i`` with probability ``1 -``
        :paramref:`connectivity_temperature`, else increase distance by
        1 until a node is found. (The search range is increased to all
        "receiving" nodes in the network if all connected nodes have
        been considered.)
        """
        found = False
        # Start with nodes within distance of 1 from the original node.
        nodes_at_distance_i = OrderedSet(self.in_nodes + self.out_nodes)
        for node in nodes_considered.copy():
            if node is self or (
                purpose == "connect to"
                and node.role != "input"
                and node.num_in_nodes == 3  # noqa: PLR2004
            ):
                nodes_considered.remove(node)  # type: ignore[arg-type]
        while not found:
            nodes_considered_at_distance_i = (
                nodes_at_distance_i & nodes_considered
            )
            if (
                np.random.uniform() < 1 - connectivity_temperature
                and nodes_considered_at_distance_i
            ):
                nearby_node = random.choice(  # noqa: S311
                    nodes_considered_at_distance_i,
                )
                found = True
            else:
                # Increase the distance by 1.
                nodes_at_distance_i_plus_1 = nodes_at_distance_i.copy()
                for node in nodes_at_distance_i:
                    nodes_at_distance_i_plus_1 |= OrderedSet(
                        node.in_nodes + node.out_nodes,
                    )
                # If all connected nodes have been considered, increase
                # the search range to all "receiving" nodes in the
                # network & set connectivity_temperature to 1 which will
                # force the selection of a node during the next
                # iteration.
                if nodes_at_distance_i == nodes_at_distance_i_plus_1:
                    nodes_at_distance_i = OrderedSet(nodes_considered)
                    connectivity_temperature = 0
                else:
                    nodes_at_distance_i = nodes_at_distance_i_plus_1
        return nearby_node

    def connect_to(self: "Node", node: "Node") -> None:  # noqa: D102
        new_weight: float = float(np.random.randn())
        node.weights[node.num_in_nodes] = new_weight
        node.num_in_nodes += 1
        self.out_nodes.append(node)
        node.in_nodes.append(self)

    def disconnect_from(self: "Node", node: "Node") -> None:  # noqa: D102
        i = node.in_nodes.index(self)
        if i == 0:
            node.weights[0] = node.weights[1]
        if i in (0, 1):
            node.weights[1] = node.weights[2]
        node.weights[2] = 0
        node.num_in_nodes -= 1
        self.out_nodes.remove(node)
        node.in_nodes.remove(self)


class ComputingNode(Node):
    """:class:`Node` that handles its own computation.

    Attributes:
        mean: Mean of the node's output.
        v: Variance of the node's output.
        std: Standard deviation of the node's output.
        n: Number of times the node's output has been updated.
    """

    def __init__(
        self: "ComputingNode",
        *args: Any,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.mean: float = 0
        self.v: float = 0
        self.std: float = 0
        self.n: int = 0

    def update_standardization_attributes(
        self: "ComputingNode",
        raw_output: float,
    ) -> None:
        """Updates standardization attributes given :attr:`raw_output`.

        Args:
            raw_output: The node's output before normalization.
        """
        self.n += 1
        temp_m = self.mean + (raw_output - self.mean) / self.n
        temp_v = self.v + (raw_output - self.mean) * (raw_output - temp_m)
        self.v = temp_v
        self.mean = temp_m
        self.std = np.sqrt(self.v / self.n)

    def compute_and_cache_output(self: "ComputingNode") -> None:
        """Computes the node's output from its :attr:`in_nodes` outputs.

        Gathers input values (:attr:`in_nodes` output values) and
        runs a dot product with the node's :attr:`weights`. The output
        is then standardized and cached as all node outputs are updated
        simultaneously at the end of the network's pass.
        """
        x = [node.output for node in self.in_nodes]
        x = float(np.dot(x, self.weights[: len(x)]))
        self.update_standardization_attributes(x)
        x = (x - self.mean) / (self.std + (self.std == 0))
        self.cached_output = x

    def update_output(self: "ComputingNode") -> None:
        """Sets :attr:`output` to :attr:`cached_output`."""
        self.output = self.cached_output
