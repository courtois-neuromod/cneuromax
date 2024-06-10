""":class:`NodeList`, :class:`Node`, and :class:`ComputingNode`."""

import logging
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
        index: Self-explanatory.

    Attributes:
        role: See :paramref:`role`.
        index: See :paramref:`uid`.
        in_nodes: List of nodes that send information to this node.
        out_nodes: List of nodes that receive information from this\
            node.
        weights: Weights to apply to received values emitted by\
            :attr:`in_nodes`. Is of length 3, as a node can have at\
            most 3 incoming connections.
        num_in_nodes: Number of incoming connections.
    """

    def __init__(
        self: "Node",
        role: An[str, one_of("input", "hidden", "output")],
        index: int,
    ) -> None:
        self.role = role
        self.index = index
        self.in_nodes: list[Node] = []
        self.out_nodes: list[Node] = []
        if self.role != "input":
            self.weights: list[float] = [0, 0, 0]
            self.num_in_nodes = 0

    def __repr__(self: "Node") -> str:  # noqa: D105
        node_inputs: tuple[Any, ...] = tuple(
            (
                "x"
                if self.role == "input"
                else (node.index for node in self.in_nodes)
            ),
        )
        node_outputs: tuple[Any, ...] = tuple(
            node.index for node in self.out_nodes
        )
        if self.role == "output":
            node_outputs = ("y", *node_outputs)
        return (
            str(node_inputs)
            + "->"
            + str(self.index)
            + "->"
            + str(node_outputs)
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
