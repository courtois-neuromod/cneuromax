""":class:`DynamicNet` & :class:`DynamicNetConfig`."""

import logging
import random
from dataclasses import dataclass
from typing import Annotated as An

import numpy as np
from ordered_set import OrderedSet

from cneuromax.utils.beartype import ge, one_of

from .node import Node, NodeList


@dataclass
class DynamicNetConfig:
    """Holds :class:`DynamicNet` config values.

    Args:
        num_inputs: Self-explanatory.
        num_outputs: Self-explanatory.
        tensorized: ...
    """

    num_inputs: An[int, ge(1)]
    num_outputs: An[int, ge(1)]


class DynamicNet:
    """Neural network with a dynamically complexifying architecture.

    The flow of computation in this network is more brain-like than
    standard multi-layered neural networks in the sense that all neurons
    run in parallel, i.e. there is no concept of layers. For any task,
    this network initially consists of input and output nodes devoid of
    connections. Hidden nodes are grown/pruned through two mutation
    functions: :meth:`grow_node` and :meth:`prune_node`. Both functions
    get called a given number of times determined by the mutable
    :attr:`num_grow_mutations` and :attr:`num_prune_mutations`
    attributes. Weights (no biases in this network as all node outputs
    are standardized) are set upon node/connection creation and are left
    fixed from that point on. New connections are grown between nodes
    that are "more or less" distant from each other (the distance
    corresponds to the number of connections between two nodes,
    regardless of connection direction). The "more or less" component is
    controlled by the mutable :attr:`connectivity_temperature`
    attribute. Finally, the number of passes through the network per
    input is controlled by the mutable
    :attr:`num_network_passes_per_input` attribute.

    Args:
        config: See :class:`DynamicNetConfig`.

    Attributes:
        config: See :paramref:`config`.
        nodes: See :class:`.NodeList`.
        weights: A list comprised of the list of weights for each node.
        outputs: A list comprised of the latest output values for each\
            node.
        num_grow_mutations: A mutable value that controls the\
            number of chained :meth:`grow_node` mutations to perform.
        num_prune_mutations: A mutable value that controls the\
            number of chained :meth:`prune_node` mutations to perform.
        connectivity_temperature: A mutable value between 0 and 1\
            that controls the probability of selecting a nearby node\
            to connect to/from. A value of 1 means that all nodes are\
            equally likely to be selected, while a value of 0 means\
            that only nodes with a distance of 1 from the original\
            node are considered.
    """

    def __init__(self: "DynamicNet", config: DynamicNetConfig) -> None:
        self.config = config
        self.nodes = NodeList()
        self.weights: list[list[float]] = []
        self.outputs: list[float] = []
        self.initialize_architecture()
        self.num_grow_mutations: float = 1.0
        self.num_prune_mutations: float = 0.5
        self.num_network_passes_per_input: float = 1.0
        self.connectivity_temperature: float = 0.5

    def initialize_architecture(self: "DynamicNet") -> None:  # noqa: D102
        for _ in range(self.config.num_inputs):
            self.grow_node(role="input")
        for _ in range(self.config.num_outputs):
            self.grow_node(role="output")

    def mutate(self: "DynamicNet") -> None:  # noqa: D102
        self.mutate_parameters()
        # :meth:`prune_node`
        if self.num_prune_mutations < 1:
            rand_num = np.random.uniform()
            num_prune_mutations = int(rand_num < self.num_prune_mutations)
        else:
            num_prune_mutations = int(self.num_prune_mutations)
        for _ in range(num_prune_mutations):
            self.prune_node()
        # :meth:`prune_node`
        if self.num_grow_mutations < 1:
            rand_num = np.random.uniform()
            num_grow_mutations = int(rand_num < self.num_grow_mutations)
        else:
            num_grow_mutations = int(self.num_grow_mutations)
        node_to_connect_with = None
        for _ in range(num_grow_mutations):
            node_to_connect_with = self.grow_node(node_to_connect_with)

    def mutate_parameters(self: "DynamicNet") -> None:  # noqa: D102
        # :attr:`num_grow_mutations`
        rand_num = np.random.randint(100)
        if rand_num == 0:
            self.num_grow_mutations /= 2
        if rand_num == 1:
            self.num_grow_mutations *= 2
        # :attr:`num_prune_mutations`
        rand_num = np.random.randint(100)
        if rand_num == 0:
            self.num_prune_mutations /= 2
        if rand_num == 1:
            self.num_prune_mutations *= 2
        # :attr:`num_network_passes_per_input`
        rand_num = np.random.randint(100)
        if rand_num == 0 and self.num_network_passes_per_input != 1:
            self.num_network_passes_per_input /= 2
        if rand_num == 1:
            self.num_network_passes_per_input *= 2
        # :attr:`connectivity_temperature`
        rand_num = np.random.randint(100)
        if rand_num == 0 and self.connectivity_temperature != 1:
            self.connectivity_temperature += 0.1
        if rand_num == 1 and self.connectivity_temperature != 0:
            self.connectivity_temperature -= 0.1

    def grow_node(  # noqa: D102
        self: "DynamicNet",
        node_to_connect_with: Node | None = None,
        role: An[str, one_of("input", "hidden", "output")] = "hidden",
    ) -> Node:
        new_node = Node(role, len(self.nodes.all))
        logging.debug(f"New {role} node: {new_node} w/ index {new_node.index}")
        self.nodes.all.append(new_node)
        if role == "input":
            self.nodes.input.append(new_node)
            self.nodes.receiving.append(new_node)
        elif role == "output":
            self.nodes.output.append(new_node)
        else:  # role == 'hidden'
            self.nodes.hidden.append(new_node)
            in_node_1 = out_node = None
            if node_to_connect_with:
                from_to = random.choice(["from", "to"])  # noqa: S311
                logging.debug(f"`node_to_connect_with`: {from_to}")
                in_node_1 = node_to_connect_with if from_to == "from" else None
                out_node = node_to_connect_with if from_to == "to" else None
            receiving_nodes_set = OrderedSet(self.nodes.receiving)
            if not in_node_1:
                in_node_1 = random.choice(receiving_nodes_set)  # noqa: S311
            self.grow_connection(in_node_1, new_node)
            logging.debug(
                f"Connected from {in_node_1} w/ index {in_node_1.index}",
            )
            in_node_2 = in_node_1.find_nearby_node(
                nodes_considered=receiving_nodes_set,
                connectivity_temperature=self.connectivity_temperature,
                purpose="connect with",
            )
            self.grow_connection(in_node_2, new_node)
            logging.debug(
                f"Connected from {in_node_2} w/ index {in_node_2.index}",
            )
            if not out_node:
                out_node = new_node.find_nearby_node(
                    nodes_considered=OrderedSet(
                        self.nodes.hidden + self.nodes.output,
                    ),
                    connectivity_temperature=self.connectivity_temperature,
                    purpose="connect to",
                )
            self.grow_connection(new_node, out_node)
            logging.debug(f"Connected to {out_node} w/ index {out_node.index}")
        if role in ["hidden", "output"]:
            self.weights.append(new_node.weights)
            self.outputs.append(0)
        return new_node

    def grow_connection(  # noqa: D102
        self: "DynamicNet",
        in_node: Node,
        out_node: Node,
    ) -> None:
        in_node.connect_to(out_node)
        self.nodes.receiving.append(out_node)
        self.nodes.emitting.append(in_node)

    def prune_node(  # noqa: D102
        self: "DynamicNet",
        node_to_prune: Node | None = None,
    ) -> None:
        node = node_to_prune
        if not node:
            if len(self.nodes.hidden) == 0:
                return
            node = random.choice(self.nodes.hidden)  # noqa: S311
        if node in self.nodes.being_pruned:
            return
        self.nodes.being_pruned.append(node)
        for out_node in node.out_nodes.copy():
            self.prune_connection(node, out_node, node)
        for in_node in node.in_nodes.copy():
            self.prune_connection(in_node, node, node)
        for node_list in self.nodes:
            while node in node_list:
                node_list.remove(node)  # type: ignore[arg-type]

    def prune_connection(  # noqa: D102
        self: "DynamicNet",
        in_node: Node,
        out_node: Node,
        current_node_in_focus: Node,
    ) -> None:
        if in_node not in out_node.in_nodes:
            return
        in_node.disconnect_from(out_node)
        self.nodes.receiving.remove(out_node)
        self.nodes.emitting.remove(in_node)
        if (
            in_node is not current_node_in_focus
            and in_node not in self.nodes.emitting
            and in_node in self.nodes.hidden
        ):
            self.prune_node(in_node)
        if (
            out_node is not current_node_in_focus
            and out_node not in self.nodes.receiving
            and out_node in self.nodes.hidden
        ):
            self.prune_node(out_node)
