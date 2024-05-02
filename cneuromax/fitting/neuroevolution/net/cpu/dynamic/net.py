""":class:`DynamicNet` & its config."""

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
    """

    num_inputs: An[int, ge(1)]
    num_outputs: An[int, ge(1)]


class DynamicNet:
    """Neural network with a dynamically complexifying architecture.

    The flow of computation in this network is more brain-like than
    standard multi-layered neural networks in the sense that all neurons
    compute in parallel, there is no concept of layers. For any task,
    this network initially consists of input and output nodes devoid of
    connections. Hidden nodes are grown/pruned through two mutation
    functions: :meth:`grow_node` and :meth:`prune_node` that each get
    called a number of times determined by the mutable
    :attr:`num_grow_mutations` and :attr:`num_prune_mutations`. Weights
    (no biases in this network as all node outputs are standardized) are
    set upon node/connection creation and are left fixed from that point
    on. New connections are grown between nodes that are "more or less"
    distant from each other (the distance corresponds to the number of
    connections between two nodes, regardless of connection direction).
    The "more or less" component is controlled by the mutable
    :attr:`connectivity_temperature` attribute. Finally, the number of
    passes through the network per input is controlled by the
    mutable :attr:`num_network_passes_per_input` attribute.

    Args:
        config: See :class:`DynamicNetConfig`.

    Attributes:
        config: See :paramref:`config`.
        nodes: See :class:`.NodeList`.
        total_nb_nodes_grown: The number of nodes grown in the\
            network since its instantiation.
        weights: A 2D list of weights. Each inner list corresponds\
            to the weights
        outputs: Node outputs.
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
        self.total_nb_nodes_grown = 0
        # self.weights: list[list[float]] = [[]]
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

    def mutate_parameters(self: "DynamicNet") -> None:  # noqa: D102
        rand_num = np.random.randint(100)
        if rand_num == 0:
            self.num_grow_mutations /= 2
        if rand_num == 1:
            self.num_grow_mutations *= 2
        rand_num = np.random.randint(100)
        if rand_num == 0:
            self.num_prune_mutations /= 2
        if rand_num == 1:
            self.num_prune_mutations *= 2
        rand_num = np.random.randint(100)
        if rand_num == 0 and self.num_network_passes_per_input != 1:
            self.num_network_passes_per_input /= 2
        if rand_num == 1:
            self.num_network_passes_per_input *= 2
        rand_num = np.random.randint(100)
        if rand_num == 0 and self.connectivity_temperature != 1:
            self.connectivity_temperature += 0.1
        if rand_num == 1 and self.connectivity_temperature != 0:
            self.connectivity_temperature -= 0.1

    def mutate(self: "DynamicNet") -> None:  # noqa: D102
        self.mutate_parameters()
        if self.num_prune_mutations < 1:
            rand_num = np.random.uniform()
            num_prune_mutations = int(rand_num < self.num_prune_mutations)
        else:
            num_prune_mutations = int(self.num_prune_mutations)
        for _ in range(num_prune_mutations):
            self.prune_node()
        if self.num_grow_mutations < 1:
            rand_num = np.random.uniform()
            num_grow_mutations = int(rand_num < self.num_grow_mutations)
        else:
            num_grow_mutations = int(self.num_grow_mutations)
        node_to_connect_with = None
        for _ in range(num_grow_mutations):
            node_to_connect_with = self.grow_node(node_to_connect_with)

    def grow_node(  # noqa: D102
        self: "DynamicNet",
        node_to_connect_with: Node | None = None,
        role: An[str, one_of("input", "hidden", "output")] = "hidden",
    ) -> Node:
        new_node = Node(role, self.total_nb_nodes_grown)
        self.total_nb_nodes_grown += 1
        self.nodes.all.append(new_node)
        if role == "input":
            self.nodes.input.append(new_node)
            self.nodes.receiving.append(new_node)
        elif role == "output":
            self.nodes.output.append(new_node)
        else:  # role == 'hidden'
            logging.debug("1")
            in_node_1 = out_node = None
            if node_to_connect_with:
                from_to = random.choice(["from", "to"])  # noqa: S311
                in_node_1 = node_to_connect_with if from_to == "from" else None
                out_node = node_to_connect_with if from_to == "to" else None
            receiving_nodes_set = OrderedSet(self.nodes.receiving)
            if not in_node_1:
                in_node_1 = random.choice(receiving_nodes_set)  # noqa: S311
            logging.debug("2")
            self.grow_connection(in_node_1, new_node)
            logging.debug("2a")
            in_node_2 = in_node_1.find_nearby_node(
                receiving_nodes_set,
                self.connectivity_temperature,
            )
            logging.debug("3")
            self.grow_connection(in_node_2, new_node)
            logging.debug("4")
            if not out_node:
                out_node = new_node.find_nearby_node(
                    OrderedSet(self.nodes.hidden + self.nodes.output),
                    self.connectivity_temperature,
                )
            logging.debug("5")
            self.grow_connection(new_node, out_node)
            logging.debug("6")
            self.nodes.all.append(new_node)
            self.nodes.hidden.append(new_node)
        # self.weights.append(new_node.weights)
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

    def reset(self: "DynamicNet") -> None:
        """Resets all node outputs to 0."""
        for node in self.nodes.all:
            node.output = 0

    def __call__(self: "DynamicNet", x: list[float]) -> list[float]:
        """Runs one pass through the network.

        Used for CPU-based computation.

        Args:
            x: Input values.

        Returns:
            Output values.
        """
        for _ in range(int(self.num_network_passes_per_input)):
            for x_i, input_node in zip(x, self.nodes.input, strict=True):
                input_node.output = x_i
            for node in self.nodes.hidden + self.nodes.output:
                node.compute_and_cache_output()
            for node in self.nodes.hidden + self.nodes.output:
                node.update_output()
        return [node.output for node in self.nodes.output]
