""":class:`DynamicNet` & its config."""

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
        runs_on_gpu: Self-explanatory.
    """

    num_inputs: An[int, ge(1)]
    num_outputs: An[int, ge(1)]


class DynamicNet:
    """Neural network with a dynamically complexifying architecture.

    Computation in this network is more brain-like in the sense that
    all neurons constantly compute, there is no concept of layers. The
    network initially consists of input and output nodes devoid of
    connections. Hidden nodes are grown/pruned through two mutation
    functions: :meth:`grow_node` and :meth:`prune_node` that each get
    called a number of times determined by the mutable
    :attr:`num_grow_mutations` and :attr:`num_prune_mutations`. Weights
    (no biases in this network) are set upon node/connection creation
    and are left fixed from that point on. New connections are grown
    between nodes that are "more or less" distant from each other (the
    distance corresponds to the number of connections between two
    nodes, regardless of connection direction). The "more or less"
    component is controlled by the mutable
    :attr:`connectivity_temperature` attribute.

    Args:
        config: See :class:`DynamicNetConfig`.

    Attributes:
        config: See :paramref:`config`.
        nodes: See :class:`.NodeList`.
        total_nb_nodes_grown: The number of nodes grown in the\
            network since its instantiation.
        weights: Node connection weights.
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
        self.weights: list[list[float]] = [[]]
        self.outputs: list[float] = []
        self.initialize_architecture()
        # Mutable attributes.
        self.num_grow_mutations: float = 1.0
        self.num_prune_mutations: float = 0.5
        self.connectivity_temperature: float = 0.5

    def initialize_architecture(self: "DynamicNet") -> None:  # noqa: D102
        for _ in range(self.config.num_inputs):
            self.grow_node("input")
        for _ in range(self.config.num_outputs):
            self.grow_node("output")

    def mutate_parameters(self: "DynamicNet") -> None:
        """Perturbs the network's mutable attributes.

        Increase/decrease with a 1% chance.
        """
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
        if rand_num == 0 and self.connectivity_temperature != 1:
            self.connectivity_temperature += 0.1
        if rand_num == 1 and self.connectivity_temperature != 0:
            self.connectivity_temperature -= 0.1

    def mutate(
        self: "DynamicNet",
    ) -> None:
        """Mutates the network's architecture and parameters."""
        self.mutate_parameters()
        node_to_prune = None
        for _ in range(self.num_prune_mutations):
            node_to_prune = self.prune_node(node_to_prune)
        node_to_connect_with = None
        for _ in range(self.num_grow_mutations):
            node_to_connect_with = self.grow_node(node_to_connect_with)

    def grow_node(
        self: "DynamicNet",
        role: An[str, one_of("input", "hidden", "output")] = "hidden",
    ) -> None:
        """Adds a new node to the network.

        If the :paramref:`role` ``== "hidden"`` (when this method is
        called during mutation), a new node is grown with two incoming
        connections from randomly selected nodes and one outgoing
        connection to another randomly selected node.
        """
        new_node = Node(role, self.total_nb_nodes_grown)
        self.total_nb_nodes_grown += 1
        self.nodes.all.append(new_node)
        if role == "input":
            self.nodes.input.append(new_node)
            self.nodes.receiving.append(new_node)
            return
        if role == "output":
            self.nodes.output.append(new_node)
        else:  # role == 'hidden'
            receiving_nodes_set = OrderedSet(self.nodes.receiving)
            in_node_1 = random.choice(receiving_nodes_set)  # noqa: S311
            self.grow_connection(in_node_1, new_node)
            in_node_2 = in_node_1.find_nearby_node(
                receiving_nodes_set,
                self.connectivity_temperature,
            )
            self.grow_connection(in_node_2, new_node)
            out_node = new_node.find_nearby_node(
                OrderedSet(self.nodes.hidden + self.nodes.output),
                self.connectivity_temperature,
            )
            self.grow_connection(new_node, out_node)
            self.nodes.all.append(new_node)
            self.nodes.hidden.append(new_node)
        self.weights.append(new_node.weights)

    def grow_connection(  # noqa: D102
        self: "DynamicNet",
        in_node: Node,
        out_node: Node,
    ) -> None:
        in_node.connect_to(out_node)
        self.nodes.receiving.append(out_node)
        self.nodes.emitting.append(in_node)

    def prune_node(self: "DynamicNet", node: Node | None = None) -> None:
        """Prunes a node from the network.

        Args:
            node: The node to prune. If not specified, a random hidden\
                node is pruned instead.
        """
        # If a node is not specified, sample one.
        if not node:
            if len(self.nodes.hidden) == 0:
                return
            node = random.choice(self.nodes.hidden)  # noqa: S311
        # If the node is already being pruned, return to avoid infinite
        # recursion.
        if node in self.nodes.being_pruned:
            return
        self.nodes.being_pruned.append(node)
        # Remove all outcoming connections.
        for out_node in node.out_nodes.copy():
            self.prune_connection(node, out_node, node)
        # Remove all incoming connections.
        for in_node in node.in_nodes.copy():
            self.prune_connection(in_node, node, node)
        # Remove the node from all node lists.
        for node_list in self.nodes:
            while node in node_list:
                node_list.remove(node)  # type: ignore[arg-type]

    def prune_connection(
        self: "DynamicNet",
        in_node: Node,
        out_node: Node,
        current_node_in_focus: Node,
    ) -> None:
        """Prunes a connection between two nodes.

        Args:
            in_node: Self-explanatory.
            out_node: Self-explanatory.
            current_node_in_focus: Either :paramref:`in_node` or\
                :paramref:`out_node`.
        """
        # Already pruned, return to avoid infinite recursion.
        if in_node not in out_node.in_nodes:
            return
        # Remove the node-wise connection.
        in_node.disconnect_from(out_node)
        # Remove each node once* from the receiving and emitting lists.
        # *Nodes can appear multiple times in these lists.
        self.nodes.receiving.remove(out_node)
        self.nodes.emitting.remove(in_node)
        # Prune the nodes if they are cut-off from the information flow.
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
        for x_i, input_node in zip(x, self.nodes.input, strict=True):
            input_node.output = x_i
        for node in self.nodes.hidden + self.nodes.output:
            node.compute_and_cache_output()
        for node in self.nodes.hidden + self.nodes.output:
            node.update_output()
        return [node.output for node in self.nodes.output]
