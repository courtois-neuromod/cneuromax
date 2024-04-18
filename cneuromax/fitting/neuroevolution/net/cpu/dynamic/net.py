""":class:`Net`, :class:`NetConfig` & :func:`find_node_layer_index`."""

import random
from dataclasses import dataclass
from typing import Annotated as An

import numpy as np

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

    This network increases/decreases in complexity by growing/pruning
    nodes and connections through two mutation functions:
    :meth:`grow_node` and :meth:`prune_node`. Weights & biases are
    set upon node/connection creation but are not updated during
    evolution.

    Args:
        config: See :class:`DynamicNetConfig`.

    Attributes:
        config: See :paramref:`config`.
        nodes: See :class:`.NodeList`.
        total_nb_nodes_grown: The number of nodes grown in the\
            network since its instantiation.
        weights: Node connection weights.
        biases: Node biases.
    """

    def __init__(self: "DynamicNet", config: DynamicNetConfig) -> None:
        self.config = config
        self.nodes = NodeList()
        self.total_nb_nodes_grown = 0
        self.weights: list[list[float]] = [[]]
        self.biases: list[float] = []
        self.initialize_architecture()

    def initialize_architecture(self: "DynamicNet") -> None:
        """Creates the initial architecture of the network.

        Grows the input and output nodes. No connection is grown and
        output node biases are set to 0.
        """
        for _ in range(self.config.num_inputs):
            self.grow_node("input")
        for _ in range(self.config.num_outputs):
            self.grow_node("output")

    def grow_node(
        self: "DynamicNet",
        role: An[str, one_of("input", "hidden", "output")] = "hidden",
    ) -> None:
        """Adds a new node to the network.

        If the :paramref:`role` ``== "hidden"`` (when this method is
        called during mutation), a new node is grown with two incoming
        connections from randomly selected nodes and one outgoing
        connection to another randomly selected node.

        TODO: Replace random selection with layered node selection.

        Args:
            role: Self-explanatory.
        """
        # Creates the node & increments the total number of nodes grown.
        new_node = Node(role, self.total_nb_nodes_grown)
        self.total_nb_nodes_grown += 1
        # Adds the node to the appropriate lists.
        self.nodes.all.append(new_node)
        if role == "input":
            self.nodes.input.append(new_node)
            self.nodes.receiving.append(new_node)
            self.nodes.layered[0].append(new_node)
            return
        if role == "output":
            self.nodes.output.append(new_node)
            self.nodes.layered[-1].append(new_node)
        else:  # role == 'hidden'
            potential_in_nodes = list(dict.fromkeys(self.nodes.receiving))
            in_node_1 = random.choice(potential_in_nodes)  # noqa: S311
            self.grow_connection(in_node_1, new_node)
            in_node_1_layer = find_node_layer_index(
                in_node_1,
                self.nodes.layered,
            )
            potential_in_nodes.remove(in_node_1)
            in_node_2 = random.choice(potential_in_nodes)  # noqa: S311
            self.grow_connection(in_node_2, new_node)
            out_node = random.choice(  # noqa: S311
                self.nodes.hidden + self.nodes.output,
            )
            self.grow_connection(new_node, out_node)
            out_node_layer = find_node_layer_index(
                out_node,
                self.nodes.layered,
            )
            self.nodes.all.append(new_node)
            self.nodes.hidden.append(new_node)
            layer_difference = out_node_layer - in_node_1_layer
            if abs(layer_difference) > 1:
                layer = in_node_1_layer + int(np.sign(layer_difference))
            else:
                if layer_difference == 1:
                    layer = out_node_layer
                else:  # layer_difference == -1 or layer_difference == 0:
                    layer = in_node_1_layer
                self.nodes.layered.insert(layer, [])
            self.nodes.layered[layer].append(new_node)
        self.weights.append(new_node.weights)
        self.biases.append(new_node.bias)

    def grow_connection(
        self: "DynamicNet",
        in_node: Node,
        out_node: Node,
    ) -> None:
        """Grows a connection between two nodes.

        Args:
            in_node: Self-explanatory.
            out_node: Self-explanatory.
        """
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
            node = random.choice(self.nodes.hidden)
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
            if node_list == self.nodes.layered:
                node_layer_index = find_node_layer_index(
                    node,
                    self.nodes.layered,
                )
                self.nodes.layered[node_layer_index].remove(node)
                # Remove the layer if it is empty.
                if (
                    node_layer_index not in (0, len(self.nodes.layered) - 1)
                    and self.nodes.layered[node_layer_index] == []
                ):
                    self.nodes.layered.remove(
                        self.nodes.layered[node_layer_index],
                    )
            else:
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
        """Forward pass through the network.

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


def find_node_layer_index(node: Node, layered_nodes: list[list[Node]]) -> int:
    """Finds layer idx of :paramref:`node` in :paramref:`layered_nodes`.

    Args:
        node: Self-explanatory.
        layered_nodes: See :paramref:`~NodeList.layered_nodes`.

    Returns:
        The layer index of :paramref:`node` in\
            :paramref:`layered_nodes`.

    Raises:
        ValueError: If the node is not found in\
            :paramref:`layered_nodes`.
    """
    for layer_index, layer in enumerate(layered_nodes):
        if node in layer:
            return layer_index
    error_msg = "Node not found in layered nodes."
    raise ValueError(error_msg)
