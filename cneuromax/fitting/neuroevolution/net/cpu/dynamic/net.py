""":class:`Net` & :class:`NetConfig`."""

import secrets
from dataclasses import dataclass
from typing import Annotated as An

import numpy as np

from cneuromax.utils.beartype import ge, one_of

from .node import Node, NodeList


@dataclass
class NetConfig:
    """Holds :class:`Net` config values.

    Args:
        num_inputs: Self-explanatory.
        num_outputs: Self-explanatory.
        node_selection_scheme: The scheme used to select nodes when\
            growing the network. If ``"layered"``, nodes grow & connect\
            to nodes in close layers preferentially. If ``"random"``,\
            nodes grow & connect to nodes in random positions.
    """

    num_inputs: An[int, ge(1)]
    num_outputs: An[int, ge(1)]
    node_selection_scheme: An[str, one_of("random", "layered")]


class Net:
    """Recurrent Neural Network.

    Args:
        config: See :class:`NetConfig`.

    Attributes:
        config: See :paramref:`config`.
        nodes: See :class:`NodeList`.
        total_nb_nodes_grown: The number of nodes grown in the\
            network since its initialization.
    """

    def __init__(self: "Net", config: NetConfig) -> None:
        self.config = config
        self.nodes = NodeList()
        self.total_nb_nodes_grown: int = 0

    def initialize_architecture(self: "Net") -> None:
        """Creates the initial architecture of the network.

        Grows the input and output nodes, with no connections between
        them and output node biases set to 0.
        """
        self.nodes.layered = [[], []]
        for _ in range(self.config.num_inputs):
            self.grow_node("input")
        for _ in range(self.config.num_outputs):
            self.grow_node("output")

    def grow_node(
        self: "Net",
        role: An[str, one_of("input", "hidden", "output")] = "hidden",
    ) -> None:
        """Grows a node in the network.

        Args:
            role: The role of the node to grow.
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
        elif role == "output":
            self.nodes.output.append(new_node)
            self.nodes.layered[-1].append(new_node)
        else:  # role == 'hidden'
            # Set of receiving nodes.
            potential_in_nodes = list(dict.fromkeys(self.nodes.receiving))
            # in_node_1
            in_node_1 = secrets.choice(potential_in_nodes)
            self.grow_connection(in_node_1, new_node)
            in_node_1_layer = find_node_layer_index(
                in_node_1,
                self.nodes.layered,
            )
            potential_in_nodes.remove(in_node_1)
            # in_node_2
            if self.config.node_selection_scheme == "random":
                in_node_2 = secrets.choice(potential_in_nodes)
            else:  # self.config.node_selection_scheme == 'layered'
                error_msg = "Layered node selection not implemented."
                raise NotImplementedError(error_msg)
            self.grow_connection(in_node_2, new_node)
            # out_node
            if self.config.node_selection_scheme == "random":
                out_node = secrets.choice(
                    self.nodes.hidden + self.nodes.output,
                )
            else:  # self.config.node_selection_scheme == 'layered'
                error_msg = "Layered node selection not implemented."
            self.grow_connection(new_node, out_node)
            out_node_layer = find_node_layer_index(
                out_node,
                self.nodes.layered,
            )
            layer_difference = out_node_layer - in_node_1_layer
            self.nodes.all.append(new_node)
            self.nodes.hidden.append(new_node)

            if abs(layer_difference) > 1:
                self.nodes.layered[
                    in_node_1_layer + int(np.sign(layer_difference))
                ].append(new_node)

            else:
                if layer_difference == 1:
                    latest_layer = out_node_layer
                else:  # layer_difference == -1 or layer_difference == 0:
                    latest_layer = in_node_1_layer

                self.nodes.layered.insert(latest_layer, [])
                self.nodes.layered[latest_layer].append(new_node)

    def grow_connection(
        self: "Net",
        in_node: Node,
        out_node: Node,
    ) -> None:
        """Grows a connection between two nodes.

        Args:
            in_node: Self-explanatory.
            out_node: Self-explanatory.
        """
        # Add a node-wise connection.
        in_node.connect_to(out_node)
        # Add each node once* to the receiving and emitting lists.
        # *Nodes can appear multiple times in these lists.
        self.nodes.receiving.append(out_node)
        self.nodes.emitting.append(in_node)

    def prune_node(self: "Net", node: Node | None = None) -> None:
        # If a node is not specified, sample one.
        if not node:
            if len(self.nodes.hidden) == 0:
                return
            node = secrets.choice(self.nodes.hidden)
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
        self: "Net",
        in_node: Node,
        out_node: Node,
        current_node_in_focus: Node | None = None,
    ) -> None:
        """Prunes a connection between two nodes.

        Args:
            in_node: Self-explanatory.
            out_node: Self-explanatory.
            current_node_in_focus: Self-explanatory.
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

    def reset(self: "Net") -> None:
        """Resets all node outputs to 0."""
        for node in self.nodes.all:
            node.output = np.ndarray([0])

    def __call__(self: "Net", x: list[float]) -> list[float]:
        """Forward pass through the network.

        Used for CPU-based computation.

        Args:
            x: Input values.

        Returns:
            Output values.
        """
        for x_i, input_node in zip(x, self.nodes.input, strict=True):
            input_node.output = x_i
        for layer in range(1, len(self.nodes.layered)):
            for node in self.nodes.layered[layer]:
                node.compute()
            for node in self.nodes.layered[layer]:
                node.update()
        return [node.output for node in self.nodes.output]


def find_node_layer_index(node: Node, layered_nodes: list[list[Node]]) -> int:
    """Finds the layer index of a node in the layered list of nodes.

    Args:
        node: The node to find the layer index of.
        layered_nodes: The list of layers of nodes.
    """
    for layer_index, layer in enumerate(layered_nodes):
        if node in layer:
            return layer_index
    error_msg = "Node not found in layered nodes."
    raise ValueError(error_msg)
