""":class:`Net` & :class:`NetConfig`."""

import random
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Annotated as An
from typing import Any

import numpy as np
from jaxtyping import Float
from numpy.typing import NDArray

from cneuromax.utils.beartype import ge

from .node import Node, NodeList


@dataclass
class NetConfig:
    """Holds :class:`Net` config values.

    Args:
        num_inputs:"""

    num_inputs: An[int, ge(1)]
    num_outputs: An[int, ge(1)]
    is_vectorized: bool = False


class Net:
    """Recurrent Neural Network.

    Attributes:
        config: The network's configuration.

    """

    def __init__(self: "Net", config: NetConfig) -> None:
        """Constructor.

        Args:
            config: The network's configuration.
        """
        self.config = config

        self.nodes = NodeList()

        self.total_nb_nodes_grown: int = 0

        self.architectural_operations: list[Callable[[Any], Any]] = [
            self.grow_node,
            self.prune_node,
            self.grow_connection,
            self.prune_connection,
        ]

    def initialize_architecture(self: "Net") -> None:
        """ """
        for _ in range(self.config.num_inputs):
            self.grow_node("input")

        for _ in range(self.config.num_outputs):
            self.grow_node("output")

    def grow_node(self: "Net", type="hidden") -> Node | None:
        if type == "input":
            new_input_node = Node("input", self.total_nb_nodes_grown)
            self.total_nb_nodes_grown += 1

            self.nodes.all.append(new_input_node)
            self.nodes.input.append(new_input_node)
            self.nodes.receiving.append(new_input_node)

            if len(self.nodes.layered) == 0:
                self.nodes.layered.append([])

            self.nodes.layered[0].append(new_input_node)

            return new_input_node

        elif type == "output":
            new_output_node = Node("output", self.total_nb_nodes_grown)
            self.total_nb_nodes_grown += 1

            self.nodes.all.append(new_output_node)
            self.nodes.output.append(new_output_node)

            while len(self.nodes.layered) < 2:
                self.nodes.layered.append([])

            self.nodes.layered[-1].append(new_output_node)

            return new_output_node

        else:  # type == 'hidden'
            #
            potential_in_nodes = list(dict.fromkeys(self.nodes.receiving))
            in_node_1 = np.random.choice(potential_in_nodes)
            potential_in_nodes.remove(in_node_1)
            if len(potential_in_nodes) != 0:
                in_node_2 = np.random.choice(potential_in_nodes)
            out_node = np.random.choice(self.nodes.hidden + self.nodes.output)
            new_hidden_node = Node("hidden", self.total_nb_nodes_grown)
            self.total_nb_nodes_grown += 1
            self.grow_connection(in_node_1, new_hidden_node)
            if len(potential_in_nodes) != 0:
                self.grow_connection(in_node_2, new_hidden_node)
            self.grow_connection(new_hidden_node, out_node)
            in_node_1_layer = find_sublist_index(in_node_1, self.nodes.layered)
            out_node_layer = find_sublist_index(out_node, self.nodes.layered)
            layer_difference = out_node_layer - in_node_1_layer
            self.nodes.all.append(new_hidden_node)
            self.nodes.hidden.append(new_hidden_node)

            if abs(layer_difference) > 1:
                self.nodes.layered[
                    in_node_1_layer + np.sign(layer_difference)
                ].append(new_hidden_node)

            else:
                if layer_difference == 1:
                    latest_layer = out_node_layer
                else:  # layer_difference == -1 or layer_difference == 0:
                    latest_layer = in_node_1_layer

                self.nodes.layered.insert(latest_layer, [])
                self.nodes.layered[latest_layer].append(new_hidden_node)

    def grow_connection(
        self: "Net",
        in_node: Node = None,
        out_node: Node = None,
    ) -> None:
        # If argument `in_node` is not specified, sample it.
        if not in_node:
            # Remove duplicates from the
            potential_in_nodes = list(dict.fromkeys(self.nodes.receiving))
            # Remove nodes (and their duplicates) being pruned.
            for node in self.nodes.being_pruned:
                while node in potential_in_nodes:
                    potential_in_nodes.remove(node)
            # If argument `out_node` was specified, remove its in nodes.
            if out_node:
                for node in out_node.in_nodes:
                    potential_in_nodes.remove(node)
            # Return if no match.
            if len(potential_in_nodes) == 0:
                return
            # Sample randomly from the rest.
            in_node = np.random.choice(potential_in_nodes)
        # If argument `out_node` is not specified, sample it.
        if not out_node:
            # Start with an initial list.
            potential_out_nodes = self.nodes.hidden + self.nodes.output
            # Remove nodes being pruned.
            for node in self.nodes.being_pruned:
                potential_out_nodes.remove(node)
            # Remove
            for node in in_node.out_nodes:
                potential_out_nodes.remove(node)

            if len(potential_out_nodes) == 0:
                return

            out_node = np.random.choice(potential_out_nodes)

        in_node.connect_to(out_node)

        self.nodes.receiving.append(out_node)
        self.nodes["emitting"].append(in_node)

    def prune_node(
        self: "Net",
        node: Node | None = None,
    ) -> None:
        # If the `node` argument is not specified, sample one.
        if not node:
            if len(self.nodes.hidden) == 0:
                return

            node = random.choice(self.nodes.hidden)

        if node in self.nodes.being_pruned:
            return

        # Add the node to the pruning buffer in case
        self.nodes.being_pruned.append(node)

        # Remove all outcoming connections.
        for out_node in node.out_nodes.copy():
            self.prune_connection(node, out_node, node)
        # Remove all incoming connections.
        for in_node in node.in_nodes.copy():
            self.prune_connection(in_node, node, node)

        for key in self.nodes:
            if key == "layered":
                node_layer = find_sublist_index(node, self.nodes.layered)
                self.nodes.layered[node_layer].remove(node)

                if (
                    node_layer != 0
                    and node_layer != len(self.nodes.layered) - 1
                ):
                    if self.nodes.layered[node_layer] == []:
                        self.nodes.layered.remove(
                            self.nodes.layered[node_layer]
                        )
            else:
                while node in self.nodes[key]:
                    self.nodes[key].remove(node)

    def prune_connection(
        self: "Net",
        in_node: Node | None = None,
        out_node: Node | None = None,
        calling_node: Node | None = None,
    ) -> None:
        if not in_node:
            if len(self.nodes["emitting"]) == 0:
                return

            in_node = np.random.choice(self.nodes["emitting"])

        if out_node == None:
            out_node = np.random.choice(in_node.out_nodes)

        connection_was_already_pruned = in_node.disconnect_from(out_node)

        if connection_was_already_pruned:
            return

        self.nodes.receiving.remove(out_node)
        self.nodes["emitting"].remove(in_node)

        if in_node != calling_node:
            if in_node not in self.nodes["emitting"]:
                if in_node in self.nodes.hidden:
                    self.prune_node(in_node)

        if out_node != calling_node:
            if out_node not in self.nodes.receiving:
                if out_node in self.nodes.hidden:
                    self.prune_node(out_node)

        for node in [in_node, out_node]:
            if node != calling_node and node not in self.nodes.being_pruned:
                if node in self.nodes.hidden:
                    if node.in_nodes == [node] or node.out_nodes == [node]:
                        self.prune_node(node)

                elif node in self.nodes.output:
                    if node.in_nodes == [node] and node.out_nodes == [node]:
                        self.prune_connection(node, node)

    def reset(self: "Net") -> None:
        """Resets all node outputs to 0."""
        for node in self.nodes.all:
            node.output = np.ndarray([0])

    def __call__(
        self: "Net",
        x: Float[NDArray, " num_inputs"],
    ):
        for x_i, input_node in zip(x, self.nodes.input, strict=True):
            input_node.output = x_i

        for layer in range(1, len(self.nodes.layered)):
            for node in self.nodes.layered[layer]:
                node.compute()

            for node in self.nodes.layered[layer]:
                node.update()

        return [node.output for node in self.nodes.output]


def find_sublist_index(element, layered_list):
    for sublist_index, sublist in enumerate(layered_list):
        if element in sublist:
            return sublist_index
