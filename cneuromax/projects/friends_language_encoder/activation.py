"""."""

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor, nn


class AttachLayerHook:
    """."""

    def __init__(
        self: "AttachLayerHook",
        nnmodule: nn.Module,
        layer_name: str,
    ) -> None:
        self.nnmodule = nnmodule
        self.layer_name = layer_name
        self.activations: dict[str, Tensor] = {}

        self.nnmodule.model.get_submodule(
            self.layer_name,
        ).register_forward_hook(self.get_activation_hook(self.layer_name))

    # Define hook function
    def get_activation_hook(
        self: "AttachLayerHook",
        name: str,
    ) -> Callable[[nn.Module, Tensor, Tensor], None]:
        """."""

        def hook(
            _model: nn.Module,
            _inputs: Tensor,
            output: Tensor,
        ) -> None:
            """."""
            # Detach gradients but only save activations
            self.activations[name] = output.detach()

        return hook

    def get_activations(
        self: "AttachLayerHook",
        *args: Any,
    ) -> dict[str, Tensor]:
        """."""
        with torch.no_grad():
            _ = self.nnmodule(*args)
        return self.activations
