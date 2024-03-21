"""."""

from collections.abc import Callable
from typing import Any

import torch
from jaxtyping import Num
from torch import Tensor, nn

from cneuromax.projects.friends_language_encoder.litmodule import (
    FriendsLitModuleConfig,
)


class AttachLayerHook:
    """."""

    def __init__(
        self: "AttachLayerHook",
        checkpoint_dir: str,
        nnmodule: nn.Module,
        layer_name: str,
    ) -> None:
        self.nnmodule = nnmodule
        self.checkpoint_dir = checkpoint_dir
        self.layer_name = layer_name
        self.activations: dict[str, Tensor] = {}

        self.fine_tuned_model = self.nnmodule.load_from_checkpoint(
            self.checkpoint_dir,
        )

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

    def attach_hook(self: "AttachLayerHook") -> None:
        """Attach the hook to the desired layer."""
        self.fine_tuned_model.model.get_submodule(
            self.layer_name,
        ).register_forward_hook(self.get_activation_hook(self.layer_name))


class ExtractLayerActivation:
    """."""

    def __init__(
        self: "ExtractLayerActivation",
        hook: AttachLayerHook,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> None:
        self.hook = hook
        self.input_ids = input_ids
        self.attention_mask = attention_mask

        # Ensure gradients are not computed to save memory

    def extract_activations(self: "ExtractLayerActivation") -> Tensor:
        """."""
        with torch.no_grad():
            self.hook.fine_tuned_model(self.input_ids, self.attention_mask)

        # Retrieve the activations from the hook:
        return self.hook.activations[self.hook.layer_name]


class ExtractActivation:
    """."""

    def __init__(
        self: "ExtractActivation",
        nnmodule: nn.Module,
        checkpoint_dir: str,
        layer_name: str,
        config: "FriendsLitModuleConfig",
        *args: Any,
    ) -> None:
        """Initializes the Activation class.

        Args:
          nnmodule: The fine-tuned model architecture.
          checkpoint_dir: Directory containing the model checkpoint.
          layer_name: Name of the layer to extract activations from.
          config: Configuration for the FriendsLitModule.
          *args:
            input_ids: Input IDs for the model.
            attention_mask: Attention mask for the input.
        """
        self.nnmodule = nnmodule
        self.checkpoint_dir = checkpoint_dir
        self.layer_name = layer_name
        self.config = config
        self.input_ids, self.attention_mask = args
        self.hook = AttachLayerHook(
            self.checkpoint_dir,
            self.nnmodule,
            self.layer_name,
        )

    def _get_activation(self: "ExtractActivation") -> Num[Tensor, " ..."]:
        """Extracts activations using the pre-created hook."""
        layer_activation = ExtractLayerActivation(
            self.hook,
            self.input_ids,
            self.attention_mask,
        )
        activations: Tensor = layer_activation.extract_activations()

        return activations
