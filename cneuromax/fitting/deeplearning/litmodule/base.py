""":class:`.BaseLitModule` & its config."""

from abc import ABC, abstractmethod
from copy import copy
from dataclasses import dataclass
from functools import partial
from typing import Annotated as An
from typing import Any, final

import wandb
from jaxtyping import Num
from lightning.pytorch import LightningModule
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from cneuromax.utils.beartype import ge, one_of


@dataclass
class BaseLitModuleConfig:
    """Holds :class:`BaseLitModule` config values.

    Args:
        wandb_column_names
        wandb_train_log_interval
        wandb_num_samples
    """

    wandb_column_names: str
    wandb_train_log_interval: int = 50
    wandb_num_samples: An[int, ge(1)] = 3


class BaseLitModule(LightningModule, ABC):
    """Base :class:`lightning.pytorch.core.LightningModule`.

    Subclasses need to implement the :meth:`step` method that inputs
    both ``data`` and  ``stage`` arguments while returning the loss
    value(s) in the form of a :class:`torch.Tensor`.

    Example definition:

    .. highlight:: python
    .. code-block:: python

        def step(
            self: "BaseClassificationLitModule",
            data: tuple[
                Float[Tensor, " batch_size *x_dim"],
                Int[Tensor, " batch_size"],
            ],
            stage: An[str, one_of("train", "val", "test")],
        ) -> Float[Tensor, " "]:
            ...

    Note:
        In :mod:`cneuromax`, we propose to split the PyTorch module
        definition from the Lightning module definition for (arguably)
        better code organization, reuse & readability. As a result, each
        Lightning module receives a PyTorch module as an argument which
        it turns into an instance attribute. This is in contrast with
        the suggested Lightning best practices where Lightning modules
        subclass PyTorch modules, and thus allow PyTorch module method
        definitions in the Lightning module.

    Args:
        config
        nnmodule
        optimizer
        scheduler

    Attributes:
        config: See :paramref:`~BaseLitModule.config`.
        nnmodule: See :paramref:`~BaseLitModule.nnmodule`.
        optimizer_partial: See :paramref:`~BaseLitModule.optimizer`.
        scheduler_partial : See :paramref:`~BaseLitModule.scheduler`.
        optimizer: :paramref:`~BaseLitModule.optimizer` instantiated.
        scheduler: :paramref:`~BaseLitModule.scheduler` instantiated.
        curr_train_step
        curr_val_epoch
        wandb_train_table (wandb.Table): Table containing the rich
            training data that gets logged to W&B.
        wandb_train_data (list[dict[str, Any]]): A list of dictionaries
            containing validation data relating to one specific example
            (ex: `input_data`, `logits`, ...).
        wandb_val_table (wandb.Table): See :attr:`wandb_train_table`.
        wandb_val_data (list[dict[str, Any]]): See
            :attr:`wandb_train_data`.
    """

    def __init__(
        self: "BaseLitModule",
        config: BaseLitModuleConfig,
        nnmodule: nn.Module,
        optimizer: partial[Optimizer],
        scheduler: partial[LRScheduler],
    ) -> None:
        super().__init__()
        self.config = config
        self.nnmodule = nnmodule
        self.optimizer_partial = optimizer
        self.scheduler_partial = scheduler
        self.curr_train_step = 0
        self.curr_val_epoch = 0
        self.initialize_wandb_objects()

    def initialize_wandb_objects(self: "BaseLitModule") -> None:  # noqa: D102
        create_wandb_table = lambda iter_type: wandb.Table(  # type: ignore[no-untyped-call]  # noqa: E731
            columns=[
                "data_idx",
                iter_type,
                *self.config.wandb_column_names.split(),
            ],
        )
        self.wandb_train_table = create_wandb_table("train_step")
        self.wandb_val_table = create_wandb_table("val_epoch")
        self.wandb_train_data: list[dict[str, Any]] = []
        self.wandb_val_data: list[dict[str, Any]] = []

    def on_save_checkpoint(  # noqa: D102
        self: "BaseLitModule",
        checkpoint: dict[str, Any],
    ) -> None:
        checkpoint["curr_train_step"] = self.curr_train_step
        checkpoint["curr_val_epoch"] = self.curr_val_epoch
        return super().on_save_checkpoint(checkpoint)

    def on_load_checkpoint(  # noqa: D102
        self: "BaseLitModule",
        checkpoint: dict[str, Any],
    ) -> None:
        self.curr_train_step = checkpoint["curr_train_step"]
        self.curr_val_epoch = checkpoint["curr_val_epoch"]
        return super().on_load_checkpoint(checkpoint)

    def on_train_batch_start(  # noqa: D102
        self: "BaseLitModule",
        *args: Any,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        self.wandb_train_data = []
        super().on_train_batch_start(*args, **kwargs)

    def on_validation_start(self: "BaseLitModule") -> None:  # noqa: D102
        self.wandb_val_data = []
        super().on_validation_start()

    def optimizer_step(  # noqa: D102
        self: "BaseLitModule",
        *args: Any,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        super().optimizer_step(*args, **kwargs)
        if self.curr_train_step % self.config.wandb_train_log_interval == 0:
            self.log_table(self.wandb_train_data)
        self.curr_train_step += 1

    def on_validation_epoch_end(self: "BaseLitModule") -> None:  # noqa: D102
        super().on_validation_epoch_end()
        self.log_table(self.wandb_val_data)
        self.curr_val_epoch += 1

    def log_table(  # noqa: D102
        self: "BaseLitModule",
        data: list[dict[str, Any]],
    ) -> None:
        if data is self.wandb_train_data:
            name = "train_data"
            table = self.wandb_train_table
            it = self.curr_train_step
        else:  # data is self.wandb_val_data
            name = "val_data"
            table = self.wandb_val_table
            it = self.curr_val_epoch
        for i, data_i in enumerate(data):
            table.add_data(  # type: ignore[no-untyped-call]
                i,
                it,
                *[
                    data_i[key]
                    for key in self.config.wandb_column_names.split()
                ],
            )
        # 1) Static type checking discrepancy:
        # `logger.experiment` is a `wandb.wandb_run.Run` instance.
        # 2) Cannot log the same table twice:
        # https://github.com/wandb/wandb/issues/2981#issuecomment-1458447291
        try:
            self.logger.experiment.log(  # type: ignore[union-attr]
                {name: copy(table)},
            )
        except Exception as e:
            error_msg = (
                "Failed to log validation data to W&B. "
                "You might be trying to log tensors."
            )
            raise ValueError(error_msg) from e

    @abstractmethod
    def step(self, data, stage): ...  # type: ignore [no-untyped-def]  # noqa: ANN001, ANN201, D102

    @final
    def stage_step(
        self: "BaseLitModule",
        data: Any,  # noqa: ANN401
        stage: An[str, one_of("train", "val", "test", "predict")],
    ) -> Num[Tensor, " *_"]:
        """Generic stage wrapper around the :meth:`step` method.

        Verifies that the :meth:`step` method exists and is callable,
        calls it and logs the loss value(s).

        Args:
            data: The batched input data.
            stage: The current stage (``train``, ``val``, ``test`` or
                ``predict``).

        Returns:
            The loss value(s).
        """
        if isinstance(data, list):
            data = tuple(data)
        loss: Num[Tensor, " *_"] = self.step(data, stage)  # type: ignore [no-untyped-call]
        self.log(name=f"{stage}/loss", value=loss)
        return loss

    @final
    def training_step(
        self: "BaseLitModule",
        data: Any,  # noqa: ANN401
    ) -> Num[Tensor, " *_"]:
        """Calls :meth:`stage_step` with argument ``stage="train"``.

        Args:
            data: See :paramref:`~stage_step.data`.

        Returns:
            The loss value(s).
        """
        return self.stage_step(data=data, stage="train")

    @final
    def validation_step(
        self: "BaseLitModule",
        data: Any,  # noqa: ANN401
        # :paramref:`*args` & :paramref:`**kwargs` type annotations
        # cannot be more specific because of
        # :meth:`LightningModule.validation_step`\'s signature.
        *args: Any,  # noqa: ANN401, ARG002
        **kwargs: Any,  # noqa: ANN401, ARG002
    ) -> Num[Tensor, " *_"]:
        """Calls :meth:`stage_step` with argument ``stage="val"``.

        Args:
            data: See :paramref:`~stage_step.data`.

        Returns:
            The loss value(s).
        """
        return self.stage_step(data=data, stage="val")

    @final
    def test_step(
        self: "BaseLitModule",
        data: Any,  # noqa: ANN401
    ) -> Num[Tensor, " *_"]:
        """Calls :meth:`stage_step` with argument ``stage="test"``.

        Args:
            data: See :paramref:`~stage_step.data`.

        Returns:
            The loss value(s).
        """
        return self.stage_step(data=data, stage="test")

    def configure_optimizers(
        self: "BaseLitModule",
    ) -> tuple[list[Optimizer], list[LRScheduler]]:
        """Returns a dict with :attr:`optimizer` and :attr:`scheduler`.

        Returns:
            This instance's
                :attr:`~torch.optim.Optimizer` and
                :class:`torch.optim.lr_scheduler.LRScheduler`.
        """
        self.optimizer = self.optimizer_partial(params=self.parameters())
        self.scheduler = self.scheduler_partial(optimizer=self.optimizer)
        return [self.optimizer], [self.scheduler]
