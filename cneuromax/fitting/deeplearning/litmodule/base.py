""":class:`BaseLitModule`."""

from abc import ABCMeta
from functools import partial
from typing import Annotated as An
from typing import Any, final

from jaxtyping import Num
from lightning.pytorch import LightningModule
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from cneuromax.utils.annotations import one_of


class BaseLitModule(LightningModule, metaclass=ABCMeta):
    """Root :mod:`lightning` ``LitModule``.

    Subclasses need to implement the :meth:`step` method that inputs
    both a tuple (``batch``) of :class:`~torch.Tensor` and string
    (``stage``) while returning the loss value(s) also in the form of
    a :class:`~torch.Tensor`.

    Example definition:

    .. highlight:: python
    .. code-block:: python

        def step(
            self: "BaseClassificationLitModule",
            batch: tuple[
                Float[Tensor, " batch_size *x_shape"],
                Int[Tensor, " batch_size"],
            ],
            stage: An[str, one_of("train", "val", "test")],
        ) -> Float[Tensor, " "]:
            ...

    Warning:
        ``batch`` and loss value(s) type hints in this class are not
        rendered properly in the documentation due to an\
        incompatibility between :mod:`sphinx` and :mod:`jaxtyping`.\
        Refer to the source code available next to the method\
        signatures to find the correct types.

    Args:
        nnmodule: The :mod:`torch` ``nn.Module`` to be used by this\
            instance.
        optimizer: The :mod:`torch` ``Optimizer`` to be used by this\
            instance. It is partial as an argument as the\
            :paramref:`nnmodule` parameters are required for its\
            initialization.
        scheduler: The :mod:`torch` ``Scheduler`` to be used by this\
            instance. It is partial as an argument as the\
            :paramref:`optimizer` is required for its initialization.

    Attributes:
        nnmodule (:class:`torch.nn.Module`): See\
            :paramref:`~BaseLitModule.nnmodule`.
        optimizer (:class:`torch.optim.Optimizer`): See\
            :paramref:`~BaseLitModule.optimizer`.
        scheduler (:class:`torch.optim.lr_scheduler.LRScheduler`): See\
            :paramref:`~BaseLitModule.scheduler`.

    Raises:
        :class:`NotImplementedError`: If the :meth:`step` method is not\
            defined or not callable.
    """

    def __init__(
        self: "BaseLitModule",
        nnmodule: nn.Module,
        optimizer: partial[Optimizer],
        scheduler: partial[LRScheduler],
    ) -> None:
        super().__init__()
        self.nnmodule = nnmodule
        self.optimizer = optimizer(params=self.parameters())
        self.scheduler = scheduler(optimizer=self.optimizer)
        if not callable(getattr(self, "step", None)):
            error_msg = (
                "The `BaseLitModule.step` method is not defined/not callable."
            )
            raise NotImplementedError(error_msg)

    @final
    def stage_step(
        self: "BaseLitModule",
        batch: Num[Tensor, " ..."]
        | tuple[Num[Tensor, " ..."], ...]
        | list[Num[Tensor, " ..."]],
        stage: An[str, one_of("train", "val", "test", "predict")],
    ) -> Num[Tensor, " ..."]:
        """Generic stage wrapper around the :meth:`step` method.

        Verifies that the :meth:`step` method exists and is callable,
        calls it and logs the loss value(s).

        Args:
            batch: The batched input data.
            stage: The current stage (``train``, ``val``, ``test`` or\
                ``predict``).

        Returns:
            The loss value(s).
        """
        if isinstance(batch, list):
            tupled_batch: tuple[Num[Tensor, " ..."], ...] = tuple(batch)
        loss: Num[Tensor, " ..."] = self.step(tupled_batch, stage)
        self.log(name=f"{stage}/loss", value=loss)
        return loss

    @final
    def training_step(
        self: "BaseLitModule",
        batch: Num[Tensor, " ..."]
        | tuple[Num[Tensor, " ..."], ...]
        | list[Num[Tensor, " ..."]],
    ) -> Num[Tensor, " ..."]:
        """Calls :meth:`stage_step` with argument ``stage="train"``.

        Args:
            batch: See :paramref:`~stage_step.batch`.

        Returns:
            The loss value(s).
        """
        return self.stage_step(batch=batch, stage="train")

    @final
    def validation_step(
        self: "BaseLitModule",
        batch: Num[Tensor, " ..."]
        | tuple[Num[Tensor, " ..."], ...]
        | list[Num[Tensor, " ..."]],
        # :paramref:`*args` & :paramref:`**kwargs` type annotations
        # cannot be more specific because of
        # :meth:`LightningModule.validation_step`\'s signature.
        *args: Any,  # noqa: ANN401, ARG002
        **kwargs: Any,  # noqa: ANN401, ARG002
    ) -> Num[Tensor, " ..."]:
        """Calls :meth:`stage_step` with argument ``stage="val"``.

        Args:
            batch: See :paramref:`~stage_step.batch`.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            The loss value(s).
        """
        return self.stage_step(batch=batch, stage="val")

    @final
    def test_step(
        self: "BaseLitModule",
        batch: Num[Tensor, " ..."]
        | tuple[Num[Tensor, " ..."], ...]
        | list[Num[Tensor, " ..."]],
    ) -> Num[Tensor, " ..."]:
        """Calls :meth:`stage_step` with argument ``stage="test"``.

        Args:
            batch: See :paramref:`~stage_step.batch`.

        Returns:
            The loss value(s).
        """
        return self.stage_step(batch=batch, stage="test")

    @final
    def configure_optimizers(
        self: "BaseLitModule",
    ) -> tuple[list[Optimizer], list[dict[str, LRScheduler | str | int]]]:
        """Returns a dict with :attr:`optimizer` and :attr:`scheduler`.

        Returns:
            A tuple containing this instance's\
            :class:`~torch.optim.Optimizer` and\
            :class:`~torch.optim.lr_scheduler.LRScheduler`\
            attributes.
        """
        return [self.optimizer], [
            {"scheduler": self.scheduler, "interval": "step", "frequency": 1},
        ]
