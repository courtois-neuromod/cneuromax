:py:mod:`cneuromax.fitting.deeplearning.litmodule`
==================================================

.. py:module:: cneuromax.fitting.deeplearning.litmodule

.. autoapi-nested-parse::

   Common :class:`~lightning.pytorch.LightningModule`\s.



Subpackages
-----------
.. toctree::
   :titlesonly:
   :maxdepth: 3

   classification/index.rst


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   base/index.rst


Package Contents
----------------

Classes
~~~~~~~

.. autoapisummary::

   cneuromax.fitting.deeplearning.litmodule.BaseLitModule




.. py:class:: BaseLitModule(nnmodule: torch.nn.Module, optimizer: functools.partial[torch.optim.Optimizer], scheduler: functools.partial[torch.optim.lr_scheduler.LRScheduler])




   Root :class:`~lightning.pytorch.LightningModule` subclass.

   Subclasses need to implement the :meth:`step` method that inputs a
   (tupled) batch and returns the loss value(s).

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

   :param nnmodule: The :mod:`torch` Module to wrap.
   :param optimizer: The :mod:`torch` Optimizer to train with.
   :param scheduler: The :mod:`torch` Scheduler to train with.

   .. attribute:: nnmodule

      See            :paramref:`~BaseLitModule.nnmodule`.

      :type: :class:`torch.nn.Module`

   .. attribute:: optimizer

      See            :paramref:`~BaseLitModule.optimizer`.

      :type: :class:`torch.optim.Optimizer`

   .. attribute:: scheduler

      See            :paramref:`~BaseLitModule.scheduler`.

      :type: :class:`torch.optim.lr_scheduler.LRScheduler`

   :raises NotImplementedError: If :meth:`step` is not defined            or not callable.

   .. py:method:: stage_step(batch: jaxtyping.Num[torch.Tensor,  ...] | tuple[jaxtyping.Num[torch.Tensor,  ...], Ellipsis] | list[jaxtyping.Num[torch.Tensor,  ...]], stage: Annotated[str, one_of('train', 'val', 'test', 'predict')]) -> jaxtyping.Num[torch.Tensor,  ...]

      Generic stage wrapper around the :meth:`step` method.

      Verifies that the :meth:`step` method exists and is callable,
      calls it and logs the loss value(s).

      :param batch: The input data batch.
      :param stage: The current stage.

      :returns: The loss value(s).


   .. py:method:: training_step(batch: jaxtyping.Num[torch.Tensor,  ...] | tuple[jaxtyping.Num[torch.Tensor,  ...], Ellipsis] | list[jaxtyping.Num[torch.Tensor,  ...]]) -> jaxtyping.Num[torch.Tensor,  ...]

      Calls :meth:`stage_step` with argument ``stage="train"``.

      :param batch: See :paramrefBaseLitModule.stage_step.batch`.

      :returns: The loss value(s).


   .. py:method:: validation_step(batch: jaxtyping.Num[torch.Tensor,  ...] | tuple[jaxtyping.Num[torch.Tensor,  ...], Ellipsis] | list[jaxtyping.Num[torch.Tensor,  ...]], *args: Any, **kwargs: Any) -> jaxtyping.Num[torch.Tensor,  ...]

      Calls :meth:`stage_step` with argument ``stage="val"``.

      :param batch: See :paramref:`~BaseLitModule.stage_step.batch`.
      :param \*args: Additional positional arguments.
      :param \*\*kwargs: Additional keyword arguments.

      :returns: The loss value(s).


   .. py:method:: test_step(batch: jaxtyping.Num[torch.Tensor,  ...] | tuple[jaxtyping.Num[torch.Tensor,  ...], Ellipsis] | list[jaxtyping.Num[torch.Tensor,  ...]]) -> jaxtyping.Num[torch.Tensor,  ...]

      Calls :meth:`stage_step` with argument ``stage="test"``.

      :param batch: See :paramref:`~BaseLitModule.stage_step.batch`.

      :returns: The loss value(s).


   .. py:method:: configure_optimizers() -> tuple[list[torch.optim.Optimizer], list[dict[str, torch.optim.lr_scheduler.LRScheduler | str | int]]]

      Returns a dict w/ ``optimizer`` & ``scheduler`` attributes.

      :returns: A tuple containing this instance's            :class:`~torch.optim.Optimizer` and            :class:`~torch.optim.lr_scheduler.LRScheduler`            attributes.



