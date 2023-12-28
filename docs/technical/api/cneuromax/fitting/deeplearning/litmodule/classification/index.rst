:py:mod:`cneuromax.fitting.deeplearning.litmodule.classification`
=================================================================

.. py:module:: cneuromax.fitting.deeplearning.litmodule.classification

.. autoapi-nested-parse::

   Classification :class:`~lightning.pytorch.LightningModule`\s.



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

   cneuromax.fitting.deeplearning.litmodule.classification.BaseClassificationLitModule
   cneuromax.fitting.deeplearning.litmodule.classification.BaseClassificationLitModuleConfig




.. py:class:: BaseClassificationLitModule(config: BaseClassificationLitModuleConfig, nnmodule: torch.nn.Module, optimizer: functools.partial[torch.optim.Optimizer], scheduler: functools.partial[torch.optim.lr_scheduler.LRScheduler])




   Root classification :class:`~lightning.pytorch.LightningModule`.

   :param config: See :class:`BaseClassificationLitModuleConfig`.
   :param nnmodule: See :paramref:`~.BaseLitModule.nnmodule`.
   :param optimizer: See :paramref:`~.BaseLitModule.optimizer`.
   :param scheduler: See :paramref:`~.BaseLitModule.scheduler`.

   .. attribute:: accuracy

      

      :type: :class:`~torchmetrics.classification.MulticlassAccuracy`

   .. py:method:: step(batch: tuple[jaxtyping.Float[torch.Tensor,  batch_size *x_shape], jaxtyping.Int[torch.Tensor,  batch_size]], stage: Annotated[str, one_of('train', 'val', 'test')]) -> jaxtyping.Float[torch.Tensor,  ]

      Computes the model accuracy and cross entropy loss.

      :param batch: See                :paramref:`~.BaseLitModule.stage_step.batch`.
      :param stage: See                :paramref:`~.BaseLitModule.stage_step.stage`.

      :returns: The cross entropy loss.



.. py:class:: BaseClassificationLitModuleConfig


   Holds :class:`BaseClassificationLitModule` config values.

   :param num_classes: Number of classes to classify between.

   .. py:attribute:: num_classes
      :type: Annotated[int, ge(2)]

      


