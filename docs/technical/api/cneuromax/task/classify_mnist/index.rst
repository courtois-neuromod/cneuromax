:py:mod:`cneuromax.task.classify_mnist`
=======================================

.. py:module:: cneuromax.task.classify_mnist

.. autoapi-nested-parse::

   MNIST classification task.



Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   datamodule/index.rst
   datamodule_test/index.rst
   litmodule/index.rst


Package Contents
----------------

Classes
~~~~~~~

.. autoapisummary::

   cneuromax.task.classify_mnist.MNISTClassificationDataModule
   cneuromax.task.classify_mnist.MNISTClassificationDataModuleConfig
   cneuromax.task.classify_mnist.MNISTClassificationLitModule




.. py:class:: MNISTClassificationDataModule(config: MNISTClassificationDataModuleConfig)




   .

   .. attribute:: train_val_split

      The train/validation            split (sums to `1`).

      :type: `tuple[float, float]`

   .. attribute:: transform

      The            :mod:`torchvision` dataset transformations.

      :type: :class:`~transforms.Compose`

   .. py:method:: prepare_data() -> None

      Downloads the MNIST dataset.


   .. py:method:: setup(stage: Annotated[str, one_of('fit', 'test')]) -> None

      Creates the train/val/test datasets.

      :param stage: Current stage type.



.. py:class:: MNISTClassificationDataModuleConfig




   Configuration for :class:`MNISTClassificationDataModuleConfig`s.

   :param val_percentage: Percentage of the training dataset to use for            validation.

   .. py:attribute:: val_percentage
      :type: Annotated[float, ge(0), lt(1)]
      :value: 0.1

      


.. py:class:: MNISTClassificationLitModule(nnmodule: torch.nn.Module, optimizer: functools.partial[torch.optim.Optimizer], scheduler: functools.partial[torch.optim.lr_scheduler.LRScheduler])




   MNIST classification Lightning Module.


