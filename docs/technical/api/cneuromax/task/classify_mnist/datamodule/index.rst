:py:mod:`cneuromax.task.classify_mnist.datamodule`
==================================================

.. py:module:: cneuromax.task.classify_mnist.datamodule

.. autoapi-nested-parse::

   Datamodule & config for MNIST classification task.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   cneuromax.task.classify_mnist.datamodule.MNISTClassificationDataModuleConfig
   cneuromax.task.classify_mnist.datamodule.MNISTClassificationDataModule




.. py:class:: MNISTClassificationDataModuleConfig




   Configuration for :class:`MNISTClassificationDataModuleConfig`s.

   :param val_percentage: Percentage of the training dataset to use for            validation.

   .. py:attribute:: val_percentage
      :type: Annotated[float, ge(0), lt(1)]
      :value: 0.1

      


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



