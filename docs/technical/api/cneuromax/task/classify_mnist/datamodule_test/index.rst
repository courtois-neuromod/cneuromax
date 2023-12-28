:py:mod:`cneuromax.task.classify_mnist.datamodule_test`
=======================================================

.. py:module:: cneuromax.task.classify_mnist.datamodule_test

.. autoapi-nested-parse::

   Tests for :mod:`~cneuromax.task.classify_mnist.datamodule`.



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   cneuromax.task.classify_mnist.datamodule_test.datamodule
   cneuromax.task.classify_mnist.datamodule_test.test_setup_fit
   cneuromax.task.classify_mnist.datamodule_test.test_setup_test



.. py:function:: datamodule(tmp_path: pathlib.Path) -> cneuromax.task.classify_mnist.MNISTClassificationDataModule

   `MNISTClassificationDataModule` fixture.

   :param tmp_path: The temporary path for the            :class:`~.MNISTClassificationDataModule`.

   :returns: A generic `MNISTDataModule` instance.


.. py:function:: test_setup_fit(datamodule: cneuromax.task.classify_mnist.MNISTClassificationDataModule) -> None

   Tests the `MNISTClassificationDataModule.setup` method #1.

   :param datamodule: A generic `MNISTDataModule` instance, see
                      :func:`datamodule`.


.. py:function:: test_setup_test(datamodule: cneuromax.task.classify_mnist.MNISTClassificationDataModule) -> None

   Tests the `MNISTClassificationDataModule.setup` method #2.

   :param datamodule: A generic `MNISTDataModule` instance, see
                      :func:`datamodule`.


