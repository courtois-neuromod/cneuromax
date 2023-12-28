:py:mod:`cneuromax.fitting.deeplearning.utils.lightning`
========================================================

.. py:module:: cneuromax.fitting.deeplearning.utils.lightning

.. autoapi-nested-parse::

   :mod:`lightning` utilities.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   cneuromax.fitting.deeplearning.utils.lightning.InitOptimParamsCheckpointConnector



Functions
~~~~~~~~~

.. autoapisummary::

   cneuromax.fitting.deeplearning.utils.lightning.instantiate_lightning_objects
   cneuromax.fitting.deeplearning.utils.lightning.set_batch_size_and_num_workers
   cneuromax.fitting.deeplearning.utils.lightning.set_checkpoint_path
   cneuromax.fitting.deeplearning.utils.lightning.find_good_per_device_batch_size
   cneuromax.fitting.deeplearning.utils.lightning.find_good_num_workers



.. py:function:: instantiate_lightning_objects(config: cneuromax.fitting.deeplearning.config.DeepLearningFittingHydraConfig, launcher_config: hydra_plugins.hydra_submitit_launcher.config.LocalQueueConf | hydra_plugins.hydra_submitit_launcher.config.SlurmQueueConf) -> tuple[lightning.pytorch.loggers.Logger | None, lightning.pytorch.Trainer, cneuromax.fitting.deeplearning.datamodule.BaseDataModule, cneuromax.fitting.deeplearning.litmodule.BaseLitModule]

   Creates several :mod:`lightning` objects based on the config.

   :param config: See :class:`~.DeepLearningFittingHydraConfig`.
   :param launcher_config: The :mod:`hydra-core` launcher configuration.

   :returns: The instantiated :mod:`lightning` objects.


.. py:function:: set_batch_size_and_num_workers(config: cneuromax.fitting.deeplearning.config.DeepLearningFittingHydraConfig, trainer: lightning.pytorch.Trainer, datamodule: cneuromax.fitting.deeplearning.datamodule.BaseDataModule) -> None

   Computes and sets the batch size and number of workers.

   If starting a new HPO run, finds and sets "good" ``batch_size`` and
   ``num_workers`` parameters.

   See :func:`find_good_batch_size` and :func:`find_good_num_workers`
   functions documentation for more details.

   We make the assumption that if we are resuming from a checkpoint
   created while running hyper-parameter optimization, we are running
   on the same hardware configuration as was used to create the
   checkpoint. Therefore, we do not need to once again look for good
   ``batch_size`` and ``num_workers`` parameters.

   :param config: See :class:`~.DeepLearningFittingHydraConfig`.
   :param trainer: The :class:`lightning.pytorch.Trainer` instance used            for this fitting run.
   :param datamodule: The :class:`~.BaseDataModule` instance used for            this fitting run.


.. py:function:: set_checkpoint_path(config: cneuromax.fitting.deeplearning.config.DeepLearningFittingHydraConfig, trainer: lightning.pytorch.Trainer) -> str | None

   Sets the path to the checkpoint to resume training from.

   TODO: Implement when enabling the Orion sweeper.

   :param config: See :class:`~.DeepLearningFittingHydraConfig`.
   :param trainer: The :class:`lightning.pytorch.Trainer` instance used            for this fitting run.

   :returns: The path to the checkpoint to resume training from.


.. py:function:: find_good_per_device_batch_size(litmodule: cneuromax.fitting.deeplearning.litmodule.BaseLitModule, datamodule: cneuromax.fitting.deeplearning.datamodule.BaseDataModule, device: str, data_dir: str) -> int

   Finds an appropriate ``per_device_batch_size`` parameter.

   This functionality makes the following, not always correct, but
   generally reasonable assumptions:
   - As long as the ``total_batch_size / dataset_size`` ratio remains
   small (e.g. ``< 0.01`` so as to benefit from the stochasticity of
   gradient updates), running the same number of gradient updates with
   a larger batch size will yield better training performance than
   running the same number of gradient updates with a smaller batch
   size.
   - Loading data from disk to RAM is a larger bottleneck than loading
   data from RAM to GPU VRAM.
   - If you are training on multiple GPUs, each GPU has roughly the
   same amount of VRAM.

   :param litmodule: A temporary :class:`~.BaseLitModule` instance with            the same configuration as the :class:`~.BaseLitModule`            instance that will be trained.
   :param datamodule: A temporary :class:`~.BaseDataModule` instance with            the same configuration as the :class:`~.BaseDataModule`            instance that will be used for training.
   :param device: See            :paramref:`~cneuromax.fitting.config.BaseFittingHydraConfig.device`.
   :param data_dir: See            :paramref:`~cneuromax.fitting.config.BaseFittingHydraConfig.data_dir`.

   :returns: The estimated proper batch size per device.


.. py:function:: find_good_num_workers(datamodule_config: Any, per_device_batch_size: int, max_num_data_passes: int = 100) -> int

   Finds an appropriate ``num_workers`` parameter.

   This function makes use of the ``per_device_batch_size`` parameter
   found by the ``find_good_per_device_batch_size`` function in order
   to find an appropriate ``num_workers`` parameter.
   It does so by iterating through a range of ``num_workers`` values
   and measuring the time it takes to iterate through a fixed number of
   data passes; picking the ``num_workers`` value that yields the
   shortest time.

   :param datamodule_config: Implicit (generated by :mod:`hydra-zen`)            ``DataModuleHydraConfig`` instance.
   :param per_device_batch_size: The batch size returned by            :func:`find_good_per_device_batch_size`.
   :param max_num_data_passes: Maximum number of data passes to iterate            through (default: ``100``).

   :returns: The estimated proper number of workers.


.. py:class:: InitOptimParamsCheckpointConnector(trainer: lightning.pytorch.Trainer)




   Tweaked ckpt connector to preserve newly instantiated parameters.

   Allows to make use of the newly instantiated optimizers'
   hyper-parameters rather than the checkpointed hyper-parameters.
   For use when resuming training with different optimizer
   hyper-parameters (e.g. with the PBT :mod:`hydra-core` sweeper).

   .. py:method:: restore_optimizers() -> None

      Tweaked method to preserve newly instantiated parameters.



