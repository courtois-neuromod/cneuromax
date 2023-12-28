:py:mod:`cneuromax.fitting.deeplearning.datamodule`
===================================================

.. py:module:: cneuromax.fitting.deeplearning.datamodule

.. autoapi-nested-parse::

   Common :class:`~lightning.pytorch.LightningDataModule`\s.



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

   cneuromax.fitting.deeplearning.datamodule.BaseDataModule
   cneuromax.fitting.deeplearning.datamodule.BaseDataModuleConfig
   cneuromax.fitting.deeplearning.datamodule.StageDataset




.. py:class:: BaseDataModule(config: BaseDataModuleConfig)




   Root :mod:`~lightning.pytorch.LightningDataModule` subclass.

   With ``stage`` being any of ``train``, ``val``, ``test`` or
   ``predict``, subclasses need to properly define the
   ``dataset.stage`` instance attribute(s) for each desired ``stage``.

   :param config: See :class:`BaseDataModuleConfig`.

   .. attribute:: config

      

      :type: :class:`BaseDataModuleConfig`

   .. attribute:: dataset

      

      :type: :class:`StageDataset`

   .. attribute:: pin_memory

      Whether to copy tensors into device            pinned memory before returning them (is set to ``True`` by            default if :paramref:`~BaseDataModuleConfig.device` is            ``"gpu"``).

      :type: ``bool``

   .. attribute:: per_device_batch_size

      Per-device number of samples            to load per iteration. Default value (``1``) is later            overwritten through function            :func:`~.deeplearning.fit.set_batch_size_and_num_workers`.

      :type: ``int``

   .. attribute:: per_device_num_workers

      Per-device number of CPU            processes to use for data loading (``0`` means that the            data will be loaded by each device's assigned CPU            process). Default value (``0``) is later overwritten            through function            :func:`~.deeplearning.fit.set_batch_size_and_num_workers`.

      :type: ``int``

   .. py:method:: load_state_dict(state_dict: dict[str, int]) -> None

      Loads saved ``per_device_batch_size`` & ``num_workers`` vals.

      :param state_dict: Dictionary containing values for                ``per_device_batch_size`` & ``num_workers``.


   .. py:method:: state_dict() -> dict[str, int]

      Returns ``per_device_batch_size`` & ``num_workers`` attribs.

      :returns: See :paramref:`~BaseDataModule.load_state_dict.state_dict`.


   .. py:method:: x_dataloader(dataset: torch.utils.data.Dataset[torch.Tensor] | None, *, shuffle: bool = True) -> torch.utils.data.DataLoader[torch.Tensor]

      Generic :class:`~torch.utils.data.DataLoader` factory method.

      :param dataset: The dataset to wrap with a                :class:`~torch.utils.data.DataLoader`
      :param shuffle: Whether to shuffle the dataset when iterating                over it.

      :raises AttributeError: If :paramref:`dataset` is ``None``.

      :returns: A new :class:`~torch.utils.data.DataLoader` instance                wrapping the :paramref:`dataset` argument.


   .. py:method:: train_dataloader() -> torch.utils.data.DataLoader[torch.Tensor]

      Calls :meth:`x_dataloader` with ``dataset.train`` attribute.

      :returns: A new training :class:`torch.utils.data.DataLoader`                instance.


   .. py:method:: val_dataloader() -> torch.utils.data.DataLoader[torch.Tensor]

      Calls :meth:`x_dataloader` with ``dataset.val`` attribute.

      :returns: A new validation :class:`~torch.utils.data.DataLoader`                instance.


   .. py:method:: test_dataloader() -> torch.utils.data.DataLoader[torch.Tensor]

      Calls :meth:`x_dataloader` with ``dataset.test`` attribute.

      :returns: A new testing :class:`~torch.utils.data.DataLoader`                instance.


   .. py:method:: predict_dataloader() -> torch.utils.data.DataLoader[torch.Tensor]

      Calls :meth:`x_dataloader` w/ ``dataset.predict`` attribute.

      :returns: A new prediction :class:`~torch.utils.data.DataLoader`                instance that does not shuffle the dataset.



.. py:class:: BaseDataModuleConfig


   Holds :class:`BaseDataModule` config values.

   :param data_dir: See            :paramref:`~cneuromax.config.BaseHydraConfig.data_dir`.
   :param device: See            :paramref:`~cneuromax.fitting.config.BaseFittingHydraConfig.device`.

   .. py:attribute:: data_dir
      :type: Annotated[str, not_empty()]
      :value: '${data_dir}'

      

   .. py:attribute:: device
      :type: Annotated[str, one_of('cpu', 'gpu')]
      :value: '${device}'

      


.. py:class:: StageDataset


   Holds stage-specific :class:`~torch.utils.data.Dataset` objects.

   :param train: Training dataset.
   :param val: Validation dataset.
   :param test: Testing dataset.
   :param predict: Prediction dataset.

   .. py:attribute:: train
      :type: torch.utils.data.Dataset[torch.Tensor] | None

      

   .. py:attribute:: val
      :type: torch.utils.data.Dataset[torch.Tensor] | None

      

   .. py:attribute:: test
      :type: torch.utils.data.Dataset[torch.Tensor] | None

      

   .. py:attribute:: predict
      :type: torch.utils.data.Dataset[torch.Tensor] | None

      


