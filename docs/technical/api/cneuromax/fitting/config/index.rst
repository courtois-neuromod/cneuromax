:py:mod:`cneuromax.fitting.config`
==================================

.. py:module:: cneuromax.fitting.config

.. autoapi-nested-parse::

   Root :mod:`hydra-core` fitting config & validation.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   cneuromax.fitting.config.BaseFittingHydraConfig



Functions
~~~~~~~~~

.. autoapisummary::

   cneuromax.fitting.config.pre_process_base_fitting_config
   cneuromax.fitting.config.post_process_base_fitting_config



.. py:class:: BaseFittingHydraConfig




   Base structured :mod:`hydra-core` fitting config.

   :param device: Computing device to use for large matrix operations.
   :param copy_data_commands: List of commands to execute to transfer the            training data to the :paramref:`~.BaseHydraConfig.data_dir`            directory. This is useful when the training data is stored            on a disk that is different from the one used by the            training machine.

   .. py:attribute:: device
      :type: Annotated[str, one_of('cpu', 'gpu')]
      :value: 'cpu'

      

   .. py:attribute:: copy_data_commands
      :type: list[str] | None

      


.. py:function:: pre_process_base_fitting_config(config: omegaconf.DictConfig) -> None

   Validates raw task config before it is made structured.

   Used for changing the computing device if CUDA is not available.

   :param config: The raw task config.


.. py:function:: post_process_base_fitting_config(config: BaseFittingHydraConfig) -> None

   Post-processes the :mod:`hydra-core` config after it is resolved.

   :param config: The processed :mod:`hydra-core` config.


