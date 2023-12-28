:py:mod:`cneuromax.config`
==========================

.. py:module:: cneuromax.config

.. autoapi-nested-parse::

   Root :mod:`hydra-core` config & utilities.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   cneuromax.config.BaseHydraConfig



Functions
~~~~~~~~~

.. autoapisummary::

   cneuromax.config.pre_process_base_config
   cneuromax.config.process_config
   cneuromax.config.post_process_base_config



Attributes
~~~~~~~~~~

.. autoapisummary::

   cneuromax.config.T


.. py:class:: BaseHydraConfig


   Base structured :mod:`hydra-core` configuration.

   :param run_dir: Path to the run's directory. Every artifact generated            during the run will be stored in this directory.
   :param data_dir: Path to the data directory. This directory is            typically shared between runs. It is used to store            datasets, pre-trained models, etc.

   .. py:attribute:: run_dir
      :type: Annotated[str, not_empty()]
      :value: 'data/untitled_run/'

      

   .. py:attribute:: data_dir
      :type: Annotated[str, not_empty()]
      :value: '${run_dir}/../'

      


.. py:function:: pre_process_base_config(config: omegaconf.DictConfig) -> None

   Validates raw task config before it is made structured.

   Makes sure that the ``run_dir`` does not already exist. If it does,
   it loops through ``{run_dir}_1``, ``{run_dir}_2``, etc. until it
   finds a directory that does not exist.

   :param config: The raw task config.


.. py:data:: T

   

.. py:function:: process_config(config: omegaconf.DictConfig, structured_config_class: type[T]) -> T

   Turns the raw task config into a structured config.

   :param config: See :paramref:`pre_process_base_config.config`.
   :param structured_config_class: The structured config class to turn            the raw config into.

   :returns: The processed structured Hydra config.


.. py:function:: post_process_base_config(config: BaseHydraConfig) -> None

   Validates the structured task config.

   Creates the run directory if it does not exist.

   :param config: The processed :mod:`hydra-core` config.


