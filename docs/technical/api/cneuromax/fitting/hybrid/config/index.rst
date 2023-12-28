:py:mod:`cneuromax.fitting.hybrid.config`
=========================================

.. py:module:: cneuromax.fitting.hybrid.config

.. autoapi-nested-parse::

   :mod:`hydra-core` DL + NE fitting config & validation.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   cneuromax.fitting.hybrid.config.HybridFittingHydraConfig



Functions
~~~~~~~~~

.. autoapisummary::

   cneuromax.fitting.hybrid.config.pre_process_hybrid_fitting_config
   cneuromax.fitting.hybrid.config.post_process_hybrid_fitting_config



.. py:class:: HybridFittingHydraConfig




   Structured :mod:`hydra-core` config for DL + NE fitting.


.. py:function:: pre_process_hybrid_fitting_config(config: omegaconf.DictConfig) -> None

   Validates raw task config before it is made structured.

   :param config: The raw task config.


.. py:function:: post_process_hybrid_fitting_config(config: HybridFittingHydraConfig) -> None

   Post-processes the :mod:`hydra-core` config after it is resolved.

   :param config: The processed :mod:`hydra-core` config.


