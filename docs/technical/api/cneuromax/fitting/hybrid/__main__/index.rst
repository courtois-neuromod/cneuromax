:py:mod:`cneuromax.fitting.hybrid.__main__`
===========================================

.. py:module:: cneuromax.fitting.hybrid.__main__

.. autoapi-nested-parse::

   Entry point for Fitting with Deep Learning.



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   cneuromax.fitting.hybrid.__main__.run



.. py:function:: run(dict_config: omegaconf.DictConfig) -> None

   Processes the :mod:`hydra-core` config and fits w/ DL + NE.

   :param dict_config: The raw config object created by the
                       :func:`hydra.main` decorator.


