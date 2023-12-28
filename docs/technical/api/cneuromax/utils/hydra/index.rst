:py:mod:`cneuromax.utils.hydra`
===============================

.. py:module:: cneuromax.utils.hydra

.. autoapi-nested-parse::

   :mod:`hydra-core` utilities.



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   cneuromax.utils.hydra.get_path
   cneuromax.utils.hydra.get_launcher_config



Attributes
~~~~~~~~~~

.. autoapisummary::

   cneuromax.utils.hydra.fs_builds
   cneuromax.utils.hydra.pfs_builds


.. py:function:: get_path(clb: collections.abc.Callable[Ellipsis, Any]) -> str

   Returns the path to the input callable.

   :param clb: The callable to get the path to.

   :returns: The path to the input callable.


.. py:function:: get_launcher_config() -> hydra_plugins.hydra_submitit_launcher.config.LocalQueueConf | hydra_plugins.hydra_submitit_launcher.config.SlurmQueueConf

   Retrieves/validates this job's :mod:`hydra-core` launcher config.

   :returns: The launcher config.


.. py:data:: fs_builds

   

.. py:data:: pfs_builds

   

