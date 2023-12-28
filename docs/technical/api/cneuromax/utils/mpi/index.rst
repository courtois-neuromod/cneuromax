:py:mod:`cneuromax.utils.mpi`
=============================

.. py:module:: cneuromax.utils.mpi

.. autoapi-nested-parse::

   :mod:`mpi4py` utilities.



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   cneuromax.utils.mpi.retrieve_mpi_variables



.. py:function:: retrieve_mpi_variables() -> tuple[mpi4py.MPI.Comm, Annotated[int, ge(0)], Annotated[int, ge(1)]]

   Retrieves MPI variables from the MPI runtime.

   :returns: The MPI communicator.
             rank: The rank of the current process.
             size: The total number of processes.
   :rtype: comm


