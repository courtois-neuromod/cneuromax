:py:mod:`cneuromax.utils.annotations`
=====================================

.. py:module:: cneuromax.utils.annotations

.. autoapi-nested-parse::

   Type annotations validator through :mod:`beartype`.



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   cneuromax.utils.annotations.not_empty
   cneuromax.utils.annotations.equal
   cneuromax.utils.annotations.one_of
   cneuromax.utils.annotations.ge
   cneuromax.utils.annotations.gt
   cneuromax.utils.annotations.le
   cneuromax.utils.annotations.lt



.. py:function:: not_empty() -> beartype.vale._core._valecore.BeartypeValidator

   Makes sure the string is not empty.

   :returns: The corresponding validator.


.. py:function:: equal(element: object) -> beartype.vale._core._valecore.BeartypeValidator

   Makes sure the value is equal to the input argument.

   :param element: The object to compare against.

   :returns: The corresponding validator.


.. py:function:: one_of(*elements: object) -> beartype.vale._core._valecore.BeartypeValidator

   Makes sure the value is one of the input arguments.

   Used to replace :class:`typing.Literal` which is not supported by
   :mod:`omegaconf`.

   :param elements: The objects to compare against.

   :returns: The corresponding validator.


.. py:function:: ge(val: float) -> beartype.vale._core._valecore.BeartypeValidator

   Makes sure the input is greater of equal than `val`.

   :param val: The value to compare against.

   :returns: The corresponding validator.


.. py:function:: gt(val: float) -> beartype.vale._core._valecore.BeartypeValidator

   Makes sure the input is greater than `val`.

   :param val: The value to compare against.

   :returns: The corresponding validator.


.. py:function:: le(val: float) -> beartype.vale._core._valecore.BeartypeValidator

   Makes sure the input is less or equal than `val`.

   :param val: The value to compare against.

   :returns: The corresponding validator.


.. py:function:: lt(val: float) -> beartype.vale._core._valecore.BeartypeValidator

   Makes sure the input is less than `val`.

   :param val: The value to compare against.

   :returns: The corresponding validator.


