:py:mod:`cneuromax.fitting.deeplearning.nnmodule`
=================================================

.. py:module:: cneuromax.fitting.deeplearning.nnmodule

.. autoapi-nested-parse::

   Common :class:`torch.nn.Module`\s (+:mod:`hydra-core` cfg store).



Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   mlp/index.rst


Package Contents
----------------

Classes
~~~~~~~

.. autoapisummary::

   cneuromax.fitting.deeplearning.nnmodule.MLP
   cneuromax.fitting.deeplearning.nnmodule.MLPConfig



Functions
~~~~~~~~~

.. autoapisummary::

   cneuromax.fitting.deeplearning.nnmodule.store_nnmodule_configs



.. py:class:: MLP(config: MLPConfig, activation_fn: torch.nn.Module)




   Multi-layer perceptron (MLP).

   Allows for a variable number of layers, activation functions, and
   dropout probability.

   :param config: See :class:`MLPConfig`.
   :param activation_fn: Activation function.

   .. attribute:: model

      

      :type: :class:`torch.nn.Sequential`

   .. py:method:: forward(x: jaxtyping.Float[torch.Tensor,  batch_size *d_input]) -> jaxtyping.Float[torch.Tensor,  batch_size output_size]

      Flattens input's dimensions and passes it through the model.

      .. note::

         This MLP isn't suitable for cases where the output
         is multidimensional.

      :param x: The input data batch.

      :returns: The output batch.



.. py:class:: MLPConfig


   Holds :class:`MLP` config values.

   :param dims: List of dimensions for each layer.
   :param p_dropout: Dropout probability.

   .. py:attribute:: dims
      :type: list[int]

      

   .. py:attribute:: p_dropout
      :type: Annotated[float, ge(0), lt(1)]
      :value: 0.0

      


.. py:function:: store_nnmodule_configs(cs: hydra.core.config_store.ConfigStore) -> None

   Stores :mod:`hydra-core` ``litmodule/nnmodule`` group configs.

   Config names: ``mlp``.

   :param cs: See :paramref:`~cneuromax.__init__.store_task_configs.cs`.


