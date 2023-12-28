:py:mod:`cneuromax.fitting.deeplearning.nnmodule.mlp`
=====================================================

.. py:module:: cneuromax.fitting.deeplearning.nnmodule.mlp

.. autoapi-nested-parse::

   :class:`MLP` & its config dataclass.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   cneuromax.fitting.deeplearning.nnmodule.mlp.MLPConfig
   cneuromax.fitting.deeplearning.nnmodule.mlp.MLP




.. py:class:: MLPConfig


   Holds :class:`MLP` config values.

   :param dims: List of dimensions for each layer.
   :param p_dropout: Dropout probability.

   .. py:attribute:: dims
      :type: list[int]

      

   .. py:attribute:: p_dropout
      :type: Annotated[float, ge(0), lt(1)]
      :value: 0.0

      


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



