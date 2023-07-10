On a Linux machine
==================

Run a python script
-------------------

.. code-block:: console

    $ # Example of a simple MNIST training run
    $ podman run -e CNEUROMAX_PATH=${CNEUROMAX_PATH} \
                 -e PYTHONPATH=${PYTHONPATH}:${CNEUROMAX_PATH} \
                 -v ${CNEUROMAX_PATH}:${CNEUROMAX_PATH} \
                 -w ${CNEUROMAX_PATH} cneuromax:deps_only-all_deps-latest \
                 python3 -m cneuromax.dl task=visual/tabular/classification/mnist/mlp

Run a notebook
--------------

From your own machine create a SSH tunnel to the running machine.

.. code-block:: console

   $ # Fill in the appropriate values
   $ ssh USER@ADDRESS -NL 8888:localhost:8888

Run the lab.

.. code-block:: console

    $ podman run -e CNEUROMAX_PATH=${CNEUROMAX_PATH} \
                 -e PYTHONPATH=${PYTHONPATH}:${CNEUROMAX_PATH} \
                 -v ${CNEUROMAX_PATH}:${CNEUROMAX_PATH} \
                 -w ${CNEUROMAX_PATH} cneuromax:deps_only-all_deps-latest \
                 jupyter-lab --allow-root --ip 0.0.0.0 --port 8888
