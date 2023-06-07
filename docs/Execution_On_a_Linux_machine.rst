.. _execution_on_a_linux_machine:

On a Linux machine
==================

Run a python script
-------------------

.. code-block:: console

    $ # Example of a simple MNIST training run
    $ podman run -e CNEUROML_PATH=${CNEUROML_PATH} -e PYTHONPATH=${PYTHONPATH}:${CNEUROML_PATH} \
        -v ${CNEUROML_PATH}:${CNEUROML_PATH} -w ${CNEUROML_PATH} cneuroml:deps_only-all_deps-latest \
        python3 -m cneuroml.train.dl experiment=visual/tabular/classification/mnist/mlp

Run a notebook
--------------

From your own machine create a SSH tunnel to the running machine.

.. code-block:: console

   $ # Fill in the appropriate values
   $ ssh USER@ADDRESS -NL 8888:localhost:8888

Run the lab.

.. code-block:: console

    $ podman run -e CNEUROML_PATH=${CNEUROML_PATH} -e PYTHONPATH=${PYTHONPATH}:${CNEUROML_PATH} \
        -v ${CNEUROML_PATH}:${CNEUROML_PATH} -w ${CNEUROML_PATH} cneuroml:deps_only-all_deps-latest \
        jupyter-lab --allow-root --ip 0.0.0.0 --port 8888
