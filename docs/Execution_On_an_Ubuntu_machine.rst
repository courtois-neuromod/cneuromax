On an Ubuntu machine
====================

.. note::

    Replace

    .. code-block:: bash

        docker run --privileged --gpus all

    with

    .. code-block:: bash

        podman run --security-opt=label=disable --device=nvidia.com/gpu=all

    if you are using Podman.

    Make sure to remove the above flags if you are not using a GPU.

.. note:: 
    By default, the Podman container's timezone is either set to the timezone of the local 
    where it is built or to the UTC. To change the timezone of the container, add
    ``--tz=local`` flag to the execution command above.

Run a python script
-------------------


.. note::

    Run ``cd ${CNEUROMAX_PATH}/cneuromax`` before the following command to get
    tab completion for the ``task`` argument.

.. note::
    By default, Docker's local timezone is set to UTC, and all the Docker operations
    will run according to UTC regardless of the host machine's timezone. If you 
    would like to set it to a specific timezone, please add ``--env TZ= YOUR_TIMEZONE`` 
    flag to the execution command below.

.. code-block:: bash

    # Example of a simple MNIST training run
    docker run --privileged --gpus all --rm -e CNEUROMAX_PATH=${CNEUROMAX_PATH} \
               -e PYTHONPATH=${PYTHONPATH}:${CNEUROMAX_PATH} \
               -v ${CNEUROMAX_PATH}:${CNEUROMAX_PATH} -v /dev/shm:/dev/shm \
               -w ${CNEUROMAX_PATH} cneuromod/cneuromax:latest \
               python -m cneuromax project=classify_mnist task=mlp


Run a notebook
--------------

From your own machine create a SSH tunnel to the running machine.

.. code-block:: bash

   # Example
   ssh MY_USER@123.456.7.8 -NL 8888:localhost:8888

Run the lab.

.. code-block:: bash

    docker run --rm -e CNEUROMAX_PATH=${CNEUROMAX_PATH} \
               -e PYTHONPATH=${PYTHONPATH}:${CNEUROMAX_PATH} \
               -v ${CNEUROMAX_PATH}:${CNEUROMAX_PATH} \
               -w ${CNEUROMAX_PATH} cneuromod/cneuromax:latest \
               jupyter-lab --allow-root --ip 0.0.0.0 --port 8888
