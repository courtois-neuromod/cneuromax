On a Slurm cluster
==================

Run a python script
-------------------

Activate the previously installed virtual environment.

.. code-block:: console

    $ cd ${CNEUROML_PATH}
    $ . ${CNEUROML_PATH}/venv/bin/activate

Run the library.

.. code-block:: console

    $ # Example of a simple MNIST training run
    $ python3 -m cneuroml.train.dl experiment=visual/tabular/classification/mnist/mlp_beluga

Run Jupyter-lab
---------------

From your own machine create a SSH tunnel to the compute node.

.. code-block:: console

    $ # Fill in the appropriate values
    $ sshuttle --dns -Nr USER@ADDRESS:8888

Run the lab.

.. code-block:: console

    $ # Fill in the appropriate values
    $ salloc --account=ACCOUNT bash -c "module load podman; \
        nvidia-ctk cdi generate --output=/var/tmp/cdi/nvidia.yaml; \
        mkdir -p ${SLURM_TMPDIR}/${SCRATCH}; \
        cp ${SCRATCH}/containers.tar ${SLURM_TMPDIR}/${SCRATCH}/.; \
        tar -xf ${SLURM_TMPDIR}/${SCRATCH}/containers.tar -C ${SLURM_TMPDIR}/${SCRATCH}/.; \
        podman run -w ${CNEUROML_PATH} -v ${CNEUROML_PATH}:${CNEUROML_PATH} \
        cneuroml:deps_only-all_deps-latest jupyter-lab --allow-root --ip $(hostname -f) --port 8888"
