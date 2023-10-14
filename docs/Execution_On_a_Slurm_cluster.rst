On a Slurm cluster
==================

Run a python script
-------------------

.. code-block:: bash

    # Example of a simple MNIST training run
    module load apptainer && cd ${CNEUROMAX_PATH} && export PYTHONPATH=${PYTHONPATH}:${CNEUROMAX_PATH} && \
        export APPTAINERENV_APPEND_PATH=/opt/software/slurm/bin:/cvmfs/soft.computecanada.ca/easybuild/software/2020/Core/apptainer/1.1.8/bin && \
        apptainer exec -B /etc/passwd -B /etc/slurm/ -B /opt/software/slurm -B /usr/lib64/libmunge.so.2 \
                       -B /cvmfs/soft.computecanada.ca/easybuild/software/2020/Core/apptainer/1.1.8/bin/apptainer \
                       -B /var/run/munge/ --env LD_LIBRARY_PATH=/opt/software/slurm/lib64/slurm  -B $CNEUROMAX_PATH $SCRATCH/cneuromax.sif \
                       python3 -m cneuromax.fitting.deeplearning -m task=classify_mnist/mlp_beluga

Run Jupyter-lab
---------------

From your own machine create a SSH tunnel to the compute node.

.. code-block:: bash

    # Fill in the appropriate values
    sshuttle --dns -Nr USER@ADDRESS:8888

Run the lab.

.. code-block:: bash

    # Fill in the appropriate values TODO FIX
    # salloc --account=ACCOUNT bash -c "module load apptainer && cd ${CNEUROMAX_PATH} && \
    #    apptainer exec -v ${CNEUROMAX_PATH}:${CNEUROMAX_PATH} \
    #    cneuromod/cneuromax:latest jupyter-lab --allow-root --ip $(hostname -f) --port 8888"
