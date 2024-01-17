On a Slurm cluster
==================

1. Update the ``APPTAINER_CACHEDIR`` variable
---------------------------------------------

.. note::

    This step assumes that storing the Apptainer cache folder in
    your ``${HOME}`` directory is less favorable than storing it in
    your ``${SCRATCH}`` directory (which is the case on Béluga).

.. code-block:: bash

    echo -e "\nexport APPTAINER_CACHEDIR=${SCRATCH}/.apptainer/" >> ~/.bashrc \
        && source ~/.bashrc

2. Build the image
------------------

.. note::

    This command takes around 5-10 minutes to complete on the Béluga cluster.

.. code-block:: bash

    module load apptainer && apptainer build ${SCRATCH}/cneuromax.sif \
        docker://cneuromod/cneuromax:latest

Make sure to re-run this command whenever you modify the Dockerfile
and want to make use of the latest changes.
