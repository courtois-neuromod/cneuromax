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

    The length of this operation depends on how much of your existing
    cache can be reused & other factors like disk utilization. On
    ``/scratch/`` of the Béluga cluster, as of March 2024, this ranges
    from 10 minutes to an hour.

.. code-block:: bash

    module load apptainer && apptainer build ${SCRATCH}/cneuromax.sif \
        docker://cneuromod/cneuromax:latest

Make sure to re-run this command whenever you modify the Dockerfile
and want to make use of the latest changes.
