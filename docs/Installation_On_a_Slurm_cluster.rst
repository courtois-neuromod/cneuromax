On a Slurm cluster
==================

1. Install the experiment manager packages
------------------------------------------

.. code-block:: console

    $ module load python/3.10
    $ python3 -m venv ${CNEUROMAX_PATH}/venv
    $ . ${CNEUROMAX_PATH}/venv/bin/activate
    $ pip install -r ${CNEUROMAX_PATH}/pipreqs/0_experiment_manager.txt

2. Prepare the Podman image for use on the cluster
--------------------------------------------------

Load Podman.

.. code-block:: console

    $ module load podman

Set-up the Podman storage configuration file.

.. code-block:: console

    $ mkdir ~/.config/containers
    $ echo -e "[storage]\ndriver = \"overlay\"\n \
        graphRoot = \"$SLURM_TMPDIR/$SCRATCH/containers/storage\"" > \
            ~/.config/containers/storage.conf

Pull the image.

.. code-block:: console

    $ module load podman \
        && podman pull docker://maximilienlc/cneuromax:deps-run-latest

Compress the Podman container folder.

.. code-block:: console

    $ tar -cvf ${SCRATCH}/containers.tar ${SCRATCH}/containers/

Delete ``bolt_state.db``.

.. code-block:: console

    $ rm ${SCRATCH}/containers/storage/libpod/bolt_state.db

The image is now ready for use on the compute nodes. At the beginning of every
Slurm job, we will (all through Hydra/Submitit) copy it over to the local
drive, decompress it, and run the necessary Podman + NVIDIA Container Toolkit
commands to start the container (which on average takes around 20 seconds).
