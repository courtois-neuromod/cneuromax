On a Slurm cluster
==================

Apptainer (previously called Singularity) is a generally available alternative
to Docker on HPC clusters. We automate the process of building the CNeuroMax
Apptainer image for CNeuroMod team members through our GitHub Actions.

The latest image is available on Ginkgo at the following location:

.. code-block:: bash

    /scratch/cneuromax/cneuromax.sif

The process of setting up SSH keys and the required tunneling is beyond the
scope of this documentaion. Reach out to seasoned team members if you need
assistance.
