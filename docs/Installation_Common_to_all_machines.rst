.. _installation_common_to_all_machines:

Common to all machines
======================

1. Setup the repository
-----------------------

Move to the desired containing folder.

.. code-block:: console

    $ # Example
    $ cd ${HOME}

Clone the repository.

.. code-block:: console

    $ git clone git@github.com:courtois-neuromod/cneuroml.git

Define a persisting ``CNEUROML_PATH`` user-variable on your system.

.. code-block:: console

    $ echo -e "\nCNEUROML_PATH=${PWD}/cneuroml" >> ~/.bashrc && source ~/.bashrc

You can now move on to either the Linux or Slurm specific steps.
