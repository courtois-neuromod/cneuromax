Common to all machines
======================

The following commands are meant to set-up the repository.

1. Move to the desired containing folder
----------------------------------------

.. code-block:: bash

    # Examples
    cd ${HOME}/Dropbox/
    cd ${SCRATCH}/Dropbox/

2. Clone the repository
-----------------------

.. code-block:: bash

    git clone git@github.com:courtois-neuromod/cneuromax.git


3. Define the ``CNEUROMAX_PATH`` variable
-----------------------------------------

.. code-block:: bash

    echo -e "\nexport CNEUROMAX_PATH=${PWD}/cneuromax" >> ~/.bashrc \
        && source ~/.bashrc

You can now move on to either the Ubuntu or Slurm specific steps.
