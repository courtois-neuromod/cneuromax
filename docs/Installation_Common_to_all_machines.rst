Common to all machines
======================

The following commands are meant to set-up the repository.

1. Move to the desired containing folder
----------------------------------------

.. note::

    We recommend, in the ``Contribution`` section, to make use of
    Dropbox/Maestral if you plan to alter the code on different machines.

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

4. Weights & biases
-------------------

Weights & Biases is a tool to log experiments. In order to make use of it, you
will need to create an account on their `website <https://www.wandb.com/>`_ and
generate an API key. Once you have done so, create a file called
``WANDB_KEY.txt`` at the root of the repository and paste your API key in it.

You can now log experiments to your personal account. To log experiments to the
CNeuroMod account, contact any existing team member with your email address to
be added to the team list.
