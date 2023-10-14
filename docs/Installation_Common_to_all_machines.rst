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

.. note::

    If you have administrator rights on the machine, in order to not have to
    specify the ``-u origin <BRANCH_NAME>`` flag when pushing/pulling to the
    repository (which forces you to keep the branch name in mind), you can
    optionally enable git's ``autoSetupRemote`` option.

    First, make sure your git version is at least 2.37.

    .. code-block:: bash

        git --version

    If not, you can update it with the following command.

    .. code-block:: bash

        sudo add-apt-repository ppa:git-core/ppa && \
            sudo apt update && \
            sudo apt install -y git

    Finally, enable the ``autoSetupRemote`` option.

    .. code-block:: bash

        git config --global push.autoSetupRemote true

3. Define the ``CNEUROMAX_PATH`` variable
-----------------------------------------

.. code-block:: bash

    echo -e "\nexport CNEUROMAX_PATH=${PWD}/cneuromax" >> ~/.bashrc \
        && source ~/.bashrc

You can now move on to either the Ubuntu or Slurm specific steps.
