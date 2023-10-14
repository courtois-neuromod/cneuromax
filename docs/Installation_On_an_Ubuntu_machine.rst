On an Ubuntu machine
====================

.. note::

    Skip to Step 4 if you do not have administrator privileges.

1. Install Docker or Podman
---------------------------

.. note::

    We suggest installing Docker if you meet any of the following conditions:

    - This is your own machine and plan to contribute to the library
      (Development Containers are smoother with Docker than Podman).
    - You are an administrator of this machine, do not want to install Podman
      and are fine with adding users to the ``docker`` group. See this `link
      <https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user>`_
      for security implications.

    Otherwise, install Podman.

**Option A) Install Docker**

.. note::

    `Official Docker installation guide
    <https://docs.docker.com/engine/install/ubuntu/>`_.

.. code-block:: bash

    sudo apt install -y docker.io

**Option B) Install Podman**

.. note::

    `Official Podman installation guide
    <https://podman.io/getting-started/installation>`_.

.. code-block:: bash

    sudo apt install -y podman

1. Install the NVIDIA driver
----------------------------

.. note::

    Skip this step if your machine does not have an NVIDIA GPU.

.. note::

    `Official NVIDIA driver installation guide
    <https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html>`_.

.. code-block:: bash

    # Example (latest version as of Apr 2023)
    sudo apt install -y nvidia-driver-525

3. Install the NVIDIA Container Toolkit
---------------------------------------

.. note::

    Skip this step if your machine does not have an NVIDIA GPU.

Follow the following `installation guide
<https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>`_.

4. Pull the image
-----------------

.. code-block:: bash

    # Substitute `docker` with `podman` if you installed Podman.
    docker pull docker.io/cneuromod/cneuromax:latest
