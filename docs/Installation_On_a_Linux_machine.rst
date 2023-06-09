On a Linux machine
==================

1. Install the NVIDIA driver
----------------------------

.. note::

    `Official NVIDIA driver installation guide
    <https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html>`_.

.. code-block:: console

    $ # Example (latest version as of Apr 2023)
    $ sudo apt install -y nvidia-driver-525

2. Install Podman
-----------------

.. note::

    `Official Podman installation guide
    <https://podman.io/getting-started/installation>`_.

.. code-block:: console

    $ sudo apt install -y podman

3. Install the NVIDIA Container Toolkit
---------------------------------------

Follow the Ubuntu section of the following `installation guide
<https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#id12>`_.

.. note::

    `Official NVIDIA Container Toolkit installation guide
    <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#id9>`_.

4. Pull the Podman image
------------------------

.. code-block:: console

    $ podman pull docker://maximilienlc/cneuroml:deps_only-all_deps-latest
