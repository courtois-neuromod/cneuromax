# ----------------------------------------------------------------------------#
# This Dockerfile (a.k.a. Dockerfile) is used to build the Docker image
# (which can turn into an Apptainer image) shared by all CNeuroMax projects.
# It installs all of the dependencies but does not install CNeuroMax itself,
# for development purposes.
# ----------------------------------------------------------------------------#
# ~ CUDA + cuDNN on Ubuntu ~ #
FROM nvcr.io/nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
# Prevents Python from creating __pycache__/ and .pyc/ folders in the project
# folder
ENV PYTHONPYCACHEPREFIX=/.cache/python/
# Fixes a bug where the container fails building the wheel for mpi4py
ENV SETUPTOOLS_USE_DISTUTILS=local
# Install system packages
RUN apt update && apt install -y \
    # For git pip install
    git \
    # OpenMPI
    libopenmpi-dev \
    # UCX for InfiniBand
    libucx0 \
    # Python dev version to get header files for mpi4py + pip
    python3-dev \
    python3-pip \
    # Java to build our fork of Hydra
    default-jre \
    # Audio libraries
    ffmpeg \
    sox \
    libavdevice-dev \
    # Required by soundfile
    libffi7 \
    # Required for uv
    curl \
    # Clean up
    && rm -rf /var/lib/apt/lists/*
RUN ln -s /usr/bin/python3 /usr/bin/python
# Add the pyproject.toml and cneuromax folder to the container
ADD pyproject.toml /cneuromax/pyproject.toml
# Install Python dependencies
RUN pip install uv \
    && uv pip install --preview --system --no-cache-dir -e /cneuromax \
    && uv pip uninstall --preview --system cneuromax
