# This Containerfile (a.k.a. Dockerfile) is used to build the Docker image
# shared by all CNeuroMax projects. It installs all of the dependencies
# but does not install CNeuroMax itself, for development purposes.

# ~ CUDA + cuDNN on Ubuntu ~ #
FROM nvcr.io/nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Prevents `apt install python3-opencv` from prompting geographical details
ENV DEBIAN_FRONTEND=noninteractive

# Prevents Python from creating __pycache__ and .pyc files in project folder
ENV PYTHONPYCACHEPREFIX=/.cache/python/

# Install system packages
RUN apt update && apt install -y \
    # OpenMPI
    libopenmpi-dev \
    # UCX for InfiniBand
    libucx0 \
    # Python dev version to get header files for mpi4py
    python3-dev \
    # Python package manager
    python3-pip \
    # OpenCV requires system packages, best to install through Ubuntu
    python3-opencv \
    # To pip install GitHub packages
    git \
    # Java (to build our fork of Hydra)
    default-jre \
    # Clean up
    && rm -rf /var/lib/apt/lists/*

# Add the pyproject.toml and cneuromax folder to the container
ADD pyproject.toml /cneuromax/pyproject.toml
ADD cneuromax /cneuromax/cneuromax

# Install Python dependencies
RUN pip install --user --no-cache-dir --upgrade pip \
    && pip install --user --no-cache-dir -e cneuromax \
    && pip uninstall -y cneuromax
