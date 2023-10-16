# This Containerfile (a.k.a. Dockerfile) is used to build the Docker image
# shared by all CNeuroMax projects. It installs all of the dependencies
# but does not install CNeuroMax itself, for development purposes.

# ~ CUDA + cuDNN on Ubuntu ~ #
FROM nvcr.io/nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

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
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e cneuromax \
    && pip uninstall -y cneuromax
