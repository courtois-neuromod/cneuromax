# ----------------------------------------------------------------------------#
# This Dockerfile (a.k.a. Dockerfile) is used to build the Docker image
# (which can turn into an Apptainer image) shared by all CNeuroMax projects.
# It installs all of the dependencies but does not install CNeuroMax itself,
# for development purposes.
# ----------------------------------------------------------------------------#
# PyTorch (w/ ecosystem) + CUDA + cuDNN + MPI + UCX + Python (w/ pip & headers)
FROM nvcr.io/nvidia/pytorch:24.04-py3
# Prevents Python from creating __pycache__/ and .pyc/ folders in the project
# folder
ENV PYTHONPYCACHEPREFIX=/.cache/python/
# Fixes a bug where the container fails building the wheel for mpi4py
ENV SETUPTOOLS_USE_DISTUTILS=local
# Install system packages
#                                Enable add-apt-repository
RUN apt update && apt install -y software-properties-common && \
    #                  To upgrade Git
    add-apt-repository ppa:git-core/ppa && apt install -y \
    # To run the follow-up `git config` commands
    git \
    # Java to build our fork of Hydra
    default-jre \
    # Audio libraries
    ffmpeg \
    sox \
    libavdevice-dev \
    # Required by soundfile
    libffi7 \
    # Clean up
    && rm -rf /var/lib/apt/lists/*
# Install torchaudio
COPY install_torchaudio_latest.sh /install_torchaudio_latest.sh
RUN /bin/bash /install_torchaudio_latest.sh
# To not have to specify `-u origin <BRANCH_NAME>` when pushing
RUN git config --global push.autoSetupRemote true
# To push the current branch to the existing same name branch
RUN git config --global push.default current
# Add the pyproject.toml and cneuromax folder to the container
ADD pyproject.toml /cneuromax/pyproject.toml
# Install Python dependencies
RUN pip install --no-cache-dir -e /cneuromax \
    && pip uninstall -y cneuromax
# Note: MPI UCX warnings on Rootless Docker
