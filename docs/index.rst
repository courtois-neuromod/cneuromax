#######################################
Welcome to the CNeuroMax documentation!
#######################################

CNeuroMax is a framework for large-scale training of machine learning models
(Deep Learning + Neuroevolution) with an emphasis on easy deployment in
high-performance computing environments (HPC). CNeuroMax aims to:

1. **Facilitate the configuration of complex models and training runs through
   tools like:** Hydra, Hydra-Zen, Lightning etc.

2. **Automate much of the process of deployment in a high-performance computing
   environment:** creating SLURM scripts, monitoring SLURM jobs, setting up
   virtual environments, upgrading packages, tuning hyperparameters, etc.

3. **Provide a space for researchers to share their code and experiment
   results:** a central repository with a common solid and well-tested
   object-oriented structure for Lightning Modules, subdirectories for each
   experiment, Weights & Biases working both locally and on SLURM with support
   for team-shared logging etc.

4. **Offer optional tools to strengthen code quality and reproducibility:**
   code linting (Ruff) and formatting (Black), unit testing (pytest), static
   (Mypy) & dynamic (Beartype) type checking that supports tensor shapes and
   types (jaxtyping, nptyping), development containers (Dev Containers),
   documentation auto-generation and auto-deployment, precommit hooks etc.

The cneuromax library is developed in the context of the `Courtois Project on
Neuronal Modelling <https://cneuromod.ca>`_, also known as CNeuroMod.
Launched in 2018, CNeuroMod aims to create more human-like AI models by
training them to emulate both human brain dynamics and behaviour.
Phase I (2018-2023) of the project saw large-scale data acquisition and
preliminary unimodal modelling. Phase II (2023-2027) of the project aims to
create multimodal phantom models of the subjects.
The cneuromax library is used as the framework to train these phantom models.

********
Contents
********

.. toctree::
   :maxdepth: 2

   Installation
   Execution
   Contribution
