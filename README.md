# CNeuroMax

[![container-build-push](
    https://github.com/courtois-neuromod/cneuromax/actions/workflows/container-build-push.yaml/badge.svg)](
        https://github.com/courtois-neuromod/cneuromax/actions/workflows/container-build-push.yaml)
[![docs-build-push](
    https://github.com/courtois-neuromod/cneuromax/actions/workflows/docs-build-push.yaml/badge.svg)](
        https://github.com/courtois-neuromod/cneuromax/actions/workflows/docs-build-push.yaml)
[![format-lint](
    https://github.com/courtois-neuromod/cneuromax/actions/workflows/format-lint.yaml/badge.svg?event=push)](
        https://github.com/courtois-neuromod/cneuromax/actions/workflows/format-lint.yaml)
[![typecheck-unittest](
    https://github.com/courtois-neuromod/cneuromax/actions/workflows/typecheck-unittest.yaml/badge.svg?event=push)](
        https://github.com/courtois-neuromod/cneuromax/actions/workflows/typecheck-unittest.yaml)
[![codecov](
    https://codecov.io/gh/courtois-neuromod/cneuromax/branch/main/graph/badge.svg?token=AN8GLFP9CB)](
        https://codecov.io/gh/courtois-neuromod/cneuromax)
[![code style: black](
    https://img.shields.io/badge/code%20style-black-000000.svg)](
        https://github.com/psf/black)

Full documentation available at [https://courtois-neuromod.github.io/cneuromax](
    https://courtois-neuromod.github.io/cneuromax/).

<h2>Introduction</h2>

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
   types (jaxtyping), development containers (Dev Containers), documentation
   auto-generation and auto-deployment, precommit hooks etc.

<h2>Repository structure:</h1>

```
cneuromax/
├─ .github/                  <-- Config files for GitHub automation (tests, containers, etc)
├─ cneuromax/                <-- Machine Learning code
│  ├─ fitting/               <-- ML model fitting code
│  │  ├─ deeplearning/       <-- Deep Learning code
│  │  │  ├─ datamodule/      <-- Lightning DataModules
│  │  │  ├─ litmodule/       <-- Lightning Modules
│  │  │  ├─ nnmodule/        <-- PyTorch Modules & Hydra configs
│  │  │  ├─ utils/           <-- Deep Learning utilities
│  │  │  ├─ __init__.py      <-- Stores Deep Learning Hydra configs
│  │  │  ├─ __main__.py      <-- Entrypoint when calling `python cneuromax.fitting.deeplearning`
│  │  │  ├─ config.yaml      <-- Default Deep Learning Hydra configs & settings
│  │  │  └─ fit.py           <-- Deep Learning fitting & Hydra config
│  │  ├─ neuroevolution/     <-- Neuroevolution code
│  │  │  ├─ agent/           <-- Neuroevolution agents (encapsulate networks)
│  │  │  ├─ net/             <-- Neuroevolution networks
│  │  │  ├─ space/           <-- Neuroevolution spaces (where agents get evaluated)
│  │  │  ├─ utils/           <-- Neuroevolution utilities
│  │  │  ├─ __init__.py      <-- Stores Neuroevolution Hydra configs
│  │  │  ├─ __main__.py      <-- Entrypoint when calling `python cneuromax.fitting.neuroevolution`
│  │  │  ├─ config.yaml      <-- Default Neuroevolution Hydra configs & settings
│  │  │  └─ fit.py           <-- Neuroevolution fitting & Hydra config
│  │  ├─ __init__.py         <-- Stores Fitting Hydra configs
│  │  ├─ __main__.py         <-- Entrypoint when calling `python cneuromax.fitting`
│  │  └─ config.py           <-- Base Structured Hydra fitting config & utilities
│  ├─ serving/               <-- Contains the code to create applications (cozmo inference, etc)
│  ├─ task/                  <-- Contains the Deep Learning tasks
│  │  │
│  │  │                          ******************************************
│  │  └─ my_new_task/        <-- *** Your new Deep Learning task folder ***
│  │     ├─ __init__.py      <-- ********** Your Hydra Configs ************
│  │     ├─ datamodule.py    <-- ******* Your Lightning DataModule ********
│  │     ├─ litmodule.py     <-- ********* Your Lightning Module **********
│  │     ├─ nnmodule.py      <-- ********** Your PyTorch Module ***********
│  │     └─ config.yaml      <-- ****** Your Hydra configuration file *****
│  │                             ******************************************
│  │
│  ├─ utils/                 <-- CNeuroMax utilities
│  ├─ __init__.py            <-- Sets up Beartype, logs in W&B, etc
│  └─ config.py              <-- Base Structured Hydra config & utilities
├─ docs/                     <-- Documentation files
├─ .devcontainer.json        <-- VSCode container development config
├─ .gitignore                <-- Files to not track with Git/GitHub
├─ .pre-commit-config.yaml   <-- Pre-"git commit" actions config (format, lint, etc)
├─ .yamllint.yaml            <-- YAML files config
├─ Containerfile             <-- To build the Docker image
├─ LICENSE                   <-- MIT License file
├─ README.md                 <-- Repository description file
├─ pyproject.toml            <-- Python code & dependencies config
└─ renovate.json             <-- Renovate Bot config (keeps dependencies up-to-date)
```

CNeuroMax is developed in the context of the
[Courtois Project on Neuronal Modelling](https://cneuromod.ca), also known as
CNeuroMod. Launched in 2018, CNeuroMod aims to create more human-like AI models
by training them to emulate both human brain dynamics and behaviour. Phase I
(2018-2023) of the project saw large-scale data acquisition and preliminary
unimodal modelling. Phase II (2023-2027) of the project aims to create
multimodal phantom models of the subjects. CNeuroMax is used as the framework
to train these phantom models.
